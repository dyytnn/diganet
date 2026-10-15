from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.pyramid_context import PyramidContext
from ..modules.boundary_refinement import BoundaryGuidedRefiner
from ..modules.boundary_patch_refinement import BoundaryProximityRefiner
from ..modules.bmask_boundary import BMaskBoundaryBranch
from ..modules.point_rend import PointRendHead
from ..modules.deep_strip import DeepStripRefiner
from ..modules.refine_mask import RefineMaskHead
from ..modules.mask_transfiner import MaskTransfinerLite
from ..modules.gated_texture import EntropyGatedRefiner
from ..modules.segfix import SegFixHead
from ..modules.cascade_psp import CascadePSPLite
from ..modules.contour_refinement import PolarContourRefiner
from ..modules.adaptive_skip_fusion import AdaptiveSkipFusion
from ..modules.frequency_multiscale import FrequencyMultiScale
from ..modules.convnext_block import ConvNeXtBlock
from ..modules.residual_encoder import ResidualEncoderBlock
from ..modules.adaptive_receptive_field import AdaptiveReceptiveField
from ..modules.anti_alias_downsample import AntiAliasedDownsample
from ..modules.cnn_ssm_block import CNNSSMBlock
from ..modules.token_moe_adapter import TokenMoEAdapter
from ..modules.dynamic_snake_conv import DynamicSnakeConv
from ..modules import (
    ADGBCBlock, BilateralGraphBlock, BiFormerBlock, CARAFEBlock, CBAMBlock,
    CENetBlock, ConvNeXtV2Block, CoordinateAttention, DCNv4Block, DLKABlock,
    DySample, ECABlock, MFMSABlock, MambaLiteFusion, MaxViTBlock, MedNeXtBlock,
    MSCABlock, ODConvBlock, PhaseAlignedFusion, PointRendRefiner, SEBlock,
    StageConditionedGraph, TokKANBlock,
)


LITERATURE_BLOCKS = {
    "mednext_block": MedNeXtBlock,
    "convnext_v2_block": ConvNeXtV2Block,
    "msca_block": MSCABlock,
    "dlka_block": DLKABlock,
    "dcnv4_block": DCNv4Block,
    "odconv_block": ODConvBlock,
    "se_block": SEBlock,
    "cbam_block": CBAMBlock,
    "eca_block": ECABlock,
    "coordinate_attention": CoordinateAttention,
    "mfmsa_block": MFMSABlock,
    "cenet_block": CENetBlock,
    "ad_gbc_block": ADGBCBlock,
    "mambalite_fusion": MambaLiteFusion,
    "dysample": DySample,
    "carafe_block": CARAFEBlock,
    "pointrend_refiner": PointRendRefiner,
    "maxvit_block": MaxViTBlock,
    "biformer_block": BiFormerBlock,
    "tokkan_block": TokKANBlock,
    "bilateral_graph_block": BilateralGraphBlock,
    "phase_aligned_fusion": PhaseAlignedFusion,
}


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=False), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=False))
    def forward(self, x): return self.net(x)


class MultiModalStem(nn.Module):
    """Intensity/texture/edge branches with learnable per-sample softmax weights."""
    def __init__(self, out_channels: int):
        super().__init__(); self.branches = nn.ModuleList([nn.Conv2d(1, out_channels, 3, padding=1), nn.Conv2d(1, out_channels, 3, padding=1), nn.Conv2d(1, out_channels, 3, padding=1)]); self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 3, 1))
    def forward(self, x):
        intensity = x[:, :1]; texture = x[:, 1:2] - F.avg_pool2d(x[:, :1], 3, 1, 1); edge = F.pad(x[:, :1][:, :, :, 1:] - x[:, :1][:, :, :, :-1], (0, 1, 0, 0))
        feats = [b(z) for b, z in zip(self.branches, (intensity, texture, edge))]; weights = torch.softmax(self.gate(torch.cat((intensity, texture, edge), 1)), 1); return sum(f * weights[:, i:i+1] for i, f in enumerate(feats)), weights.flatten(1)


class PooledGlobalContext(nn.Module):
    """Efficient global context attention over a pooled bottleneck map."""
    def __init__(self, channels: int, pool_size: int = 8):
        super().__init__()
        inner_channels = max(channels // 4, 8)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.reduce = nn.Conv2d(channels, inner_channels, 1)
        self.attention = nn.MultiheadAttention(inner_channels, 4, batch_first=True)
        self.expand = nn.Conv2d(inner_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.reduce(self.pool(x))
        shape = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens, _ = self.attention(tokens, tokens, tokens)
        context = tokens.transpose(1, 2).reshape(shape)
        context = F.interpolate(self.expand(context), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + context


class ClassContextRefiner(nn.Module):
    """Refine decoder features using soft class-specific prototypes."""
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(channels, num_classes, 1)
        self.context_projection = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.classifier(features).softmax(1)
        weighted_features = torch.einsum("bkhw,bchw->bkc", probabilities, features)
        normalizer = probabilities.flatten(2).sum(-1).clamp_min(1e-6).unsqueeze(-1)
        prototypes = weighted_features / normalizer
        context = torch.einsum("bkhw,bkc->bchw", probabilities, prototypes)
        refined = features + self.gate(features) * self.context_projection(context)
        return refined, probabilities


class MATBlastUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24, num_classes=4, attention=False, multimodal=False, dropout=0.0, global_context=False, global_pool_size=8, class_context=False, deep_supervision=False, class_boundary=False, pyramid_context=False, pyramid_dilations=(1, 2, 4, 8), pyramid_reduction=4, boundary_refinement=False, boundary_hidden_channels=None, adaptive_skip_fusion=False, adaptive_skip_hidden_channels=None, frequency_multiscale=False, frequency_hidden_channels=None, frequency_dilations=(1, 2, 4), stage_conditioned_graph=False, stage_count=5, convnext_block=False, convnext_expansion=4, convnext_kernel_size=7, convnext_layer_scale=1e-6, convnext_dropout=0.0, residual_encoder_block=False, residual_expansion=2, residual_dropout=0.0, adaptive_receptive_field=False, adaptive_receptive_dilations=(1, 2, 4), anti_aliased_downsampling=False, cnn_ssm_block=False, cnn_ssm_state_ratio=4, token_moe_adapter=False, token_moe_experts=4, token_moe_expansion=2, dynamic_snake_conv=False, dynamic_snake_kernel_size=5, dynamic_snake_max_offset=2.0, bpr_refinement=False, bpr_hidden_channels=None, bpr_proximity_kernel_size=5, bmask_boundary_branch=False, bmask_hidden_channels=None, bmask_fusion="gate", pointrend=False, pointrend_coarse_level=2, pointrend_num_points=2048, pointrend_oversample_ratio=3.0, pointrend_importance_sample_ratio=0.75, pointrend_mlp_hidden=256, pointrend_mlp_layers=3, pointrend_classes_of_interest=(1, 2, 4), pointrend_subdivision_steps=2, pointrend_subdivision_num_points=8192, deep_strip=False, deep_strip_num_strips=1024, deep_strip_length=9, deep_strip_spacing_px=1.0, deep_strip_hidden_channels=64, deep_strip_classes_of_interest=(1, 2, 4), deep_strip_oversample_ratio=3.0, deep_strip_importance_sample_ratio=0.75, deep_strip_inference_num_strips=4096, refine_mask=False, refine_mask_hidden_channels=32, refine_mask_boundary_aware=True, mask_transfiner=False, transfiner_nodes_per_level=(1024, 512, 256), transfiner_d_model=64, transfiner_nhead=4, transfiner_num_layers=2, transfiner_dim_feedforward=128, transfiner_dropout=0.0, transfiner_classes_of_interest=(1, 2, 4), transfiner_oversample_ratio=3.0, transfiner_importance_ratio=0.75, segfix=False, segfix_hidden_channels=32, segfix_offset_px=2, segfix_steps=2, segfix_boundary_threshold=0.5, cascade_psp=False, cascade_hidden_channels=32, cascade_global_scale=0.5, cascade_patch_size=64, cascade_num_patches=8, cascade_classes_of_interest=(1, 2, 4), contour_refiner=None, contour_classes_of_interest=(1, 2, 4), contour_num_vertices=64, contour_iterations=3, contour_hidden_channels=64, contour_step_px=2.0, contour_ray_steps=64, contour_max_radius=0.75, contour_logit_boost=2.0, entropy_gated_refinement=False, entropy_gate_hidden_channels=None, multiscale_texture_gate=False, literature_block=None, literature_block_kwargs=None):
        super().__init__(); c = base_channels; self.multimodal = multimodal; self.deep_supervision = deep_supervision; self.class_boundary = class_boundary; self.boundary_refinement = boundary_refinement; self.adaptive_skip_fusion = adaptive_skip_fusion; self.stage_conditioned_graph = stage_conditioned_graph; self.stem = MultiModalStem(c) if multimodal else nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=False)); self.enc1=ConvBlock(c,c); self.enc2=ConvBlock(c,c*2); self.enc3=ConvBlock(c*2,c*4); self.bottleneck=ConvBlock(c*4,c*8); self.convnext_block=ConvNeXtBlock(c*8,convnext_expansion,convnext_kernel_size,convnext_layer_scale,convnext_dropout) if convnext_block else nn.Identity(); self.dropout=nn.Dropout2d(dropout) if dropout else nn.Identity(); self.pool=nn.MaxPool2d(2); self.pyramid_context=PyramidContext(c*8,pyramid_dilations,pyramid_reduction) if pyramid_context else nn.Identity(); self.global_context=PooledGlobalContext(c*8,global_pool_size) if global_context else nn.Identity(); self.frequency_multiscale=FrequencyMultiScale(c*8,frequency_hidden_channels,frequency_dilations) if frequency_multiscale else nn.Identity(); self.stage_graph=StageConditionedGraph(num_classes,stage_count) if stage_conditioned_graph else None; self.graph_classifier=nn.Conv2d(c*8,num_classes,1) if stage_conditioned_graph else None; self.graph_projection=nn.Conv2d(c*8,c*8,1) if stage_conditioned_graph else None; self.attention=nn.MultiheadAttention(c*8, 4, batch_first=True) if attention else None; self.up3=nn.ConvTranspose2d(c*8,c*4,2,2); self.dec3=ConvBlock(c*8,c*4); self.up2=nn.ConvTranspose2d(c*4,c*2,2,2); self.dec2=ConvBlock(c*4,c*2); self.up1=nn.ConvTranspose2d(c*2,c,2,2); self.dec1=ConvBlock(c*2,c); self.boundary_refiner=BoundaryGuidedRefiner(c,num_classes,boundary_hidden_channels) if boundary_refinement else None; self.class_context=ClassContextRefiner(c,num_classes) if class_context else nn.Identity(); self.seg_head=nn.Conv2d(c,num_classes,1); self.boundary_head=nn.Conv2d(c,1,1); self.class_boundary_head=nn.Conv2d(c,num_classes,1) if class_boundary and not boundary_refinement else None; self.aux3=nn.Conv2d(c*4,num_classes,1) if deep_supervision else None; self.aux2=nn.Conv2d(c*2,num_classes,1) if deep_supervision else None; self.skip_fusion3=AdaptiveSkipFusion(c*4,c*4,adaptive_skip_hidden_channels) if adaptive_skip_fusion else None; self.skip_fusion2=AdaptiveSkipFusion(c*2,c*2,adaptive_skip_hidden_channels) if adaptive_skip_fusion else None; self.skip_fusion1=AdaptiveSkipFusion(c,c,adaptive_skip_hidden_channels) if adaptive_skip_fusion else None
        self.residual_encoder_block = ResidualEncoderBlock(c * 8, residual_expansion, residual_dropout) if residual_encoder_block else nn.Identity()
        self.adaptive_receptive_field = AdaptiveReceptiveField(c * 8, tuple(adaptive_receptive_dilations)) if adaptive_receptive_field else nn.Identity()
        self.anti_aliased_downsampling = anti_aliased_downsampling
        self.pool = AntiAliasedDownsample(c) if anti_aliased_downsampling else nn.MaxPool2d(2)
        self.cnn_ssm_block = CNNSSMBlock(c * 8, cnn_ssm_state_ratio) if cnn_ssm_block else nn.Identity()
        self.token_moe_adapter = TokenMoEAdapter(c * 8, token_moe_experts, token_moe_expansion) if token_moe_adapter else nn.Identity()
        self.bpr_refiner = BoundaryProximityRefiner(c, bpr_hidden_channels, bpr_proximity_kernel_size) if bpr_refinement else None
        if entropy_gated_refinement and not deep_supervision:
            raise ValueError("entropy_gated_refinement requires deep_supervision=True (the gate is the entropy of the aux2 head)")
        self.entropy_gate = EntropyGatedRefiner(c, num_classes, entropy_gate_hidden_channels) if entropy_gated_refinement else None
        self.texture_gate3 = PointRendRefiner(c * 4) if multiscale_texture_gate else None
        self.texture_gate2 = PointRendRefiner(c * 2) if multiscale_texture_gate else None
        if bmask_boundary_branch and (class_boundary or boundary_refinement):
            raise ValueError("bmask_boundary_branch is mutually exclusive with class_boundary and boundary_refinement")
        self.bmask = BMaskBoundaryBranch(c, c * 2, num_classes, bmask_hidden_channels, bmask_fusion) if bmask_boundary_branch else None
        self.pointrend = None
        self.pointrend_coarse_level = int(pointrend_coarse_level)
        if deep_strip and pointrend:
            raise ValueError("deep_strip and pointrend both refine sampled points; enable only one per stage")
        if refine_mask and (deep_strip or pointrend):
            raise ValueError("refine_mask replaces the segmentation head; enable only one refinement module per stage")
        if mask_transfiner and (refine_mask or deep_strip or pointrend):
            raise ValueError("mask_transfiner refines sampled nodes; enable only one refinement module per stage")
        if segfix and (mask_transfiner or refine_mask or deep_strip or pointrend):
            raise ValueError("segfix post-processes the dense logits; enable only one refinement module per stage")
        if cascade_psp and (segfix or mask_transfiner or refine_mask or deep_strip or pointrend):
            raise ValueError("cascade_psp refines the dense logits; enable only one refinement module per stage")
        if contour_refiner is not None and (cascade_psp or segfix or mask_transfiner or refine_mask or deep_strip or pointrend):
            raise ValueError("contour_refiner biases the dense logits; enable only one refinement module per stage")
        self.contour_refiner = PolarContourRefiner(c, num_classes, contour_refiner, tuple(contour_classes_of_interest), contour_num_vertices, contour_iterations, contour_hidden_channels, contour_step_px, contour_ray_steps, contour_max_radius, contour_logit_boost) if contour_refiner is not None else None
        self.cascade_psp = CascadePSPLite(in_channels, num_classes, cascade_hidden_channels, cascade_global_scale, cascade_patch_size, cascade_num_patches, tuple(cascade_classes_of_interest)) if cascade_psp else None
        self.segfix = SegFixHead(c, segfix_hidden_channels, segfix_offset_px, segfix_steps, segfix_boundary_threshold) if segfix else None
        self.mask_transfiner = MaskTransfinerLite((c, c * 2, c * 4), num_classes, tuple(transfiner_nodes_per_level), transfiner_d_model, transfiner_nhead, transfiner_num_layers, transfiner_dim_feedforward, transfiner_dropout, tuple(transfiner_classes_of_interest), transfiner_oversample_ratio, transfiner_importance_ratio) if mask_transfiner else None
        self.refine_mask = RefineMaskHead((c, c * 2, c * 4), num_classes, refine_mask_hidden_channels, refine_mask_boundary_aware) if refine_mask else None
        self.deep_strip = DeepStripRefiner(c, num_classes, deep_strip_num_strips, deep_strip_length, deep_strip_spacing_px, deep_strip_hidden_channels, tuple(deep_strip_classes_of_interest), deep_strip_oversample_ratio, deep_strip_importance_sample_ratio, deep_strip_inference_num_strips) if deep_strip else None
        if pointrend:
            if not deep_supervision:
                raise ValueError("pointrend requires deep_supervision=True (the coarse prediction is the auxiliary head)")
            if self.pointrend_coarse_level not in (2, 3):
                raise ValueError(f"pointrend_coarse_level must be 2 or 3, got {pointrend_coarse_level}")
            self.pointrend = PointRendHead(c, num_classes, pointrend_num_points, pointrend_oversample_ratio, pointrend_importance_sample_ratio, pointrend_mlp_hidden, pointrend_mlp_layers, tuple(pointrend_classes_of_interest), pointrend_subdivision_steps, pointrend_subdivision_num_points)
        self.dynamic_snake_conv = dynamic_snake_conv
        self.snake1 = DynamicSnakeConv(c, dynamic_snake_kernel_size, dynamic_snake_max_offset) if dynamic_snake_conv else nn.Identity()
        self.snake2 = DynamicSnakeConv(c * 2, dynamic_snake_kernel_size, dynamic_snake_max_offset) if dynamic_snake_conv else nn.Identity()
        if literature_block is not None:
            if literature_block not in LITERATURE_BLOCKS:
                raise ValueError(f"unknown literature_block {literature_block!r}; choose from {sorted(LITERATURE_BLOCKS)}")
            self.literature_block = LITERATURE_BLOCKS[literature_block](c * 8, **dict(literature_block_kwargs or {}))
        else:
            self.literature_block = nn.Identity()

    def forward(self, x, stage=None):
        image = x
        aux = {}; x, alpha = (self.stem(x) if self.multimodal else (self.stem(x), None));
        if alpha is not None: aux["modality_weights"] = alpha
        e1=self.snake1(self.enc1(x)); e2=self.snake2(self.enc2(self.pool(e1))); e3=self.enc3(self.pool(e2)); b=self.frequency_multiscale(self.global_context(self.pyramid_context(self.dropout(self.convnext_block(self.bottleneck(self.pool(e3)))))))
        b = self.token_moe_adapter(self.cnn_ssm_block(self.adaptive_receptive_field(self.residual_encoder_block(b))))
        b = self.literature_block(b)
        if self.stage_graph is not None:
            if stage is None:
                raise ValueError("stage is required when stage_conditioned_graph is enabled")
            probabilities = self.graph_classifier(b).softmax(1)
            normalizer = probabilities.flatten(2).sum(-1).clamp_min(1e-6).unsqueeze(-1)
            class_features = torch.einsum("bkhw,bchw->bkc", probabilities, b) / normalizer
            class_features = self.stage_graph(class_features, stage.to(b.device).long())
            context = torch.einsum("bkhw,bkc->bchw", probabilities, class_features)
            b = b + self.graph_projection(context)
        if self.attention:
            shape=b.shape; tokens=b.flatten(2).transpose(1,2); tokens,_=self.attention(tokens,tokens,tokens); b=tokens.transpose(1,2).reshape(shape)
        up3 = self.up3(b); d3=self.dec3(self.skip_fusion3(up3, e3) if self.skip_fusion3 is not None else torch.cat((up3,e3),1))
        if self.texture_gate3 is not None: d3 = self.texture_gate3(d3)
        up2 = self.up2(d3); d2=self.dec2(self.skip_fusion2(up2, e2) if self.skip_fusion2 is not None else torch.cat((up2,e2),1))
        if self.texture_gate2 is not None: d2 = self.texture_gate2(d2)
        up1 = self.up1(d2); d1=self.dec1(self.skip_fusion1(up1, e1) if self.skip_fusion1 is not None else torch.cat((up1,e1),1));
        if self.entropy_gate is not None:
            d1, entropy_gate = self.entropy_gate(d1, self.aux2(d2)); aux["entropy_gate"] = entropy_gate
        if self.bmask is not None:
            d1, class_boundary_logits = self.bmask(d1, d2); aux["class_boundary"] = class_boundary_logits
        if self.boundary_refiner is not None:
            d1, class_boundary_logits = self.boundary_refiner(d1); aux["class_boundary"] = class_boundary_logits
        if isinstance(self.class_context, ClassContextRefiner):
            d1, context_probs = self.class_context(d1); aux["class_context"] = context_probs
        aux["boundary"] = self.boundary_head(d1)
        if self.bpr_refiner is not None:
            d1, bpr_proximity = self.bpr_refiner(d1, aux["boundary"]); aux["bpr_proximity"] = bpr_proximity
        if self.class_boundary and self.class_boundary_head is not None:
            aux["class_boundary"] = self.class_boundary_head(d1)
        if self.deep_supervision:
            aux3_raw = self.aux3(d3); aux2_raw = self.aux2(d2)
            aux["aux_logits_3"] = F.interpolate(aux3_raw, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            aux["aux_logits_2"] = F.interpolate(aux2_raw, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        if self.refine_mask is not None:
            refined, stage_logits = self.refine_mask([d1, d2, d3])
            for index, stage in enumerate(stage_logits[:-1], start=1):
                aux[f"refine_logits_{index}"] = F.interpolate(stage, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            return {"logits": refined, **aux}
        if self.contour_refiner is not None:
            dense_logits = self.seg_head(d1)
            contour = self.contour_refiner(d1, dense_logits)
            aux["dense_logits"] = dense_logits
            aux["contour_centers"], aux["contour_outer"], aux["contour_inner"] = contour["centers"], contour["outer"], contour["inner"]
            return {"logits": contour["logits"], **aux}
        if self.cascade_psp is not None:
            refined, global_logits = self.cascade_psp(image, self.seg_head(d1))
            aux["refine_logits_1"] = global_logits
            return {"logits": refined, **aux}
        if self.segfix is not None:
            dense_logits = self.seg_head(d1)
            aux["segfix_boundary"], aux["segfix_direction"] = self.segfix(d1)
            if self.training:
                return {"logits": dense_logits, **aux}
            aux["dense_logits"] = dense_logits
            return {"logits": self.segfix.refine(dense_logits, aux["segfix_boundary"], aux["segfix_direction"]), **aux}
        if self.mask_transfiner is not None:
            dense_logits = self.seg_head(d1)
            if self.training:
                aux["point_logits"], aux["point_coords"] = self.mask_transfiner.forward_train([d1, d2, d3], dense_logits)
                return {"logits": dense_logits, **aux}
            aux["dense_logits"] = dense_logits
            return {"logits": self.mask_transfiner.forward_inference([d1, d2, d3], dense_logits), **aux}
        if self.deep_strip is not None:
            dense_logits = self.seg_head(d1)
            if self.training:
                aux["point_logits"], aux["point_coords"] = self.deep_strip.forward_train(d1, dense_logits)
                return {"logits": dense_logits, **aux}
            aux["dense_logits"] = dense_logits
            return {"logits": self.deep_strip.forward_inference(d1, dense_logits), **aux}
        if self.pointrend is not None:
            coarse = aux2_raw if self.pointrend_coarse_level == 2 else aux3_raw
            if self.training:
                aux["point_logits"], aux["point_coords"] = self.pointrend.forward_train(d1, coarse)
                return {"logits": self.seg_head(d1), **aux}
            aux["dense_logits"] = self.seg_head(d1)
            return {"logits": self.pointrend.forward_inference(d1, coarse, tuple(d1.shape[-2:])), **aux}
        return {"logits": self.seg_head(d1), **aux}
