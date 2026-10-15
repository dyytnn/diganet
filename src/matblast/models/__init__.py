import torch

from .matblast_unet import MATBlastUNet, PooledGlobalContext, ClassContextRefiner
from .matblast_v2 import MATBlastResUNet
from .heads import DecoderHeads
from .baselines import unet, attention_unet, nnunet2d
from .paper_baselines import BTFormer, I2CNet, MiMicsNet
from .canonical_baselines import DeepLabV3Plus, SegFormerB0, TransUNet
from .blastnet import BlastNet
from .massnet import MASSNet
from .unetpp import UNetPlusPlus
from .blasto_net import BlastoNet


def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg.get("model", cfg)
    kwargs = {k: v for k, v in m.items() if k != "name"}
    kwargs.setdefault("num_classes", cfg.get("data", {}).get("num_classes", 4))
    name = m.get("name", "matblast_unet")
    constructors = {
        "matblast_unet": MATBlastUNet,
        "matblast_v2": MATBlastResUNet,
        "unet": unet,
        "attention_unet": attention_unet,
        "unetpp": UNetPlusPlus,
        "nnunet2d": nnunet2d,
        "btformer": BTFormer,
        "i2cnet": I2CNet,
        "mimics_net": MiMicsNet,
        "massnet": MASSNet,
        "blastnet_reconstruction": BlastNet,
        "blasto_net_segmentation": BlastoNet,
        "deeplabv3plus": DeepLabV3Plus,
        "transunet": TransUNet,
        "segformer_b0": SegFormerB0,
    }
    if name not in constructors:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(constructors)}")
    return constructors[name](**kwargs)


__all__ = [
    "MATBlastUNet",
    "MATBlastResUNet",
    "DecoderHeads",
    "PooledGlobalContext",
    "ClassContextRefiner",
    "BTFormer",
    "I2CNet",
    "MiMicsNet",
    "DeepLabV3Plus",
    "TransUNet",
    "SegFormerB0",
    "MASSNet",
    "BlastNet",
    "UNetPlusPlus",
    "BlastoNet",
    "build_model",
]
