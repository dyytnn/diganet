"""Named baseline constructors used by Stage 0 ablations."""
from .matblast_unet import MATBlastUNet
from .attention_unet import AttentionUNet


def unet(**kwargs):
    kwargs.pop("attention", None); kwargs.pop("multimodal", None)
    return MATBlastUNet(**kwargs, attention=False, multimodal=False)


def attention_unet(**kwargs):
    return AttentionUNet(**kwargs)


def nnunet2d(**kwargs):
    # A compact fallback with the same I/O contract. Replace with the official
    # nnU-Net trainer when its environment is available.
    kwargs.pop("attention", None); kwargs.pop("multimodal", None)
    return MATBlastUNet(**kwargs, attention=True, multimodal=False)
