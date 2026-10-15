import torch


def temporal_consistency_loss(current, previous, flow=None, gate=None):
    """Warp-free baseline consistency loss; flow/gate hooks are explicit for S7."""
    if flow is not None:
        # flow is expected as normalized grid [B,H,W,2]
        previous = torch.nn.functional.grid_sample(previous, flow, align_corners=True)
    error = (current - previous).abs()
    return (error * gate).mean() if gate is not None else error.mean()
