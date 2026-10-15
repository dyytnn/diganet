import torch


@torch.no_grad()
def update_ema(teacher, student, decay: float = 0.999):
    for t, s in zip(teacher.parameters(), student.parameters()): t.mul_(decay).add_(s, alpha=1 - decay)


def confidence_mask(logits, thresholds):
    probs = logits.softmax(1); threshold = torch.as_tensor(thresholds, device=logits.device).view(1, -1, 1, 1)
    return (probs >= threshold).all(1), probs
