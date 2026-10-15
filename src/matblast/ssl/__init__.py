"""Mean-teacher and domain-adaptation building blocks."""
from .mean_teacher import update_ema, confidence_mask
__all__ = ["update_ema", "confidence_mask"]
