"""Evaluation namespace; metric implementations live in :mod:`matblast.metrics`."""

from .paired import multiseed_paired_summary, paired_summary, per_image_iou

__all__ = ["multiseed_paired_summary", "paired_summary", "per_image_iou"]
