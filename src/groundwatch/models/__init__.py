"""Model code lands here: shared backbone, Phase-1 surface segmentation head, and later
the panoptic / corrosion / ordinal grading heads (docs/perception-model-recommendation.md).

Not part of this skeleton. Planned v1, per the docs: off-the-shelf DINOv2 backbone plus a
segmentation head trained on the labelled tiles; domain SSL folded in for v2 only if it
beats off-the-shelf on the leave-one-drive-out validation.
"""
