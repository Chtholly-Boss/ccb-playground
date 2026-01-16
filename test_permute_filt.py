"""
Batched Non-Maximum Suppression (NMS) unit tests using PyTorch

This test focuses on permuting scores and filtering based on a score threshold
"""

from torch import Tensor
import torch
import numpy as np
from utils import (
    print_title,
)

torch.random.manual_seed(43)
np.random.seed(43)

# DEBUG = True
DEBUG = False


def test_permute_scores_and_filt(
    num_rois=8,
    num_classes=4,
    score_thresh=0.5,
    background_label_id=0,
    debug=False,
):
    if debug:
        print_title("Test Permute Scores and Filter")
        print(f"num_rois: {num_rois}, num_classes: {num_classes}, score_thresh: {score_thresh}, background_label_id: {background_label_id}")
    scores = torch.rand((num_rois, num_classes)).float()
    # torch.save(scores, f"{filename_prefix_}_input_0.data")

    if debug:
        print_title("Input Scores")
        print(scores)

    scores_perm = scores.permute(1, 0).contiguous()
    indices = torch.arange(num_rois * num_classes)
    for i in range(num_classes * num_rois):
        class_idx = i // num_rois
        roi_idx = i % num_rois
        if class_idx == background_label_id or scores_perm[class_idx, roi_idx] < score_thresh:
            scores_perm[class_idx, roi_idx] = 0.0
            indices[i] = -1
    from build import ccb

    ccb_scores_perm = torch.zeros_like(scores_perm)
    ccb_indices = torch.full_like(indices, -1)
    ccb.permute_scores_and_filt(
        ccb_scores_perm,
        ccb_indices,
        scores,
        num_rois,
        num_classes,
        background_label_id,
        score_thresh,
    )
    if debug:
        print_title("Filtered Indices")
        print(indices.view(num_classes, num_rois))
        print_title("Permuted Scores")
        print(scores_perm)
    # torch.save(scores_perm, f"{filename_prefix_}_golden_0.data")
    # torch.save(indices.view(num_classes, num_rois), f"{filename_prefix_}_golden_1.data")


if __name__ == "__main__":
    test_permute_scores_and_filt(num_rois=8, num_classes=4, score_thresh=0.5, background_label_id=-1, debug=DEBUG)
