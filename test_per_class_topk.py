"""
Batched Non-Maximum Suppression (NMS) unit tests using PyTorch
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


def test_per_class_sort(num_rois=8, num_classes=4, random_zeros=False, debug=False):
    if debug:
        print_title("Test Per Class Sort")
        print(f"num_rois: {num_rois}, num_classes: {num_classes}, random_zeros: {random_zeros}")
    scores = torch.rand((num_classes, num_rois)).float()
    indices = torch.arange(num_classes * num_rois).view(num_classes, num_rois)
    if random_zeros:
        for i in range(num_classes):
            zero_indices = torch.randperm(num_rois)[: np.random.randint(0, num_rois)]
            scores[i, zero_indices] = 0.0
            indices[i, zero_indices] = -1
    # torch.save(scores, f"{filename_prefix_}_input_0.data")

    if debug:
        print_title("Input Scores")
        print(scores)
        print_title("Input Indices")
        print(indices)

    sort_order = torch.argsort(scores, dim=1, descending=True)
    scores_sorted = torch.gather(scores, 1, sort_order)
    indices = torch.gather(indices, 1, sort_order)
    # torch.save(scores_sorted, f"{filename_prefix_}_golden_0.data")
    # torch.save(indices, f"{filename_prefix_}_golden_1.data")

    if debug:
        print_title("Sorted Scores")
        print(scores_sorted)
        print_title("Sorted Indices")
        print(indices)


if __name__ == "__main__":
    test_per_class_sort(num_rois=8, num_classes=4, random_zeros=True, debug=DEBUG)
