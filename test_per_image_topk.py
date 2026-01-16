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


def test_per_image_sort(num_classes, topk, debug=False):
    if debug:
        print_title("Test Per Image Sort")
        print(f"num_classes: {num_classes}, topk: {topk}")
    scores = torch.rand((num_classes * topk)).float()
    # torch.save(scores, f"{filename_prefix_}_input_0.data")
    if debug:
        print_title("Input Scores")
        print(scores)
    scores_sorted = torch.sort(scores, descending=True)[0][:topk]
    # torch.save(scores_sorted, f"{filename_prefix_}_golden_0.data")
    if debug:
        print_title("Sorted Scores")
        print(scores_sorted)


if __name__ == "__main__":
    test_per_image_sort(num_classes=4, topk=8, debug=DEBUG)
