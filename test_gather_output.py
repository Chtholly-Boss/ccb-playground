"""
Batched Non-Maximum Suppression (NMS) unit tests using PyTorch
"""

from torch import Tensor
import torch
import numpy as np
from utils import (
    print_title,
    get_bbox_tensors_and_decode,
)

torch.random.manual_seed(43)
np.random.seed(43)

# DEBUG = True
DEBUG = False


def test_gather_outputs(topk, keep_topk, num_rois, num_classes, return_idx, debug=False):
    if debug:
        print_title("Test Gather Outputs")
        print(f"topk: {topk}, keep_topk: {keep_topk}, num_rois: {num_rois}, num_classes: {num_classes}, return_idx: {return_idx}")
    bbox_delta, rois, im_info, bbox_decode = get_bbox_tensors_and_decode(num_rois, num_classes)
    # torch.save(bbox_delta, f"{filename_prefix}_gather_outputs_input_0.data")
    # torch.save(rois, f"{filename_prefix}_gather_outputs_input_1.data")
    # torch.save(im_info, f"{filename_prefix}_gather_outputs_input_2.data")

    scores = torch.rand((keep_topk,)).float()
    indices = torch.randint(0, num_rois * num_classes, (keep_topk,)).int()
    # torch.save(scores, f"{filename_prefix}_gather_outputs_input_3.data")
    # torch.save(indices, f"{filename_prefix}_gather_outputs_input_4.data")
    if debug:
        print_title("Input Scores")
        print(scores)
        print_title("Input Indices")
        print(indices)
        print_title("Decoded BBoxes")
        decoded_bboxes = torch.zeros((keep_topk, 4)).float()
        for i in range(keep_topk):
            idx = indices[i].item()
            roi_idx = idx % num_rois
            class_idx = idx // num_rois
            decoded_bboxes[i, :] = bbox_decode[roi_idx, class_idx, :]
        print(decoded_bboxes)

    nmsed_boxes = torch.zeros((keep_topk, 5)).float()  # (xmin, ymin, xmax, ymax, score)
    nmsed_labels = torch.zeros((keep_topk,)).int()
    for i in range(keep_topk):
        idx = indices[i].item()
        roi_idx = idx % num_rois
        class_idx = idx // num_rois
        nmsed_boxes[i, :4] = bbox_decode[roi_idx, class_idx, :]
        nmsed_boxes[i, 4] = scores[i]
        nmsed_labels[i] = class_idx

    # torch.save(nmsed_boxes, f"{filename_prefix_}_golden_0.data")
    if return_idx:
        nmsed_indices = indices[:keep_topk]
        # torch.save(nmsed_indices, f"{filename_prefix_}_golden_1.data")
    from build import ccb

    ccb_nmsed_boxes = torch.zeros_like(nmsed_boxes)
    ccb_nmsed_labels = torch.zeros_like(nmsed_labels)
    ccb_nmsed_indices = torch.zeros_like(indices)
    ccb.gather_output(
        ccb_nmsed_boxes,
        ccb_nmsed_labels,
        ccb_nmsed_indices,
        bbox_delta,
        rois,
        im_info,
        scores,
        indices,
        num_rois,
        num_classes,
        topk,
        keep_topk,
        return_idx,
    )
    if debug:
        print_title("NMSed Boxes")
        print(nmsed_boxes)
        print_title("NMSed Labels")
        print(nmsed_labels)
        if return_idx:
            print_title("NMSed Indices")
            print(nmsed_indices)


if __name__ == "__main__":
    test_gather_outputs(topk=8, keep_topk=4, num_rois=8, num_classes=2, return_idx=False, debug=DEBUG)
