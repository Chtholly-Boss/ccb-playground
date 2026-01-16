"""
Batched Non-Maximum Suppression (NMS) unit tests using PyTorch

This test focuses on per-class NMS implementation
"""

from torch import Tensor
import torch
import numpy as np
from utils import (
    print_title,
    get_roi_class_idx,
    get_bbox_tensors_and_decode,
    compute_iou,
)

torch.random.manual_seed(43)
np.random.seed(43)

# DEBUG = True
DEBUG = False


def test_per_class_nms(
    num_rois,
    num_classes,
    topk,
    iou_threshold,
    debug=False,
):
    if debug:
        print_title("Test Per Class NMS")
        print(f"num_rois: {num_rois}, num_classes: {num_classes}, topk: {topk}, iou_threshold: {iou_threshold}")
    bbox_delta, rois, im_info, bbox_decode = get_bbox_tensors_and_decode(num_rois, num_classes, random_intersect=True, intersect_ratio=0.8)
    # torch.save(bbox_delta, f"{filename_prefix_}_input_0.data")
    # torch.save(rois, f"{filename_prefix_}_input_1.data")
    # torch.save(im_info, f"{filename_prefix_}_input_2.data")
    scores = torch.rand((num_classes, num_rois)).float()
    indices = torch.arange(num_classes * num_rois).to(torch.int32).view(num_classes, num_rois)

    sort_order = torch.argsort(scores, dim=1, descending=True)
    scores_sorted = torch.gather(scores, 1, sort_order)
    indices = torch.gather(indices, 1, sort_order)
    scores_unnmsed = scores_sorted[:, :topk]
    indices_unnmsed = indices[:, :topk]
    # torch.save(scores_unnmsed, f"{filename_prefix_}_input_3.data")
    # torch.save(indices_unnmsed, f"{filename_prefix_}_input_4.data")

    if debug:
        print_title("Scores Before NMS")
        print(scores_unnmsed)
        print_title("Indices Before NMS")
        print(indices_unnmsed)
        print_title("Decoded BBoxes")
        print(bbox_decode)
        topk_bboxes_decoded = torch.zeros((num_classes, topk, 4)).float()
        for c in range(num_classes):
            for i in range(topk):
                roi_idx, class_idx = get_roi_class_idx(indices_unnmsed[c, i].item(), num_rois, num_classes)
                topk_bboxes_decoded[c, i, :] = bbox_decode[roi_idx, class_idx, :]
        print_title("TopK Decoded BBoxes")
        print(topk_bboxes_decoded)

    nmsed_scores = scores_unnmsed.clone()
    nmsed_indices = indices_unnmsed.clone()
    for c in range(num_classes):
        keep_flags = torch.ones((topk,)).bool()
        for i in range(topk):
            if not keep_flags[i]:
                continue
            ref_roi_idx, ref_class_idx = get_roi_class_idx(indices_unnmsed[c, i].item(), num_rois, num_classes)
            ref_box = bbox_decode[ref_roi_idx, ref_class_idx, :]

            for j in range(i + 1, topk):
                if not keep_flags[j]:
                    continue
                curr_roi_idx, curr_class_idx = get_roi_class_idx(indices_unnmsed[c, j].item(), num_rois, num_classes)
                curr_box = bbox_decode[curr_roi_idx, curr_class_idx, :]

                iou = compute_iou(ref_box, curr_box)
                if iou >= iou_threshold:
                    keep_flags[j] = False
        for i in range(topk):
            if not keep_flags[i]:
                nmsed_scores[c, i] = 0.0
                nmsed_indices[c, i] = -1
    if debug:
        print_title("NMSed Scores")
        print(nmsed_scores)
        print_title("NMSed Indices")
        print(nmsed_indices)
    # torch.save(nmsed_scores, f"{filename_prefix_}_golden_0.data")
    # torch.save(nmsed_indices, f"{filename_prefix_}_golden_1.data")

    ccb_nmsed_scores = scores_unnmsed.clone()
    ccb_nmsed_indices = indices_unnmsed.clone()
    from build import ccb

    ccb.per_class_nms(
        ccb_nmsed_scores,
        ccb_nmsed_indices,
        bbox_delta,
        rois,
        im_info,
        scores_unnmsed,
        indices_unnmsed,
        num_rois,
        num_classes,
        topk,
        iou_threshold,
    )
    torch.testing.assert_close(nmsed_scores, ccb_nmsed_scores, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(nmsed_indices, ccb_nmsed_indices, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_per_class_nms(num_rois=512, num_classes=32, topk=63, iou_threshold=0.5, debug=DEBUG)
