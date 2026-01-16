"""
Batched Non-Maximum Suppression (NMS) unit tests using PyTorch
"""

from torch import Tensor
import torch
import os
import numpy as np
import json

torch.random.manual_seed(43)
np.random.seed(43)

# DEBUG = True
DEBUG = False


def add_tensor(name, shape, dtype="float", layout="linear", mem_loc="ddr"):
    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "layout": layout,
        "mem_loc": mem_loc,
    }


def print_title(title):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20 + "\n")


def get_roi_class_idx(index, num_rois, num_classes):
    roi_idx = index // num_classes
    class_idx = index % num_classes
    return roi_idx, class_idx


def get_bbox_tensors_and_decode(num_rois, num_classes, random_intersect=False, intersect_ratio=0.3):
    """
    Generate bbox tensors and decode them.

    Args:
        num_rois: Number of RoIs
        num_classes: Number of classes
        random_intersect: If True, randomly pick some boxes and make them heavily intersect
        intersect_ratio: Ratio of boxes to make intersect (default 0.3)
    """
    bbox_delta = torch.rand((num_rois, num_classes, 4)).float()
    # rois 放大到合理的像素范围 (0, 100)
    rois = torch.rand((num_rois, 4)).float() * 100.0
    # 确保 rois 格式正确 (xmin < xmax, ymin < ymax)
    rois[:, 0], rois[:, 2] = (
        torch.min(rois[:, 0], rois[:, 2]),
        torch.max(rois[:, 0], rois[:, 2]),
    )
    rois[:, 1], rois[:, 3] = (
        torch.min(rois[:, 1], rois[:, 3]),
        torch.max(rois[:, 1], rois[:, 3]),
    )
    # im_info: [height, width, scale], 基于 rois 的最大值动态生成
    max_coord = max(rois.max().item(), 1.0)
    im_info = torch.tensor([max_coord + 50.0, max_coord + 50.0, 1.0]).float()
    if random_intersect and num_rois >= 2:
        # 随机选择一些框作为"锚点框"，然后让其他一些框与之高度重叠
        # 通过修改 rois 和 bbox_delta 来实现
        num_intersect_groups = max(1, int(num_rois * intersect_ratio))

        for _ in range(num_intersect_groups):
            # 随机选择一个锚点框索引和一个跟随框索引
            indices = torch.randperm(num_rois)[:2]
            anchor_idx = indices[0].item()
            follower_idx = indices[1].item()

            # 让 follower 的 rois 与 anchor 的 rois 非常接近（添加小扰动）
            perturbation = (torch.rand(4) - 0.5) * 2.0  # 小扰动 [-1, 1]
            rois[follower_idx] = rois[anchor_idx] + perturbation
            # 确保 rois 仍然有效
            rois[follower_idx] = torch.clamp(rois[follower_idx], min=0.0, max=100.0)
            rois[follower_idx, 0], rois[follower_idx, 2] = (
                min(rois[follower_idx, 0], rois[follower_idx, 2]),
                max(rois[follower_idx, 0], rois[follower_idx, 2]),
            )
            rois[follower_idx, 1], rois[follower_idx, 3] = (
                min(rois[follower_idx, 1], rois[follower_idx, 3]),
                max(rois[follower_idx, 1], rois[follower_idx, 3]),
            )

            # 对于每个类别，让 follower 的 bbox_delta 与 anchor 接近
            for c in range(num_classes):
                # 添加小扰动使解码后的框高度重叠但不完全相同
                delta_perturbation = (torch.rand(4) - 0.5) * 0.1  # 非常小的扰动
                bbox_delta[follower_idx, c] = bbox_delta[anchor_idx, c] + delta_perturbation
    # scale rois according to im_info
    rois_scaled = rois / im_info[2]

    widths = rois_scaled[:, 2] - rois_scaled[:, 0] + 1.0
    heights = rois_scaled[:, 3] - rois_scaled[:, 1] + 1.0
    ctr_x = rois_scaled[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = rois_scaled[:, 1] + 0.5 * (heights - 1.0)

    dx = bbox_delta[:, :, 0]
    dy = bbox_delta[:, :, 1]
    dw = bbox_delta[:, :, 2]
    dh = bbox_delta[:, :, 3]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    offset_w = 0.5 * (pred_w - 1.0)
    offset_h = 0.5 * (pred_h - 1.0)
    bbox_decode = torch.zeros_like(bbox_delta)
    bbox_decode[:, :, 0] = torch.clamp(pred_ctr_x - offset_w, min=0.0, max=im_info[1].item() - 1.0)
    bbox_decode[:, :, 1] = torch.clamp(pred_ctr_y - offset_h, min=0.0, max=im_info[0].item() - 1.0)
    bbox_decode[:, :, 2] = torch.clamp(pred_ctr_x + offset_w, min=0.0, max=im_info[1].item() - 1.0)
    bbox_decode[:, :, 3] = torch.clamp(pred_ctr_y + offset_h, min=0.0, max=im_info[0].item() - 1.0)
    return bbox_delta, rois, im_info, bbox_decode


def compute_iou(box1: Tensor, box2: Tensor) -> float:
    # box: (xmin, ymin, xmax, ymax)
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter_width = torch.clamp(x2 - x1, min=0.0)
    inter_height = torch.clamp(y2 - y1, min=0.0)
    inter_area = inter_width * inter_height

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area_box1 + area_box2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


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
