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

filename_prefix = "./data/test_nms"

# DEBUG = True
DEBUG = False


def get_bbox_tensors_and_decode(
    num_rois, num_classes, random_intersect=False, intersect_ratio=0.3
):
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
                bbox_delta[follower_idx, c] = (
                    bbox_delta[anchor_idx, c] + delta_perturbation
                )

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
    bbox_decode[:, :, 0] = torch.clamp(
        pred_ctr_x - offset_w, min=0.0, max=im_info[1].item() - 1.0
    )
    bbox_decode[:, :, 1] = torch.clamp(
        pred_ctr_y - offset_h, min=0.0, max=im_info[0].item() - 1.0
    )
    bbox_decode[:, :, 2] = torch.clamp(
        pred_ctr_x + offset_w, min=0.0, max=im_info[1].item() - 1.0
    )
    bbox_decode[:, :, 3] = torch.clamp(
        pred_ctr_y + offset_h, min=0.0, max=im_info[0].item() - 1.0
    )
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


def test_permute_scores_and_filt(
    num_rois=8,
    num_classes=4,
    score_thresh=0.5,
    background_label_id=0,
    debug=False,
):
    if debug:
        print_title("Test Permute Scores and Filter")
        print(
            f"num_rois: {num_rois}, num_classes: {num_classes}, score_thresh: {score_thresh}, background_label_id: {background_label_id}"
        )
    filename_prefix_ = f"{filename_prefix}_permute_scores_and_filt_rois{num_rois}_classes{num_classes}_bgid{background_label_id}"
    scores = torch.rand((num_rois, num_classes)).float()
    torch.save(scores, f"{filename_prefix_}_input_0.data")

    if debug:
        print_title("Input Scores")
        print(scores)

    scores_perm = scores.permute(1, 0).contiguous()
    indices = torch.arange(num_rois * num_classes)
    for i in range(num_classes * num_rois):
        class_idx = i // num_rois
        roi_idx = i % num_rois
        if (
            class_idx == background_label_id
            or scores_perm[class_idx, roi_idx] < score_thresh
        ):
            scores_perm[class_idx, roi_idx] = 0.0
            indices[i] = -1
    if debug:
        print_title("Filtered Indices")
        print(indices.view(num_classes, num_rois))
        print_title("Permuted Scores")
        print(scores_perm)
    torch.save(scores_perm, f"{filename_prefix_}_golden_0.data")
    torch.save(indices.view(num_classes, num_rois), f"{filename_prefix_}_golden_1.data")
    config = {
        "op_type": "PERM",
        "input_tensors": [
            add_tensor("input_0", [num_rois, num_classes]),
        ],
        "output_tensors": [
            add_tensor("output_0", [num_classes, num_rois]),
            add_tensor("output_1", [num_classes, num_rois], dtype="int"),
        ],
        "evaluator": [
            {
                "type": "diff1",
                "threshold": 0.003,
            },
        ],
        "num_rois": num_rois,
        "num_classes": num_classes,
        "score_threshold": score_thresh,
        "background_label_id": background_label_id,
    }
    with open(f"{filename_prefix_}.json", "w") as f:
        json.dump(config, f, indent=4)


def test_per_class_sort(num_rois=8, num_classes=4, random_zeros=False, debug=False):
    if debug:
        print_title("Test Per Class Sort")
        print(
            f"num_rois: {num_rois}, num_classes: {num_classes}, random_zeros: {random_zeros}"
        )
    filename_prefix_ = f"{filename_prefix}_per_class_sort_rois{num_rois}_classes{num_classes}_{'zeros' if random_zeros else 'nozeros'}"
    scores = torch.rand((num_classes, num_rois)).float()
    indices = torch.arange(num_classes * num_rois).view(num_classes, num_rois)
    if random_zeros:
        for i in range(num_classes):
            zero_indices = torch.randperm(num_rois)[: np.random.randint(0, num_rois)]
            scores[i, zero_indices] = 0.0
            indices[i, zero_indices] = -1
    torch.save(scores, f"{filename_prefix_}_input_0.data")

    if debug:
        print_title("Input Scores")
        print(scores)
        print_title("Input Indices")
        print(indices)

    sort_order = torch.argsort(scores, dim=1, descending=True)
    scores_sorted = torch.gather(scores, 1, sort_order)
    indices = torch.gather(indices, 1, sort_order)
    torch.save(scores_sorted, f"{filename_prefix_}_golden_0.data")
    torch.save(indices, f"{filename_prefix_}_golden_1.data")

    if debug:
        print_title("Sorted Scores")
        print(scores_sorted)
        print_title("Sorted Indices")
        print(indices)

    config = {
        "op_type": "PER_CLASS_SORT",
        "input_tensors": [
            add_tensor("input_0", [num_classes, num_rois]),
        ],
        "output_tensors": [
            add_tensor("output_0", [num_classes, num_rois]),
            add_tensor("output_1", [num_classes, num_rois], dtype="int"),
        ],
        "evaluator": [
            {
                "type": "diff1",
                "threshold": 0.003,
            },
        ],
        "num_rois": num_rois,
        "num_classes": num_classes,
    }
    with open(f"{filename_prefix_}.json", "w") as f:
        json.dump(config, f, indent=4)


def test_per_class_nms(
    num_rois,
    num_classes,
    topk,
    iou_threshold,
    debug=False,
):
    if debug:
        print_title("Test Per Class NMS")
        print(
            f"num_rois: {num_rois}, num_classes: {num_classes}, topk: {topk}, iou_threshold: {iou_threshold}"
        )
    filename_prefix_ = f"{filename_prefix}_per_class_nms_rois{num_rois}_classes{num_classes}_topk{topk}"
    bbox_delta, rois, im_info, bbox_decode = get_bbox_tensors_and_decode(
        num_rois, num_classes, random_intersect=True, intersect_ratio=0.8
    )
    # torch.save(bbox_delta, f"{filename_prefix_}_input_0.data")
    # torch.save(rois, f"{filename_prefix_}_input_1.data")
    # torch.save(im_info, f"{filename_prefix_}_input_2.data")
    scores = torch.rand((num_classes, num_rois)).float()
    indices = torch.arange(num_classes * num_rois).view(num_classes, num_rois)
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
        topk_bboxes_decoded = torch.zeros((num_classes, topk, 4)).float()
        for c in range(num_classes):
            for i in range(topk):
                roi_idx = indices_unnmsed[c, i].item() % num_rois
                class_idx = indices_unnmsed[c, i].item() // num_rois
                topk_bboxes_decoded[c, i, :] = bbox_decode[roi_idx, class_idx, :]
        print(topk_bboxes_decoded)

    nmsed_scores = scores_unnmsed.clone()
    nmsed_indices = indices_unnmsed.clone()
    for c in range(num_classes):
        keep_flags = torch.ones((topk,)).bool()
        for i in range(topk):
            if not keep_flags[i]:
                continue
            ref_box_idx = indices_unnmsed[c, i].item() % num_rois
            ref_class_idx = indices_unnmsed[c, i].item() // num_rois
            ref_box = bbox_decode[ref_box_idx, ref_class_idx, :]
            for j in range(i + 1, topk):
                if not keep_flags[j]:
                    continue
                curr_box_idx = indices_unnmsed[c, j].item() % num_rois
                curr_class_idx = indices_unnmsed[c, j].item() // num_rois
                curr_box = bbox_decode[curr_box_idx, curr_class_idx, :]
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

    config = {
        "op_type": "PER_CLASS_NMS",
        "input_tensors": [
            add_tensor("input_0", [num_rois, num_classes, 4]),
            add_tensor("input_1", [num_rois, 4]),
            add_tensor("input_2", [3]),
            add_tensor("input_3", [num_classes, topk]),
            add_tensor("input_4", [num_classes, topk], dtype="int"),
        ],
        "output_tensors": [
            add_tensor("output_0", [num_classes, topk]),
            add_tensor("output_1", [num_classes, topk], dtype="int"),
        ],
        "evaluator": [
            {
                "type": "diff1",
                "threshold": 0.003,
            },
        ],
        "num_rois": num_rois,
        "num_classes": num_classes,
        "topk": topk,
        "iou_threshold": iou_threshold,
    }
    with open(f"{filename_prefix_}.json", "w") as f:
        json.dump(config, f, indent=4)


def test_per_image_sort(num_classes, topk, debug=False):
    if debug:
        print_title("Test Per Image Sort")
        print(f"num_classes: {num_classes}, topk: {topk}")
    filename_prefix_ = (
        f"{filename_prefix}_per_image_sort_classes{num_classes}_topk{topk}"
    )
    scores = torch.rand((num_classes * topk)).float()
    torch.save(scores, f"{filename_prefix_}_input_0.data")
    if debug:
        print_title("Input Scores")
        print(scores)
    scores_sorted = torch.sort(scores, descending=True)[0][:topk]
    torch.save(scores_sorted, f"{filename_prefix_}_golden_0.data")
    if debug:
        print_title("Sorted Scores")
        print(scores_sorted)
    config = {
        "op_type": "PER_IMAGE_SORT",
        "input_tensors": [
            add_tensor("input_0", [num_classes * topk]),
        ],
        "output_tensors": [
            add_tensor("output_0", [topk]),
        ],
        "evaluator": [
            {
                "type": "diff1",
                "threshold": 0.003,
            },
        ],
        "num_classes": num_classes,
        "topk": topk,
    }
    with open(f"{filename_prefix_}.json", "w") as f:
        json.dump(config, f, indent=4)


def test_gather_outputs(
    topk, keep_topk, num_rois, num_classes, return_idx, debug=False
):
    if debug:
        print_title("Test Gather Outputs")
        print(
            f"topk: {topk}, keep_topk: {keep_topk}, num_rois: {num_rois}, num_classes: {num_classes}, return_idx: {return_idx}"
        )
    filename_prefix_ = f"{filename_prefix}_gather_outputs_topk{topk}_keeptopk{keep_topk}_rois{num_rois}_classes{num_classes}_{'withidx' if return_idx else 'noidx'}"
    bbox_delta, rois, im_info, bbox_decode = get_bbox_tensors_and_decode(
        num_rois, num_classes
    )
    torch.save(bbox_delta, f"{filename_prefix}_gather_outputs_input_0.data")
    torch.save(rois, f"{filename_prefix}_gather_outputs_input_1.data")
    torch.save(im_info, f"{filename_prefix}_gather_outputs_input_2.data")

    scores = torch.rand((keep_topk,)).float()
    indices = torch.randint(0, num_rois * num_classes, (keep_topk,)).int()
    torch.save(scores, f"{filename_prefix}_gather_outputs_input_3.data")
    torch.save(indices, f"{filename_prefix}_gather_outputs_input_4.data")
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

    torch.save(nmsed_boxes, f"{filename_prefix_}_golden_0.data")
    if return_idx:
        nmsed_indices = indices
        torch.save(nmsed_indices, f"{filename_prefix_}_golden_1.data")
    if debug:
        print_title("NMSed Boxes")
        print(nmsed_boxes)
        print_title("NMSed Labels")
        print(nmsed_labels)
        if return_idx:
            print_title("NMSed Indices")
            print(nmsed_indices)
    config = {
        "op_type": "GATHER_OUTPUTS",
        "input_tensors": [
            add_tensor("input_0", [num_rois, num_classes, 4]),
            add_tensor("input_1", [num_rois, 4]),
            add_tensor("input_2", [3]),
            add_tensor("input_3", [keep_topk]),
            add_tensor("input_4", [keep_topk], dtype="int"),
        ],
        "output_tensors": [
            add_tensor("output_0", [keep_topk, 5]),
            add_tensor("output_1", [keep_topk], dtype="int"),
        ]
        + ([add_tensor("output_2", [keep_topk], dtype="int")] if return_idx else []),
        "evaluator": [
            {
                "type": "diff1",
                "threshold": 0.003,
            },
        ],
        "topk": topk,
        "keep_topk": keep_topk,
        "num_rois": num_rois,
        "num_classes": num_classes,
        "return_indices": return_idx,
    }
    with open(f"{filename_prefix_}.json", "w") as f:
        json.dump(config, f, indent=4)


def test_end_to_end_nms(): ...


if __name__ == "__main__":
    # test_permute_scores_and_filt(num_rois=8, num_classes=4, score_thresh=0.5, background_label_id=-1, debug=DEBUG)
    # test_per_class_sort(num_rois=8, num_classes=4, random_zeros=True, debug=DEBUG)
    test_per_class_nms(
        num_rois=16, num_classes=1, topk=8, iou_threshold=0.6, debug=DEBUG
    )
    # test_per_image_sort(num_classes=4, topk=8, debug=DEBUG)
    # test_gather_outputs(topk=8, keep_topk=4, num_rois=8, num_classes=2, return_idx=False, debug=DEBUG)
