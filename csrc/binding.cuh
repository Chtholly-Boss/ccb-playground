#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

void add_tensor(const nb::ndarray<> &out, const nb::ndarray<> &a,
                const nb::ndarray<> &b);

void per_class_nms(const nb::ndarray<> &nmsed_scores,
                   const nb::ndarray<> &nmsed_indices,
                   const nb::ndarray<> &bbox_delta, const nb::ndarray<> &rois,
                   const nb::ndarray<> &im_info, const nb::ndarray<> &scores,
                   const nb::ndarray<> &indices, int num_rois, int num_classes,
                   int topk, float iou_threshold);

void permute_scores_and_filt(const nb::ndarray<> &out_scores,
                             const nb::ndarray<> &out_indices,
                             const nb::ndarray<> &scores, int num_rois,
                             int num_classes, int background_label,
                             float score_threshold);

void per_class_topk(const nb::ndarray<> &topk_scores,
                    const nb::ndarray<> &topk_indices,
                    const nb::ndarray<> &scores, const nb::ndarray<> &indices,
                    int num_rois, int num_classes, int topk);

void gather_output(const nb::ndarray<> &nmsed_detections,
                   const nb::ndarray<> &nmsed_labels,
                   const nb::ndarray<> &nmsed_indices,
                   const nb::ndarray<> &bbox_delta, const nb::ndarray<> &rois,
                   const nb::ndarray<> &im_info, const nb::ndarray<> &scores,
                   const nb::ndarray<> &indices, int num_rois, int num_classes,
                   int topk, int keep_topk, bool return_idx);
