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