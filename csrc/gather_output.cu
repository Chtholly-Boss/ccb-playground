#include "binding.cuh"

void gather_output(const nb::ndarray<> &nmsed_detections,
                   const nb::ndarray<> &nmsed_labels,
                   const nb::ndarray<> &nmsed_indices,
                   const nb::ndarray<> &bbox_delta, const nb::ndarray<> &rois,
                   const nb::ndarray<> &im_info, const nb::ndarray<> &scores,
                   const nb::ndarray<> &indices, int num_rois, int num_classes,
                   int topk, int keep_topk, bool return_idx) {
  printf("Not implemented yet: gather_output\n");
}