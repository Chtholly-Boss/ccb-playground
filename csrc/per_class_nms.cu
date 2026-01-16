#include "binding.cuh"

__global__ void k_per_class_nms(float *nmsed_scores, int *nmsed_indices,
                                float *bbox_delta, float *rois, float *im_info,
                                float *scores, int *indices, int num_rois,
                                int num_classes, int topk,
                                float iou_threshold) {
  printf("hello nms\n");
}

void per_class_nms(const nb::ndarray<> &nmsed_scores,
                   const nb::ndarray<> &nmsed_indices,
                   const nb::ndarray<> &bbox_delta, const nb::ndarray<> &rois,
                   const nb::ndarray<> &im_info, const nb::ndarray<> &scores,
                   const nb::ndarray<> &indices, int num_rois, int num_classes,
                   int topk, float iou_threshold) {
  printf("hello world\n");
  return;
}