#include "binding.cuh"

void per_class_topk(const nb::ndarray<> &topk_scores,
                    const nb::ndarray<> &topk_indices,
                    const nb::ndarray<> &scores, const nb::ndarray<> &indices,
                    int num_rois, int num_classes, int topk) {
  printf("Not implemented yet: per_class_topk\n");
}