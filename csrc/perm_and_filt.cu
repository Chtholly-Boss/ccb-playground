#include "binding.cuh"

void permute_scores_and_filt(const nb::ndarray<> &out_scores,
                             const nb::ndarray<> &out_indices,
                             const nb::ndarray<> &scores, int num_rois,
                             int num_classes, int background_label,
                             float score_threshold) {
  printf("Not implemented yet: permute_scores_and_filt\n");
}