#include "binding.cuh"

NB_MODULE(ccb, m) {
  m.def("add_tensor", &add_tensor);
  m.def("per_class_topk", &per_class_topk);
  m.def("per_class_nms", &per_class_nms);
  m.def("permute_scores_and_filt", &permute_scores_and_filt);
  m.def("gather_output", &gather_output);
}