#include "binding.cuh"

NB_MODULE(ccb, m) {
  m.def("add", &add);
  m.def("add_tensor", &add_tensor);
}