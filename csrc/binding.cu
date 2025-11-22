#include "binding.cuh"

NB_MODULE(ccb, m) {
  m.def("sgemm_nn", &sgemm);
  m.def("sm90_ws_gemm", &sm90_ws_gemm);
}