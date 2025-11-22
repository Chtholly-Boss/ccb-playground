#include "binding.cuh"

NB_MODULE(ccb, m) { m.def("sgemm_nn", &sgemm_nn); }