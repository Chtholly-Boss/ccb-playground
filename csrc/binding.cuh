#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

void sgemm_nn(const nb::ndarray<float> &c, const nb::ndarray<float> &a,
              const nb::ndarray<float> &b, float alpha, float beta, int M,
              int N, int K);
