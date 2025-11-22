#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

int add(int a, int b);
void add_tensor(const nb::ndarray<> &out, const nb::ndarray<> &a,
                const nb::ndarray<> &b);
