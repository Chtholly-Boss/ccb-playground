/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel

  modified from
  https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/
#include "binding.cuh"
#include "cutlass/gemm/device/gemm.h"

cudaError_t CutlassSgemm(int M, int N, int K, float alpha, float const *A,
                         float const *B, float beta, float *C) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions including the following example for single-precision GEMM.
  // Typical values are used as default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  using LayoutA = RowMajor;
  using LayoutB = ColumnMajor;
  using LayoutC = RowMajor;

  int lda = std::is_same_v<LayoutA, RowMajor> ? K : M;
  int ldb = std::is_same_v<LayoutB, RowMajor> ? N : K;
  int ldc = std::is_same_v<LayoutC, RowMajor> ? N : M;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,    // Data-type of A matrix
                                  LayoutA,  // Layout of A matrix
                                  float,    // Data-type of B matrix
                                  LayoutB,  // Layout of B matrix
                                  float,    // Data-type of C matrix
                                  LayoutC>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible in host code and passed to kernels by value. These may
  // include pointers, strides, scalars, and other arguments needed by Gemm and
  // its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible arguments to kernels and (2.) minimized
  // initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args(
      {M, N, K}, // Gemm Problem dimensions
      {A, lda},  // Tensor-ref for source matrix A
      {B, ldb},  // Tensor-ref for source matrix B
      {C, ldc},  // Tensor-ref for source matrix C
      {C, ldc},  // Tensor-ref for destination matrix D (may be different memory
                 // than source C matrix)
      {alpha, beta}); // Scalars used in the Epilogue

  // Launch the CUTLASS GEMM kernel
  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

void sgemm(const nb::ndarray<float> &c, const nb::ndarray<float> &a,
           const nb::ndarray<float> &b, float alpha, float beta, int M, int N,
           int K) {
  cudaError_t result =
      CutlassSgemm(M, N, K, alpha, a.data(), b.data(), beta, c.data());
  if (result != cudaSuccess) {
    throw std::runtime_error("Cutlass SGEMM failed");
  }
}