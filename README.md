# ccb-playground

**C**uda-**C**-**B**inding Playground for Faster Experimentation

## Overview

This repository serves as a playground for experimenting with Cuda C/C++ using **lightweight bindings**. It is designed to facilitate rapid prototyping and testing of Cuda C/C++ code with minimal overhead.

Binding C/C++ code to Python have many advantages like:
- Easier to **construct input/output data** and **validate correctness**. Using libraries like [Pytorch](https://pytorch.org/), we can directly generate data by calling `torch.randn`, `.cuda()`, instead of writing `cudaMalloc`, `cudaMemcpy`, `cudaFree`. And compare results using `torch.testing.assert_allclose` instead of writing custom comparison code.
- Easier to **do benchmarking**. Nowadays, most of the ML research codebases are written in Python, or provide Python bindings. And Domain-Specific Languages (DSLs) like [Triton](https://triton-lang.org/main/index.html) are also Python-first. So having Python bindings allows us to easily integrate our Cuda C/C++ code into existing Python codebases for benchmarking and comparison.
- Easier to **Post-Process** results using Python libraries like [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), etc. Don't need to output to intermediate files and write separate scripts for post-processing.

Current Cuda C/C++ binding solutions are written using [PyTorch C++ API](https://docs.pytorch.org/cppdocs/), aka LibTorch. As it is built on [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) and consists of a lot of boilerplate code, it is not very lightweight and requires significant compilation time, which hinders rapid experimentation.

To address this, we use [nanobind](https://nanobind.readthedocs.io/en/latest/) to create lightweight bindings for Cuda C/C++ code. Our principle is to keep the Cuda C/C++ code focus on the core logic, and keep the binding code as minimal as possible. We encourage users to do pre-condition checks and data preparation in Python, and only pass necessary data to Cuda C/C++ functions for direct computation.

## Getting Started

We use `nanobind` for binding and `cmake` for building the Cuda C/C++ code. Please make sure you have them installed. If not, you can follow the instructions below: 

```sh
pip install nanobind
apt-get install cmake
# ninja is optional but recommended for faster builds
apt-get install ninja-build
```

You may also need to modify the `CMakeLists.txt` file to set the correct **CUDA architecture** and **Python Version** for your system. After that, you can build the Cuda C/C++ code using `cmake`:

```sh
mkdir build
cd build
# apt-get install libssl-dev  # if you encounter SSL issues
cmake -G Ninja ..
cmake --build . # or directly call `ninja`
```

Then you will obtain a shared library file `ccb.cpython-<version>-<platform>.so` in the `build` directory, which can be directly imported in Python.

```py
import ccb
import torch
a = torch.randn(4).cuda()
b = torch.randn(4).cuda()
c = torch.zeros(4).cuda()
ccb.add_tensor(c, a, b)
torch.testing.assert_close(c, a + b)
```

To add new kernel, simply create a new `.cu` file in the `csrc/` directory, and add declaration in `csrc/binding.hpp` and binding code in `csrc/binding.cu`. Then rebuild the project.
