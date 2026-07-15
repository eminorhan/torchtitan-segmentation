# Copyright (c) Emin Orhan.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.parallelize_dino import parallelize_dino


__all__ = [
    "parallelize_dino",
    "ParallelDims",
]