# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nemo_run as run
import torch

from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision


@run.cli.factory
def bf16_mixed() -> run.Config[MegatronMixedPrecision]:
    return run.Config(
        MegatronMixedPrecision,
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


@run.cli.factory
def fp16_mixed() -> run.Config[MegatronMixedPrecision]:
    return run.Config(
        MegatronMixedPrecision,
        precision="16-mixed",
        params_dtype=torch.half,
        pipeline_dtype=torch.half,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )


def bf16_with_fp8_mixed() -> run.Config[MegatronMixedPrecision]:
    """FP8 recipes are experimental and have not been tested for training convergence."""
    cfg = bf16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_params = True
    return cfg


def fp16_with_fp8_mixed() -> run.Config[MegatronMixedPrecision]:
    """FP8 recipes are experimental and have not been tested for training convergence."""
    cfg = fp16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_params = True
    return cfg
