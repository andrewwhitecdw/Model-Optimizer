# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nemotron specs (HF model type ``nemotron``); Nemotron-H lives in ``nemotron_h.py``."""

from ..registry import register
from ..specs import NormSpec

# LayerNorm1P stores weight - 1 (zero-centered gamma). Both the plain Megatron-style
# class name and the HF Nemotron port are listed; modules exposing a
# ``zero_centered_gamma`` attribute are additionally caught by the engine's
# structural fallback.
register(
    NormSpec(
        model_type="nemotron",
        weight_plus_one_norm_names=("LayerNorm1P", "NemotronLayerNorm1P"),
    )
)
