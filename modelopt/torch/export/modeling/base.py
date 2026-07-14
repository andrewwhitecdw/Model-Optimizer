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

"""Per-model-family export descriptor.

A ``ModelSpec`` declares how one model family differs from the generic export path,
so the export code can read these values instead of branching on model names. Each
spec holds per-model data only, not export logic.
"""

from dataclasses import dataclass

__all__ = ["ModelSpec"]


@dataclass
class ModelSpec:
    """Per-model-family export data.

    A spec is resolved from a model sub-module via its matching keys and read for its
    per-model data fields. ``moe_block_names`` is the matching key: MoE block
    class-name substrings compared case-insensitively against
    ``type(module).__name__`` (e.g. ``"Qwen3MoeSparseMoeBlock"``).
    """

    name: str
    """Unique name of the model-family spec."""

    moe_block_names: tuple[str, ...] = ()
    """Matching key: MoE block class-name substrings (case-insensitive)."""

    expert_linear_names: tuple[str, ...] | None = None
    """Expert linear projection names, e.g. ``("gate_proj", "down_proj", "up_proj")``."""

    has_iterable_experts: bool = False
    """True when experts are per-expert iterable sub-modules (Mixtral, Qwen MoE,
    NemotronH, Gemma4) and can be grouped by ``get_experts_list``; False for stacked
    or fused layouts (DBRX, GptOss)."""
