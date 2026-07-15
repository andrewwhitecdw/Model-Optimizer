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

"""Per-model data consumed by the unified HF export path."""

from dataclasses import dataclass

from .base import ModelSpec

__all__ = ["ExportSpec"]


@dataclass
class ExportSpec(ModelSpec):
    """Per-model data for the unified HF export path.

    Resolved from a model sub-module via ``moe_block_names``, the matching key: MoE
    block class-name substrings compared case-insensitively against
    ``type(module).__name__`` (e.g. ``"Qwen3MoeSparseMoeBlock"``).
    """

    moe_block_names: tuple[str, ...] = ()
    """Matching key: MoE block class-name substrings (case-insensitive)."""

    expert_linear_names: tuple[str, ...] | None = None
    """Expert linear projection names, e.g. ``("gate_proj", "down_proj", "up_proj")``."""

    has_iterable_experts: bool = False
    """True when experts are per-expert iterable sub-modules (Mixtral, Qwen MoE,
    NemotronH, Gemma4) and can be grouped by ``get_experts_list``; False for stacked
    or fused layouts (DBRX, GptOss)."""

    pqs_fuse_rules: tuple[tuple[tuple[str, ...], str, str], ...] = ()
    """AWQ ``pre_quant_scale`` fusion rules, each a ``(module_class_substrings,
    fuse_into, fuse_from)`` triple: for a module whose class name contains one of the
    substrings, the pre_quant_scale on ``fuse_from`` is folded into ``fuse_into``
    (e.g. attention ``o_proj`` -> ``v_proj``, MLP ``down_proj`` -> ``up_proj``)."""
