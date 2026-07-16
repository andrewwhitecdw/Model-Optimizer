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

"""Per-model MoE architecture facts.

Unlike the subsystem specs (e.g. ``ExportSpec``), a ``MoESpec`` describes what a
model's MoE blocks *are* — which class, what the expert projections are called —
so any modelopt subsystem (export, quantization, speculative decoding, ...) can
read it instead of keeping its own per-model MoE table.
"""

from dataclasses import dataclass

from .base import ModelSpec

__all__ = ["MoESpec"]


@dataclass
class MoESpec(ModelSpec):
    """MoE architecture facts for one model (or one of its MoE-block variants).

    Resolved from a model sub-module via ``block_names``, the matching key: MoE block
    class names (e.g. ``"Qwen3MoeSparseMoeBlock"``) compared case-insensitively
    against the class names in the module's MRO (see ``matching.match_class_names``).
    """

    block_names: tuple[str, ...] = ()
    """Matching key: MoE block class names, matched against the module's MRO
    (case-insensitive exact names, not substrings)."""

    expert_linear_names: tuple[str, ...] | None = None
    """Expert linear projection names, e.g. ``("gate_proj", "down_proj", "up_proj")``.
    For layouts modelopt rewrites (e.g. quantized DBRX), these are the names on the
    rewritten module."""

    has_iterable_experts: bool = False
    """True when experts are per-expert iterable sub-modules (Mixtral, Qwen MoE,
    NemotronH, Gemma4) and can be grouped by ``get_experts_list``; False for stacked
    or fused layouts (DBRX, GptOss). NOTE: currently also doubles as the grouped-export
    support gate, so it is conservatively False for structurally iterable but
    unvalidated models (see ``deepseek``)."""
