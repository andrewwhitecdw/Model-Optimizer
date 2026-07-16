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

"""Per-model descriptor classes.

``ModelSpec`` subclasses come in two kinds, both registered per model so consumers
read values instead of branching on model names:

- **topic specs** hold architecture facts shared across subsystems (``MoESpec``:
  what a model's MoE blocks are; ``NormSpec``: norm-layer conventions);
- **subsystem specs** hold one subsystem's per-model policy (``ExportSpec``;
  quantization / speculative-decoding specs to follow).

Specs hold per-model data plus trivial accessors over that data; subsystem logic
never lives here.
"""

from dataclasses import dataclass

__all__ = ["ExportSpec", "MoESpec", "MoEVariant", "ModelSpec", "NormSpec", "match_class_names"]


@dataclass
class ModelSpec:
    """Base class for per-model data specs.

    Subclasses add the data fields of one topic or subsystem; a model registers one
    spec instance per kind it customizes (see ``models/``).
    """

    model_type: str
    """The HF model type this spec belongs to (``config.model_type``, e.g.
    ``"qwen3_moe"``). A model registers one spec instance per spec kind; same-model
    layout variants nest inside the spec (see ``MoESpec.variants``)."""


def match_class_names(module, names: tuple[str, ...]) -> bool:
    """Return True if any of ``names`` equals a class name in ``module``'s MRO.

    Case-insensitive exact-name comparison against ``cls.__name__`` for every class
    in ``type(module).__mro__`` — the same semantics as the export dispatch
    registry's string keys (``modelopt.torch.export.registry``). Dynamically
    generated quantized classes are subclasses of the original module class, so they
    match through their base; exact-name comparison avoids substring false
    positives. Comparison is case-insensitive because some registered names predate
    this registry and their casing was never exercised by the legacy substring
    matching.
    """
    mro_names = {cls.__name__.lower() for cls in type(module).__mro__}
    return any(name.lower() in mro_names for name in names)


@dataclass
class MoEVariant:
    """One concrete MoE-block layout of a model.

    A model type usually has exactly one; it has several when the same checkpoint
    materializes with different module classes and projection names (e.g. Mixtral
    across transformers generations). ``block_names`` tells the variants apart.
    """

    block_names: tuple[str, ...] = ()
    """Matching key: MoE block class names, matched against the module's MRO
    (case-insensitive exact names, not substrings; see ``match_class_names``)."""

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

    gate_up_pair: tuple[str, str] | None = None
    """The (gate, up) pair among ``expert_linear_names`` that serving engines fuse
    into a single ``gate_up_proj``, e.g. ``("gate_proj", "up_proj")`` or
    ``("w1", "w3")``. ``None`` for non-gated experts (NemotronH) and already-fused
    layouts (GptOss, DBRX). Consumed by amax syncing before quantized export (see
    ``sync_moe_gate_up_amax``) and by calibration grouping."""


@dataclass
class MoESpec(ModelSpec):
    """MoE architecture facts for one model: its MoE-block layout variant(s).

    Unlike the subsystem specs, this describes what a model's MoE blocks *are* —
    which class, what the expert projections are called — so any modelopt subsystem
    (export, quantization, speculative decoding, ...) can read it instead of keeping
    its own per-model MoE table.
    """

    variants: tuple[MoEVariant, ...] = ()
    """The model's MoE-block layouts; more than one when the same checkpoint
    materializes differently (see ``MoEVariant``)."""

    def match_variant(self, module) -> MoEVariant | None:
        """Return the variant whose ``block_names`` matches ``module``, else None."""
        for variant in self.variants:
            if match_class_names(module, variant.block_names):
                return variant
        return None

    def expert_linear_names_for(self, module) -> tuple[str, ...] | None:
        """Resolve ``module``'s expert linear names within this model.

        When every variant agrees on one naming, the module's class is irrelevant
        (a spec can provide naming without the block class being known); with
        several namings, the module's class picks the variant.
        """
        namings = {
            variant.expert_linear_names
            for variant in self.variants
            if variant.expert_linear_names is not None
        }
        if len(namings) == 1:
            return next(iter(namings))
        variant = self.match_variant(module)
        return variant.expert_linear_names if variant is not None else None


@dataclass
class NormSpec(ModelSpec):
    """Normalization-layer architecture facts for one model.

    Topic spec (like ``MoESpec``): shared facts any subsystem can read.
    """

    weight_plus_one_norm_names: tuple[str, ...] = ()
    """Class names of norm layers whose stored weight is ``w - 1`` (the effective
    scale is ``weight + 1``), e.g. Gemma's RMSNorm variants and LayerNorm1P.
    Matched against a norm module's MRO (case-insensitive exact names). Engines
    must account for the +1 when folding scales into the norm weight (AWQ
    pre_quant_scale fusion). A structural fallback (``zero_centered_gamma``) stays
    in the engine."""


@dataclass
class ExportSpec(ModelSpec):
    """Per-model policy for the unified HF export path.

    Architecture facts (MoE block classes, expert naming) live in ``MoESpec``; this
    spec holds decisions that belong to the export/quantization algorithms only.
    """

    pqs_fuse_rules: tuple[tuple[tuple[str, ...], str, str], ...] = ()
    """AWQ ``pre_quant_scale`` fusion rules, each a ``(module_class_substrings,
    fuse_into, fuse_from)`` triple: for a module whose class name contains one of the
    substrings, the pre_quant_scale on ``fuse_from`` is folded into ``fuse_into``
    (e.g. attention ``o_proj`` -> ``v_proj``, MLP ``down_proj`` -> ``up_proj``).
    A rule is a validated mathematical-equivalence claim for that model's modules,
    which is why it is declared per model rather than applied generically."""
