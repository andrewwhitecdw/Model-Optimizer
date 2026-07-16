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

"""Registry that resolves a model sub-module to its spec.

Model modules register their specs at import time (see ``models/``). Lookups return
``None`` when nothing matches, so callers can fall back to their default behavior.

Matching is by class-name string only, so this package stays dependency-free (any
``nn.Module`` — or any object — can be passed to the lookups without importing torch
here).
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar

from .specs import ExportSpec, ModelSpec, MoESpec, MoEVariant, NormSpec

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "collect_model_types",
    "get_specs",
    "iter_gate_up_pairs",
    "iter_pqs_fuse_rules",
    "match_moe_block",
    "register",
    "weight_plus_one_norm_names",
]

SpecT = TypeVar("SpecT", bound=ModelSpec)

_SPECS: list[ModelSpec] = []


def register(spec: SpecT) -> SpecT:
    """Register a model spec and return it."""
    _SPECS.append(spec)
    return spec


def get_specs(spec_cls: type[SpecT], model_types: set[str] | None = None) -> list[SpecT]:
    """Return the registered specs of type ``spec_cls``, in registration order.

    This is the model_type index: with ``model_types`` (from
    ``collect_model_types(model.config)``), only the specs belonging to the model's
    own model types are returned, so consumers resolve per-model data without
    scanning the registry. Without it, all specs of the type are returned
    (aggregators, no-config compatibility).
    """
    return [
        spec
        for spec in _SPECS
        if isinstance(spec, spec_cls) and (not model_types or spec.model_type in model_types)
    ]


def iter_pqs_fuse_rules():
    """Yield every ``(module_class_substrings, fuse_into, fuse_from)`` AWQ fusion rule.

    Aggregated across all registered export specs (the consumer matches each model
    module against the substrings, so the order across specs does not matter).
    """
    for spec in get_specs(ExportSpec):
        yield from spec.pqs_fuse_rules


def iter_gate_up_pairs() -> Iterator[tuple[str, str]]:
    """Yield the distinct (gate, up) projection-name pairs across all MoE specs.

    GLOBAL-VOCABULARY semantics: consumers (currently only calibration sibling
    grouping in ``quantization/model_calib.py``, which also walks dense MLPs that
    no MoE spec can match) try every pair opportunistically on every module,
    getattr-guarded. Adding a pair to any spec therefore changes behavior for ALL
    models whose modules happen to carry those attribute names — prefer per-module
    resolution (``match_moe_block(module).gate_up_pair``) wherever the module is an
    identifiable MoE block. The dense-MLP case moves to a fusion-group topic spec
    in a follow-up (see MODEL_SPECIFIC_REFACTOR.md P5).
    """
    seen = set()
    for spec in get_specs(MoESpec):
        for variant in spec.variants:
            pair = variant.gate_up_pair
            if pair is not None and pair not in seen:
                seen.add(pair)
                yield pair


def weight_plus_one_norm_names() -> tuple[str, ...]:
    """All norm class names whose stored weight is ``w - 1``, across all norm specs."""
    return tuple(name for spec in get_specs(NormSpec) for name in spec.weight_plus_one_norm_names)


def collect_model_types(config) -> set[str]:
    """Collect every HF model type in a config tree (root plus nested sub-configs).

    Walks attribute values duck-typed as configs (anything exposing a string
    ``model_type``), so composite models contribute their tower types too — e.g. a
    VLM yields ``{"kimi_vl", "kimi_k2"}`` via ``config.text_config`` — without this
    package importing transformers. Pass ``model.config``; ``None`` yields an empty
    set.
    """
    found: set[str] = set()
    seen: set[int] = set()

    def _walk(cfg) -> None:
        if id(cfg) in seen:
            return
        seen.add(id(cfg))
        model_type = getattr(cfg, "model_type", None)
        if isinstance(model_type, str) and model_type:
            found.add(model_type)
        for value in vars(cfg).values():
            if isinstance(getattr(value, "model_type", None), str):
                _walk(value)

    if config is not None:
        _walk(config)
    return found


def match_moe_block(module: "nn.Module", model_types: set[str] | None = None) -> MoEVariant | None:
    """Return the MoE layout variant for ``module``, resolved by model type.

    ``model_types`` (from ``collect_model_types(model.config)``: root plus
    sub-config types, so VLM towers are covered) is a strict filter: only specs
    registered under the model's own model types are considered. A model whose
    model_type has no spec resolves to ``None`` even if its module class names
    coincide with another model's — register a spec instead of inheriting a
    neighbor's data. ``None`` or an empty set (no config available: unit tests,
    the TRT-LLM path) searches all specs.

    Within the scope, each spec's variant ``block_names`` identifies the block and
    disambiguates same-model layout variants (``MoESpec.match_variant``); quantized
    wrapper classes match through their original base class in the MRO.
    """
    for spec in get_specs(MoESpec, model_types):
        variant = spec.match_variant(module)
        if variant is not None:
            return variant
    return None
