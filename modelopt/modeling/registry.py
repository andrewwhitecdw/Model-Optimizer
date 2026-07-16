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

from .specs import ExportSpec, ModelSpec, MoESpec, NormSpec

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "collect_model_types",
    "iter_gate_up_pairs",
    "iter_pqs_fuse_rules",
    "iter_specs",
    "match_class_names",
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


def iter_specs(spec_cls: type[SpecT]) -> Iterator[SpecT]:
    """Yield every registered spec of type ``spec_cls`` (in registration order)."""
    for spec in _SPECS:
        if isinstance(spec, spec_cls):
            yield spec


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


def iter_pqs_fuse_rules():
    """Yield every ``(module_class_substrings, fuse_into, fuse_from)`` AWQ fusion rule.

    Aggregated across all registered export specs (the consumer matches each model
    module against the substrings, so the order across specs does not matter).
    """
    for spec in iter_specs(ExportSpec):
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
    for spec in iter_specs(MoESpec):
        pair = spec.gate_up_pair
        if pair is not None and pair not in seen:
            seen.add(pair)
            yield pair


def weight_plus_one_norm_names() -> tuple[str, ...]:
    """All norm class names whose stored weight is ``w - 1``, across all norm specs."""
    return tuple(name for spec in iter_specs(NormSpec) for name in spec.weight_plus_one_norm_names)


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


def match_moe_block(module: "nn.Module", model_types: set[str] | None = None) -> MoESpec | None:
    """Return the MoE spec whose ``block_names`` matches ``module``.

    Identification is by case-insensitive exact-name match against the class names
    in ``module``'s MRO (see ``match_class_names``); quantized wrapper classes match
    through their original base class.

    ``model_types`` (e.g. from ``collect_model_types(model.config)``) scopes the
    result: when several specs' ``block_names`` match, one belonging to the model's
    own model types wins. Scope prefers, never excludes — a class-name match outside
    the scope still resolves, because remote-code towers reuse other models' module
    classes under their own model_type (e.g. ``DeepseekMoE`` blocks inside a Kimi
    model).
    """
    fallback = None
    for spec in iter_specs(MoESpec):
        if match_class_names(module, spec.block_names):
            if model_types and spec.model_type in model_types:
                return spec
            if fallback is None:
                fallback = spec
    return fallback
