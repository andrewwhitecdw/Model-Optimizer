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
    """Yield the distinct (gate, up) expert-projection pairs across all MoE specs.

    Deduplicated because consumers apply every pair opportunistically to every
    iterable-experts module (getattr-guarded), matching the legacy engine behavior —
    unknown models still benefit if their naming matches any registered pair.
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


def match_moe_block(module: "nn.Module") -> MoESpec | None:
    """Return the MoE spec whose ``block_names`` matches ``module``.

    Case-insensitive exact-name match against the class names in ``module``'s MRO
    (see ``match_class_names``); quantized wrapper classes match through their
    original base class.
    """
    for spec in iter_specs(MoESpec):
        if match_class_names(module, spec.block_names):
            return spec
    return None
