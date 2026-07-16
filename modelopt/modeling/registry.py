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

"""Registry of per-model specs, queried by spec type.

Model modules register their specs at import time (see ``models/``). Lookups return
``None`` when nothing matches, so callers can fall back to their default behavior.

Matching is by class-name string only (see ``matching``), so this package stays
dependency-free (any ``nn.Module`` — or any object — can be passed to the lookups
without importing torch here).
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar

from .base import ModelSpec
from .export import ExportSpec
from .matching import match_class_names
from .moe import MoESpec

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "iter_pqs_fuse_rules",
    "iter_specs",
    "match_moe_block",
    "register",
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


def iter_pqs_fuse_rules():
    """Yield every ``(module_class_substrings, fuse_into, fuse_from)`` AWQ fusion rule.

    Aggregated across all registered export specs (the consumer matches each model
    module against the substrings, so the order across specs does not matter).
    """
    for spec in iter_specs(ExportSpec):
        yield from spec.pqs_fuse_rules


def match_moe_block(module: "nn.Module") -> MoESpec | None:
    """Return the MoE spec whose ``block_names`` matches ``module``.

    Case-insensitive exact-name match against the class names in ``module``'s MRO
    (see ``matching.match_class_names``); quantized wrapper classes match through
    their original base class.
    """
    for spec in iter_specs(MoESpec):
        if match_class_names(module, spec.block_names):
            return spec
    return None
