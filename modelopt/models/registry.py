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

"""Registry indexing the per-model ``ModelSpec`` by HF model type.

Model modules register their spec at import time; one spec per model type. Lookups
return ``None`` (or an empty list) when nothing matches, so callers can fail loudly
or fall back per their own policy. Matching is by model-type and class-name strings
only, so this package needs no torch import.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from .specs import ModelSpec, MoEVariant

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "get_spec",
    "get_specs",
    "hf_model_type",
    "iter_gate_up_pairs",
    "iter_pqs_fuse_rules",
    "match_moe_block",
    "register",
    "weight_plus_one_norm_names",
]

_SPECS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """Register a model spec and return it. One spec per model type."""
    if spec.model_type in _SPECS:
        raise ValueError(f"ModelSpec for model type {spec.model_type!r} already registered")
    _SPECS[spec.model_type] = spec
    return spec


def get_spec(model_type: str) -> ModelSpec | None:
    """Return the spec registered for ``model_type``, or ``None``."""
    return _SPECS.get(model_type)


def get_specs() -> list[ModelSpec]:
    """Return all registered specs, in registration order."""
    return list(_SPECS.values())


def iter_pqs_fuse_rules():
    """Yield each spec's ``(module_class_substrings, fuse_into, fuse_from)`` AWQ fusion rules."""
    for spec in get_specs():
        yield from spec.pqs_fuse_rules


def iter_gate_up_pairs() -> Iterator[tuple[str, str]]:
    """Yield the distinct (gate, up) projection-name pairs across all MoE variants.

    Global-vocabulary semantics: consumers try every pair on every module,
    getattr-guarded, so adding a pair to any spec affects all models whose modules
    carry those attribute names. Prefer per-module resolution
    (``match_moe_block(module).gate_up_pair``) wherever the module is an
    identifiable MoE block.
    """
    seen = set()
    for spec in get_specs():
        for variant in spec.moe_variants:
            pair = variant.gate_up_pair
            if pair is not None and pair not in seen:
                seen.add(pair)
                yield pair


def weight_plus_one_norm_names() -> tuple[str, ...]:
    """All norm class names whose stored weight is ``w - 1``, across all specs."""
    return tuple(name for spec in get_specs() for name in spec.weight_plus_one_norm_names)


def hf_model_type(model) -> str | None:
    """Return the root HF model type (``model.config.model_type``), or ``None``.

    Accepts a model or a config object (duck-typed, no transformers import). This
    is the key for ``get_spec`` / ``match_moe_block``.
    """
    config = getattr(model, "config", model)
    model_type = getattr(config, "model_type", None)
    return model_type if isinstance(model_type, str) else None


def match_moe_block(module: "nn.Module", model_type: str | None = None) -> MoEVariant | None:
    """Return the MoE layout variant for ``module``, resolved by model type.

    ``model_type`` (the root ``model.config.model_type``) is a strict filter: only
    that model's own spec is consulted, and an unregistered model type resolves to
    ``None`` even if the module's class names coincide with another model's.
    ``model_type=None`` searches all specs. A composite model whose MoE lives under
    a sub-model type registers the root type too (see ``gemma4.py``). Within the
    spec, variant ``block_names`` matched against the module's MRO picks the layout.
    """
    if model_type:
        spec = get_spec(model_type)
        return spec.match_moe_variant(module) if spec is not None else None
    for spec in get_specs():
        variant = spec.match_moe_variant(module)
        if variant is not None:
            return variant
    return None
