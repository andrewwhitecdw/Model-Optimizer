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

"""Registry that resolves a model sub-module to its ``ModelSpec``.

Families register their specs at import time (see ``families/``). Lookups return
``None`` when nothing matches, so callers can fall back to their default behavior.
"""

import torch.nn as nn

from .base import ModelSpec

__all__ = [
    "match_moe_block",
    "register",
]

_SPECS: list[ModelSpec] = []


def register(spec: ModelSpec) -> ModelSpec:
    """Register a model-family spec and return it."""
    _SPECS.append(spec)
    return spec


def match_moe_block(module: nn.Module) -> ModelSpec | None:
    """Return the spec matching ``module``'s class name against ``moe_block_names``.

    Case-insensitive substring match against ``type(module).__name__``.
    """
    cls_name = type(module).__name__.lower()
    for spec in _SPECS:
        if any(name.lower() in cls_name for name in spec.moe_block_names):
            return spec
    return None
