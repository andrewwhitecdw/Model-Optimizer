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

"""Common base for per-model descriptors.

Each modelopt subsystem declares its per-model data in its own ``ModelSpec``
subclass (e.g. ``ExportSpec``; quantization / speculative-decoding specs to follow),
so consumers can read these values instead of branching on model names. Specs hold
per-model data only, no logic.
"""

from dataclasses import dataclass

__all__ = ["ModelSpec"]


@dataclass
class ModelSpec:
    """Base class for per-model data specs.

    Subclasses add the data fields of one modelopt subsystem; a model registers one
    spec instance per subsystem it customizes (see ``models/``).
    """

    model_type: str
    """The HF model type this spec belongs to (``config.model_type``, e.g.
    ``"qwen3_moe"``). Not necessarily unique: a model type may register several specs
    for different module layouts (e.g. two ``mixtral`` MoE-block variants)."""
