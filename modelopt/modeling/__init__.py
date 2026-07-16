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

"""Per-model descriptors, organized by HF model type.

Holds declarative per-model data (no algorithms), one module per HF model type under
``models/``, mirroring ``transformers.models``. Architecture facts live in topic
specs shared across subsystems (``MoESpec``); per-subsystem policy lives in
subsystem specs (``ExportSpec``). Consumers resolve a spec via the registry lookups
and read its fields; an unmatched lookup returns ``None`` so callers fall back to
their default behavior.
"""

# Importing models registers every spec as a side effect.
from . import models
from .base import *
from .export import *
from .moe import *
from .registry import *
