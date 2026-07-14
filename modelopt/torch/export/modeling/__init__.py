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

"""Model-family export descriptors.

Holds per-model export data organized by model family. The export code resolves a
``ModelSpec`` via the registry lookups and reads its fields; an unmatched lookup
returns ``None`` so callers fall back to their default behavior.
"""

# Importing families registers every family spec as a side effect.
from . import families
from .base import *
from .registry import *
