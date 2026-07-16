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

"""Module-to-name matching shared by the spec registry (and, later, dispatch registries).

Same semantics as the export dispatch registry's string keys
(``modelopt.torch.export.registry``): a name matches when it equals the ``__name__``
of a class in ``type(module).__mro__``. Dynamically generated quantized classes are
subclasses of the original module class, so they match through their base; exact-name
comparison avoids substring false positives. Comparison is case-insensitive because
some registered names predate this registry and their casing was never exercised by
the legacy substring matching.

Only inspects ``type(module).__mro__``, so this module stays stdlib-only.
"""

__all__ = ["match_class_names"]


def match_class_names(module, names: tuple[str, ...]) -> bool:
    """Return True if any of ``names`` equals a class name in ``module``'s MRO.

    Case-insensitive exact-name comparison against ``cls.__name__`` for every class
    in ``type(module).__mro__``.
    """
    mro_names = {cls.__name__.lower() for cls in type(module).__mro__}
    return any(name.lower() in mro_names for name in names)
