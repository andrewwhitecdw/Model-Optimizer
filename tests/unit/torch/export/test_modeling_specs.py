# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the model-family export spec registry (modelopt.torch.export.modeling)."""

import pytest
import torch.nn as nn

from modelopt.torch.export.layer_utils import get_expert_linear_names, get_experts_list, is_moe
from modelopt.torch.export.modeling import iter_pqs_fuse_rules, match_moe_block


class _FakeQwen3MoeSparseMoeBlock(nn.Module):
    pass


class _QuantMixtralSparseMoeBlock(nn.Module):
    """Dynamically generated quantized classes keep the family name as a substring."""


class _UnknownMoeBlock(nn.Module):
    pass


def test_match_moe_block_by_substring():
    spec = match_moe_block(_FakeQwen3MoeSparseMoeBlock())
    assert spec is not None
    assert spec.name == "qwen_moe"
    assert spec.expert_linear_names == ("gate_proj", "down_proj", "up_proj")
    assert spec.has_iterable_experts


def test_match_moe_block_matches_quantized_class_name():
    spec = match_moe_block(_QuantMixtralSparseMoeBlock())
    assert spec is not None
    assert spec.name == "mixtral"


def test_match_moe_block_unmatched_returns_none():
    assert match_moe_block(_UnknownMoeBlock()) is None


def test_get_expert_linear_names_falls_back_when_unmatched():
    # No family spec matches — the legacy default (w1/w2/w3) must be preserved.
    assert get_expert_linear_names(_UnknownMoeBlock()) == ["w1", "w2", "w3"]


class _FakeNemotronHMOE(nn.Module):
    def __init__(self):
        super().__init__()

        class _Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(8, 16)
                self.down_proj = nn.Linear(16, 8)

        self.experts = nn.ModuleList([_Expert() for _ in range(3)])


def test_get_experts_list_groups_by_spec_linear_names():
    module = _FakeNemotronHMOE()
    groups = get_experts_list(module, "nemotronhforcausallm")
    assert len(groups) == 2  # up_proj group + down_proj group
    assert all(len(group) == 3 for group in groups)
    assert groups[0][0] is module.experts[0].up_proj
    assert groups[1][2] is module.experts[2].down_proj


def test_get_experts_list_rejects_non_iterable_families():
    # DBRX matches a spec but is not an iterable-experts layout; grouped export
    # must keep rejecting it (legacy behavior).
    class _FakeDBRXMoeSparseMoeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList()

    with pytest.raises(NotImplementedError):
        get_experts_list(_FakeDBRXMoeSparseMoeBlock(), "dbrxforcausallm")

    with pytest.raises(NotImplementedError):
        get_experts_list(_UnknownMoeBlock(), "unknownforcausallm")


class _FakeArcticMoE(nn.Module):
    pass


class _FakeDbrxFFN(nn.Module):
    pass


def test_is_moe_matches_registered_non_standard_names():
    # Non-standard MoE block names (no *SparseMoeBlock suffix, no router/experts
    # attributes) resolve through the family registry.
    assert is_moe(_FakeArcticMoE())
    assert is_moe(_FakeDbrxFFN())
    assert not is_moe(_UnknownMoeBlock())


def test_pqs_fuse_rules_match_legacy_mapping():
    # Aggregated per-family rules must reproduce the legacy PQS_FUSE_MODULE_MAPPING.
    rules = {
        (frozenset(substrings), fuse_into, fuse_from)
        for substrings, fuse_into, fuse_from in iter_pqs_fuse_rules()
    }
    legacy = {
        (frozenset({"LlamaAttention"}), "v_proj", "o_proj"),
        (frozenset({"LlamaMLP"}), "up_proj", "down_proj"),
        (frozenset({"Qwen3Attention", "Qwen3MoeAttention"}), "v_proj", "o_proj"),
        (frozenset({"Qwen3MLP", "Qwen3MoeMLP"}), "up_proj", "down_proj"),
    }
    assert rules == legacy
