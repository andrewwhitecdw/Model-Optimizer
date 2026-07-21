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

"""Unit tests for the per-model spec registry (modelopt.models)."""

import pytest
import torch.nn as nn

from modelopt.models import ModelSpec, MoEVariant, hf_model_type, list_all_possible, match_moe_block
from modelopt.models.registry import _SPECS
from modelopt.torch.export.layer_utils import (
    get_expert_linear_names,
    get_experts_list,
    is_moe,
    sync_moe_gate_up_amax,
)
from modelopt.torch.export.quant_utils import _layernorm_uses_weight_plus_one


class Qwen3MoeSparseMoeBlock(nn.Module):
    pass


class MixtralSparseMoeBlock(nn.Module):
    pass


class QuantMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    """Quantized classes subclass the original module class; matching goes through the MRO."""


class _UnknownMoeBlock(nn.Module):
    pass


def test_match_moe_block_by_class_name():
    variant = match_moe_block(Qwen3MoeSparseMoeBlock())
    assert variant is not None
    assert variant.expert_linear_names == ("gate_proj", "down_proj", "up_proj")
    assert variant.has_iterable_experts


def test_match_moe_block_matches_quantized_class_via_mro():
    variant = match_moe_block(QuantMixtralSparseMoeBlock())
    assert variant is not None
    assert variant.expert_linear_names == ("w1", "w2", "w3")


def test_match_moe_block_unmatched_returns_none():
    assert match_moe_block(_UnknownMoeBlock()) is None


def test_get_expert_linear_names_raises_when_unmatched():
    # No spec for these model types and no fused-expert structure — must fail loudly
    # instead of guessing another model's naming (the legacy w1/w2/w3 default was
    # removed).
    with pytest.raises(NotImplementedError, match="expert linear names"):
        get_expert_linear_names(_UnknownMoeBlock(), "some_unknown_model")


def test_get_expert_linear_names_from_specs():
    # Arctic keeps the w1/w2/w3 naming it previously got from the engine default.
    assert get_expert_linear_names(_UnknownMoeBlock(), "arctic") == ["w1", "w2", "w3"]
    # DBRX resolves the quantized per-expert ModuleList names (previously it fell
    # through to the w1/w2/w3 default, which never existed on the quantized module).
    assert get_expert_linear_names(_UnknownMoeBlock(), "dbrx") == [
        "w1_linear",
        "w2_linear",
        "v1_linear",
    ]


class NemotronHMOE(nn.Module):
    def __init__(self):
        super().__init__()

        class _Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(8, 16)
                self.down_proj = nn.Linear(16, 8)

        self.experts = nn.ModuleList([_Expert() for _ in range(3)])


def test_get_experts_list_groups_by_spec_linear_names():
    module = NemotronHMOE()
    groups = get_experts_list(module, "nemotron_h")
    assert len(groups) == 2  # up_proj group + down_proj group
    assert all(len(group) == 3 for group in groups)
    assert groups[0][0] is module.experts[0].up_proj
    assert groups[1][2] is module.experts[2].down_proj


def test_get_experts_list_rejects_non_iterable_layouts():
    # DBRX matches a spec but is not an iterable-experts layout; grouped export
    # must keep rejecting it (legacy behavior).
    class DbrxFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList()

    with pytest.raises(NotImplementedError):
        get_experts_list(DbrxFFN(), "dbrx")

    with pytest.raises(NotImplementedError):
        get_experts_list(_UnknownMoeBlock(), "some_unknown_model")


class ArcticMoE(nn.Module):
    pass


class DbrxFFN(nn.Module):
    pass


def test_is_moe_matches_registered_non_standard_names():
    # Non-standard MoE block names (no *SparseMoeBlock suffix, no router/experts
    # attributes) resolve through the model spec registry.
    assert is_moe(ArcticMoE())
    assert is_moe(DbrxFFN())
    assert not is_moe(_UnknownMoeBlock())


def test_match_is_exact_name_not_substring():
    # A class whose name merely CONTAINS a registered name must not match; only
    # exact MRO class names do (quantized wrappers match via their base class).
    class MyArcticMoEWrapper(nn.Module):
        pass

    assert match_moe_block(MyArcticMoEWrapper()) is None


def test_pqs_fuse_rules_match_legacy_mapping():
    # Aggregated per-model rules, flattened to (class_substring, fuse_into, fuse_from)
    # triples, must reproduce the legacy PQS_FUSE_MODULE_MAPPING.
    rules = {
        (substring, fuse_into, fuse_from)
        for substrings, fuse_into, fuse_from in list_all_possible("pqs_fuse_rules")
        for substring in substrings
    }
    legacy = {
        ("LlamaAttention", "v_proj", "o_proj"),
        ("LlamaMLP", "up_proj", "down_proj"),
        ("Qwen3Attention", "v_proj", "o_proj"),
        ("Qwen3MoeAttention", "v_proj", "o_proj"),
        ("Qwen3MLP", "up_proj", "down_proj"),
        ("Qwen3MoeMLP", "up_proj", "down_proj"),
    }
    assert rules == legacy


def test_gate_up_pairs_match_legacy():
    # Aggregated per-model pairs must reproduce the legacy _GATE_UP_PAIRS set.
    assert set(list_all_possible("gate_up_pairs")) == {("gate_proj", "up_proj"), ("w1", "w3")}


def test_list_all_possible_only_accepts_collectable_attrs():
    assert {"pqs_fuse_rules", "gate_up_pairs", "weight_plus_one_norm_names"} <= set(
        ModelSpec.collectable_names()
    )
    with pytest.raises(ValueError, match="not a collectable"):
        list_all_possible("model_type")


def test_weight_plus_one_norm_names_cover_legacy():
    names = set(list_all_possible("weight_plus_one_norm_names"))
    assert {"GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm", "LayerNorm1P"} <= names


def test_layernorm_weight_plus_one_via_specs():
    class GemmaRMSNorm(nn.Module):
        pass

    class NemotronLayerNorm1P(nn.Module):
        pass

    class PlainRMSNorm(nn.Module):
        pass

    class ZeroCentered(nn.Module):
        zero_centered_gamma = True

    assert _layernorm_uses_weight_plus_one(GemmaRMSNorm())
    assert _layernorm_uses_weight_plus_one(NemotronLayerNorm1P())
    assert not _layernorm_uses_weight_plus_one(PlainRMSNorm())
    # Structural fallback stays in the engine.
    assert _layernorm_uses_weight_plus_one(ZeroCentered())


class _FakeQuantizer:
    def __init__(self, amax):
        self.amax = amax


def _make_gated_block(block_cls, gate_name, up_name, gate_amax, up_amax):
    import torch

    class _Expert(nn.Module):
        def __init__(self):
            super().__init__()
            setattr(self, gate_name, nn.Linear(4, 8))
            setattr(self, up_name, nn.Linear(4, 8))
            getattr(self, gate_name).weight_quantizer = _FakeQuantizer(torch.tensor(gate_amax))
            getattr(self, up_name).weight_quantizer = _FakeQuantizer(torch.tensor(up_amax))

    block = block_cls()
    block.experts = nn.ModuleList([_Expert()])
    return block


def test_sync_moe_gate_up_amax_uses_own_spec():
    import torch

    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    model = nn.Module()
    model.moe = _make_gated_block(
        Qwen3MoeSparseMoeBlock, "gate_proj", "up_proj", [1.0, 3.0], [2.0, 2.0]
    )
    assert sync_moe_gate_up_amax(model) == 1
    expert = model.moe.experts[0]
    assert torch.equal(expert.gate_proj.weight_quantizer.amax, torch.tensor([2.0, 3.0]))
    assert torch.equal(expert.up_proj.weight_quantizer.amax, torch.tensor([2.0, 3.0]))


def test_sync_moe_gate_up_amax_warns_on_unmatched_block():
    class UnknownSparseMoeBlock(nn.Module):
        """Passes is_moe by naming convention but has no MoESpec."""

    model = nn.Module()
    model.moe = _make_gated_block(UnknownSparseMoeBlock, "gate_proj", "up_proj", [1.0], [2.0])
    # No spec -> no cross-model guessing: nothing synced, one warning.
    with pytest.warns(UserWarning, match="no registered MoESpec"):
        assert sync_moe_gate_up_amax(model) == 0


def test_match_moe_block_scope_prefers_own_model_type():
    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    # A hypothetical remote-code fork registering the same block class name under
    # its own model type: scope must pick the model's own spec among candidates.
    fork_variant = MoEVariant(
        block_names=("Qwen3MoeSparseMoeBlock",),
        expert_linear_names=("a_proj", "b_proj"),
    )
    fork_spec = ModelSpec(model_type="zz_fork", moe_variants=(fork_variant,))
    _SPECS[fork_spec.model_type] = fork_spec
    try:
        assert match_moe_block(Qwen3MoeSparseMoeBlock(), "zz_fork") is fork_variant
        assert match_moe_block(Qwen3MoeSparseMoeBlock(), "qwen3_moe").expert_linear_names == (
            "gate_proj",
            "down_proj",
            "up_proj",
        )
        # No scope -> first registered class-name match wins (legacy order).
        assert match_moe_block(Qwen3MoeSparseMoeBlock()).expert_linear_names == (
            "gate_proj",
            "down_proj",
            "up_proj",
        )
    finally:
        del _SPECS[fork_spec.model_type]


def test_match_moe_block_scope_is_strict():
    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    # A model whose model_type has no spec resolves to None even when its module
    # class name coincides with another model's — register a spec instead of
    # inheriting a neighbor's data.
    assert match_moe_block(Qwen3MoeSparseMoeBlock(), "some_unknown_vlm") is None
    # No scope (no config available) searches all specs.
    assert match_moe_block(Qwen3MoeSparseMoeBlock()) is not None


def test_get_expert_linear_names_by_model_type_only():
    # With a scope, naming resolves from the model's own spec — the block class
    # name is irrelevant (a spec need not declare block_names to provide naming).
    assert get_expert_linear_names(_UnknownMoeBlock(), "qwen3_moe") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    with pytest.raises(NotImplementedError, match="model type"):
        get_expert_linear_names(_UnknownMoeBlock(), "some_unknown_vlm")


def test_mixtral_variants_disambiguated_by_block_class():
    class MixtralMoeSparseMoeBlock(nn.Module):
        """Legacy-naming Mixtral layout — same model type, different projections."""

    assert get_expert_linear_names(MixtralMoeSparseMoeBlock(), "mixtral") == [
        "linear_fc1",
        "linear_fc2",
    ]
    # An unrecognized block class under a multi-naming model type cannot resolve.
    with pytest.raises(NotImplementedError):
        get_expert_linear_names(_UnknownMoeBlock(), "mixtral")


def test_gemma4_both_root_types_resolve():
    # A gemma4 VLM's root model_type is gemma4; a text-only checkpoint's is
    # gemma4_text (gemma3 precedent). Both register the same layout.
    assert get_expert_linear_names(_UnknownMoeBlock(), "gemma4") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    assert get_expert_linear_names(_UnknownMoeBlock(), "gemma4_text") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]


def test_hf_model_type_accepts_model_or_config():
    from types import SimpleNamespace

    config = SimpleNamespace(model_type="qwen3_moe")
    model = SimpleNamespace(config=config)
    assert hf_model_type(model) == "qwen3_moe"
    assert hf_model_type(config) == "qwen3_moe"
    assert hf_model_type(SimpleNamespace()) is None
