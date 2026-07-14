# Export: Model-Specific Logic Refactor — Action Plan

**Goal:** Give the unified HF export path a per-model-family data registry
(`modelopt/torch/export/modeling/`), so that supporting a new model family means
adding one declarative spec file instead of editing if/elif chains across the
export engine.

**Scope:** The unified HF export path (`unified_export_hf.py` + its helpers) only.

- The **Megatron** path (`plugins/mcore_*`) already follows a per-family registry
  pattern and is untouched.
- The **TRT-LLM** checkpoint path (`model_config_export.py`, the `build_*`
  functions in `layer_utils.py`, `tensorrt_llm_utils.py`) is legacy: the plan is
  to move it out as-is (separate track), not to refactor it. Its per-model
  branches stay where they are.

## 1. Where the HF path stands after PR #1939

PR #1939 gave the HF path registry-based **module dispatch**: `ExportModuleRegistry`
and `PrepareMoEInputsRegistry` (`registry.py`, `hf_export_handlers.py`) select
*which handler processes a module* by class/predicate match, replacing the if/elif
chains that used to live in `unified_export_hf.py`.

What is still hardcoded is the **per-family data** those handlers (and other HF-path
helpers) consume. Inventory of model-specific logic reachable from the HF path:

| Item | Location | Kind |
|---|---|---|
| MoE expert linear names (Qwen/DeepSeek/Mixtral/DBRX/GptOss/NemotronH/Gemma4) | `layer_utils.get_expert_linear_names` | data (if/elif chain) |
| Duplicate expert-naming table + iterable-experts support gate | `layer_utils.get_experts_list` | data (if/elif chain) |
| MoE block class-name list | `layer_utils.is_moe` | data (shared with TRT-LLM path) |
| AWQ `pre_quant_scale` fusion rules (Llama/Qwen3) | `quant_utils.PQS_FUSE_MODULE_MAPPING` | data (table) |
| weight+1 layernorm class names (Gemma RMSNorm, LayerNorm1P) | `quant_utils._layernorm_uses_weight_plus_one` | data |
| Handler match keys (Llama4TextExperts, GptOssExperts, DbrxExperts, QuantMoELinear) | `hf_export_handlers.py` | dispatch (stays: structural, per-module) |
| Fused-expert gated/non-gated split (`gate_up_proj` vs `up_proj`) | `moe_utils.py` | data + structure |
| dummy-forward special cases (Whisper input, Nemotron-VL tower) | `unified_export_hf.requantize_resmooth_fused_llm_layers` | behavior |
| VLM language-tower extraction, DiffusionGemma tied-key reorder | `model_utils.py` | behavior |

## 2. Design

Two layers:

- **Engine** (existing files, organized by operation): owns all algorithms and the
  module walk; consults the registry for per-family values.
- **Modeling library** (`modeling/`, organized by model family): declarative
  per-family **data only**. No export logic, no imports from the parent package
  (only `torch.nn` + stdlib), so it sits at the bottom of the dependency graph.

```text
modeling/
  base.py        # ModelSpec dataclass — the per-family contract
  registry.py    # register() + lookups; returns None when unmatched
  __init__.py    # re-exports; importing it registers all families
  families/      # one small file per family; import == registration
```

Call sites follow a fallback-first shape, which keeps migration incremental and
behavior-preserving — a family not in the registry behaves exactly as before:

```python
spec = match_moe_block(module)
if spec is not None and spec.expert_linear_names is not None:
    return list(spec.expert_linear_names)
# ... legacy branch preserved as fallback ...
```

Note on naming: `modeling/registry.py` (per-family **data**, "what are this
family's values") is distinct from the top-level `registry.py` from PR #1939
(per-module **dispatch**, "which handler processes this module"). The two layers
compose: handlers look up family data through `modeling/`.

## 3. Migration plan

Each step is one PR with a fallback to the legacy path and an equivalence check
against existing export tests.

| Step | What | Status |
|---|---|---|
| **P1** | Registry skeleton + MoE expert naming: `get_expert_linear_names` and `get_experts_list` read `spec.expert_linear_names` / `spec.has_iterable_experts`. The #1 "add a MoE model" shotgun-surgery driver. | this PR |
| **P2** | `PQS_FUSE_MODULE_MAPPING` → `spec.pqs_fuse_rules`, aggregated across specs. | planned |
| **P3** | `is_moe` class-name list → `spec.moe_block_names` (shared consumer: keep fallback until the TRT-LLM path moves out). | planned |
| **P4** | HF handlers consume specs directly; fold remaining `moe_utils` naming data into specs. | planned |
| **OUT** | TRT-LLM path branches (`decoder_type` chains in `build_*`, `model_config_export.py`, `tensorrt_llm_utils.py`): frozen, moved out unchanged on a separate track. Dead code found during inventory (`MODEL_NAME_TO_TYPE`, `get_model_type`, `adjust_attn_amax_values`, `update_experts_avg_prequant_scale`) is deleted on that track too. | separate track |

**Guardrails:** fallback-first; one data category per PR; the engine keeps the
algorithms — families supply values only, never fork functions.
