#!/usr/bin/env python3
"""Build a CPU-only workload and cache-accounting manifest from a Qwen config.

No model loading, GPU allocation, benchmark, or runtime-policy modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

VERIFY_ROWS = [1, 2, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 32]
CONTEXTS = [8192, 32768, 65536, 131072]


def positive(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Missing or invalid positive integer: {key}")
    return value


def make_manifest(raw: dict[str, Any], *, kv_bytes: int = 2, activation_bytes: int = 2,
                  state_bytes: int | None = None, budget_bytes: int = 90_000_000_000,
                  resident_weight_bytes: int | None = None,
                  workspace_reserve_bytes: int | None = None) -> dict[str, Any]:
    """Account for the dense/hybrid Qwen geometry used in this research.

    Estimates exclude allocator rounding, fragmentation, scores, draft-model
    weights, output buffers and other processes unless included by the caller.
    Budget flags are NOT permission to allocate or evidence of safe execution.
    """
    config = raw.get("text_config", raw)
    if not isinstance(config, dict):
        raise ValueError("text_config must be an object")
    if state_bytes is None:
        widths = {"float32": 4, "float16": 2, "bfloat16": 2}
        if config.get("mamba_ssm_dtype") not in widths:
            raise ValueError("Unknown mamba_ssm_dtype; provide an explicit state width")
        state_bytes = widths[config["mamba_ssm_dtype"]]
    if min(kv_bytes, activation_bytes, state_bytes, budget_bytes) <= 0:
        raise ValueError("Element widths and budget must be positive")
    for value in (resident_weight_bytes, workspace_reserve_bytes):
        if value is not None and value < 0:
            raise ValueError("Caller-supplied byte totals cannot be negative")
    layers = positive(config, "num_hidden_layers")
    hidden = positive(config, "hidden_size")
    pattern = config.get("layer_types")
    if pattern is None:
        interval = positive(config, "full_attention_interval")
        pattern = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                   for i in range(layers)]
    if len(pattern) != layers or set(pattern) - {"linear_attention", "full_attention"}:
        raise ValueError("Unsupported layer pattern; do not infer another attention architecture")
    linear, full = pattern.count("linear_attention"), pattern.count("full_attention")
    hkv = positive(config, "num_key_value_heads")
    head = positive(config, "head_dim")
    qheads = positive(config, "num_attention_heads")
    if qheads % hkv:
        raise ValueError("Query heads must be divisible by KV heads")
    hk = positive(config, "linear_num_key_heads")
    hv = positive(config, "linear_num_value_heads")
    dk = positive(config, "linear_key_head_dim")
    dv = positive(config, "linear_value_head_dim")
    conv = positive(config, "linear_conv_kernel_dim")
    if hv % hk:
        raise ValueError("Linear value heads must be divisible by key heads")
    experts = config.get("num_experts", 0)
    if not isinstance(experts, int) or isinstance(experts, bool) or experts < 0:
        raise ValueError("num_experts must be a nonnegative integer")
    is_moe = experts > 0
    selected = positive(config, "num_experts_per_tok") if is_moe else 1
    if selected > experts and is_moe:
        raise ValueError("num_experts_per_tok exceeds num_experts")
    intermediate = positive(config, "moe_intermediate_size" if is_moe else "intermediate_size")
    shared = config.get("shared_expert_intermediate_size", 0) if is_moe else 0
    if not isinstance(shared, int) or isinstance(shared, bool) or shared < 0:
        raise ValueError("shared_expert_intermediate_size must be nonnegative")
    vocab = positive(config, "vocab_size")
    qkv_width = 2 * hk * dk + hv * dv
    committed_state = linear * hv * dv * dk * state_bytes
    conv_state = linear * (conv - 1) * qkv_width * activation_bytes
    kv_per_token = full * 2 * hkv * head * kv_bytes
    # For scalar-gate replay that retains k, v, g and beta. Q is not needed
    # to rebuild state, but projection/output temporaries are not included.
    journal_per_position = linear * ((hk * dk + hv * dv) * activation_bytes + 2 * hv * 4)
    shapes = [
        {"name": "ffn_gate_and_up", "K": hidden, "N": intermediate, "matrices": 2},
        {"name": "ffn_down", "K": intermediate, "N": hidden, "matrices": 1},
        {"name": "gdn_qkv", "K": hidden, "N": qkv_width, "matrices": 1},
        {"name": "gdn_z", "K": hidden, "N": hv * dv, "matrices": 1},
        {"name": "gdn_a_and_b", "K": hidden, "N": hv, "matrices": 2},
        {"name": "gdn_out", "K": hv * dv, "N": hidden, "matrices": 1},
        {"name": "lm_head", "K": hidden, "N": vocab, "matrices": 1},
    ]
    if is_moe:
        shapes.extend([{"name": "router", "K": hidden, "N": experts, "matrices": 1}])
        if shared:
            shapes.extend([
                {"name": "shared_expert_gate_and_up", "K": hidden, "N": shared, "matrices": 2},
                {"name": "shared_expert_down", "K": shared, "N": hidden, "matrices": 1},
            ])
    cells = []
    for concurrency in (1, 4):
        for context in CONTEXTS:
            kv = concurrency * context * kv_per_token
            state = concurrency * committed_state
            conv_bytes = concurrency * conv_state
            known = kv + state + conv_bytes
            total = None
            if resident_weight_bytes is not None and workspace_reserve_bytes is not None:
                total = known + resident_weight_bytes + workspace_reserve_bytes
            cells.append({
                "concurrency": concurrency, "context_tokens": context,
                "bf16_or_selected_width_kv_bytes": kv,
                "committed_gdn_state_bytes": state, "convolution_state_bytes": conv_bytes,
                "known_cache_bytes": known, "caller_budget_total_bytes": total,
                "fits_caller_estimate_only": None if total is None else total <= budget_bytes,
                "safe_to_run": None,
            })
    return {
        "schema": 1, "status": "planning_only_not_a_benchmark_or_admission_guard",
        "config_content_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
        "model_type": config.get("model_type"), "kind": "moe" if is_moe else "dense",
        "geometry": {"hidden": hidden, "intermediate": intermediate, "layers": layers,
                     "linear_layers": linear, "full_attention_layers": full,
                     "experts": experts, "top_k": selected,
                     "shared_expert_intermediate": shared},
        "assumptions": {"kv_element_bytes": kv_bytes, "activation_element_bytes": activation_bytes,
                        "gdn_state_element_bytes": state_bytes, "kv_quantization": "not modeled",
                        "budget_bytes": budget_bytes,
                        "resident_weight_bytes": resident_weight_bytes,
                        "workspace_reserve_bytes": workspace_reserve_bytes},
        "one_request": {"gdn_state_bytes": committed_state,
                        "convolution_state_bytes": conv_state,
                        "kv_bytes_per_token": kv_per_token,
                        "k_v_g_beta_journal_bytes_per_position": journal_per_position},
        "verification_rows": VERIFY_ROWS,
        "projection_shapes": shapes,
        "required_precision_coverage": ["affine4", "affine6", "affine8", "nvfp4_where_present"],
        "current_indirect_pilot_coverage": ["affine4", "affine8"],
        "state_strategy_accounting": [
            {"verified_positions": n,
             "extra_full_state_snapshots_bytes": n * committed_state,
             "k_v_g_beta_journal_bytes": n * journal_per_position,
             "note": "Alternative strategies, not a statement of current runtime allocation"}
            for n in VERIFY_ROWS],
        "workload_cells": cells,
        "required_profiles": ["single_request", "serving_c4", "prefix_hit", "prefix_miss",
                              "cancellation", "ragged_batch", "forced_rejection_positions"],
        "limits": "No observed physical footprint, GPU capability, timing or correctness result; "
                  "journal accounting omits conv-history and live Q/output/workspace; "
                  "draft weights must be included in caller budget; do not change quantization to pass a gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, default=90_000_000_000)
    parser.add_argument("--resident-weight-bytes", type=int)
    parser.add_argument("--workspace-reserve-bytes", type=int)
    args = parser.parse_args()
    try:
        raw = json.loads(args.config.read_text())
        manifest = make_manifest(raw, budget_bytes=args.budget_bytes,
                                 resident_weight_bytes=args.resident_weight_bytes,
                                 workspace_reserve_bytes=args.workspace_reserve_bytes)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.exit(2, f"Cannot build manifest: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}. No workload was run or admitted.")


if __name__ == "__main__":
    main()
