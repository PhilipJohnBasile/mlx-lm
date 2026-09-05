"""Read-only checkpoint inventory; incomplete downloads are never admitted."""

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path

GEOMETRY = (
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "full_attention_interval",
    "layer_types",
    "rms_norm_eps",
    "rope_parameters",
    "ple_layer_ids",
    "ngram_size",
    "ngram_vocab_size_base",
    "split_ngram_parts",
)


def inspect_model(path):
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    text = config.get("text_config", config)
    quantization = config.get("quantization", config.get("quantization_config", {}))
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    expected = set(index["weight_map"].values()) if index else set()
    files = sorted(path.glob("*.safetensors"))
    for f in files:
        match = re.fullmatch(r"(model)-(\d+)-of-(\d+)\.safetensors", f.name)
        if match:
            expected.update(
                f"model-{i:05d}-of-{int(match[3]):05d}.safetensors"
                for i in range(1, int(match[3]) + 1)
            )
    missing = sorted(f for f in expected if not (path / f).is_file())
    tensors, categories, dtypes = {}, Counter(), Counter()
    invalid_payloads = []
    for f in files:
        with f.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {f}")
            header = json.loads(handle.read(header_size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            if (
                tensor["data_offsets"][0] < 0
                or size < 0
                or tensor["data_offsets"][1] > f.stat().st_size - 8 - header_size
            ):
                invalid_payloads.append({"file": f.name, "tensor": name})
            category = (
                "mtp"
                if "mtp" in name
                else (
                    "vision"
                    if "vision" in name or "visual" in name
                    else "ple" if "ple" in name or "ngram" in name else "body"
                )
            )
            categories[category] += size
            dtypes[tensor["dtype"]] += size
            tensors.setdefault(name, []).append(f.name)
    overrides = Counter(
        (v.get("bits"), v.get("group_size"), v.get("mode", "affine"))
        for v in quantization.values()
        if isinstance(v, dict) and "bits" in v
    )
    provenance = {}
    for name in (
        "ax_provenance.json",
        "native_mtp_adoption.json",
        "mtplx_runtime.json",
    ):
        f = path / name
        if f.exists():
            data = json.loads(f.read_text())
            if name == "mtplx_runtime.json":
                data = {
                    k: data[k]
                    for k in ("arch_id", "base_trunk", "forge_provenance")
                    if k in data
                }
            provenance[name] = data
    wrong_index_files = (
        {k: f for k, f in index["weight_map"].items() if f not in tensors.get(k, [])}
        if index
        else {}
    )
    return {
        "path": str(path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "geometry": {k: text[k] for k in GEOMETRY if k in text},
        "quantization": {
            k: quantization[k]
            for k in ("bits", "group_size", "mode", "quant_method")
            if k in quantization
        },
        "quantization_overrides": [
            {"bits": k[0], "group_size": k[1], "mode": k[2], "modules": v}
            for k, v in overrides.items()
        ],
        "index_present": index is not None,
        "missing_shards": missing,
        "missing_index_tensors": (
            sorted(k for k in index["weight_map"] if k not in tensors)
            if index
            else None
        ),
        "duplicate_tensor_keys": {k: v for k, v in tensors.items() if len(v) > 1},
        "wrong_index_files": wrong_index_files,
        "invalid_payloads": invalid_payloads,
        "complete_weight_inventory": bool(index)
        and not missing
        and not wrong_index_files
        and not invalid_payloads,
        "files": {f.name: f.stat().st_size for f in files},
        "tensor_bytes_by_category": dict(categories),
        "tensor_bytes_by_dtype": dict(dtypes),
        "provenance": provenance,
        "revision_note": "Header and file-extent inventory only. Provenance claims require comparison with remote file hashes; config identity alone does not prove weight identity.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output path")
    result = [inspect_model(path) for path in args.models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for item in result:
        print(
            item["path"],
            "complete:",
            item["complete_weight_inventory"],
            "geometry:",
            item["geometry"].get("model_type"),
            "missing shards:",
            len(item["missing_shards"]),
        )


if __name__ == "__main__":
    main()
