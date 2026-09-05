"""Pretrained GDN diagnostic. Performance qualification uses a separate runner."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
from gdn import select_gdn
from mlx.utils import tree_flatten

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# Declared before pretrained testing. No tolerance applies to pure data movement.
TOLERANCES = {
    "logits": {"atol": 0.05, "rtol": 0.01},
    "cache": {"atol": 0.005, "rtol": 0.02},
}


def compare(a, b, kind):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError("Shape or dtype changed")
    finite = bool((mx.all(mx.isfinite(a)) & mx.all(mx.isfinite(b))).item())
    aa, bb = a.astype(mx.float32), b.astype(mx.float32)
    error = float(mx.max(mx.abs(aa - bb)).item()) if a.size else 0.0
    tol = TOLERANCES[kind]
    close = bool(mx.allclose(aa, bb, **tol).item())
    integer = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    exact = bool(mx.array_equal(a.view(integer), b.view(integer)).item())
    return {
        "finite": finite,
        "within_tolerance": close,
        "bitwise_equal": exact,
        "max_abs": error,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }


def parity(model, ids, mode):
    caches = [model.make_cache(), model.make_cache()]
    rows = []
    # Chunked prompt, multi-token continuation and following decode steps.
    boundaries = [0, 1, 8, len(ids) - 2, len(ids) - 1, len(ids)]
    for lo, hi in zip(boundaries, boundaries[1:]):
        x = mx.array(ids[lo:hi])[None]
        outputs = []
        for arm, selected in enumerate(("reference", mode)):
            with select_gdn(model, selected):
                outputs.append(model(x, cache=caches[arm]))
                mx.eval(outputs[-1], [c.state for c in caches[arm]])
        row = {"range": [lo, hi], "logits": compare(*outputs, "logits"), "cache": []}
        row["argmax_equal"] = bool(
            mx.array_equal(mx.argmax(outputs[0], -1), mx.argmax(outputs[1], -1)).item()
        )
        for i, (a, b) in enumerate(zip(*caches)):
            for (name, av), (_, bv) in zip(
                tree_flatten(a.state), tree_flatten(b.state)
            ):
                if isinstance(av, mx.array) and av.size:
                    entry = compare(av, bv, "cache")
                    entry.update(layer=i, name=name)
                    row["cache"].append(entry)
            if getattr(a, "offset", None) != getattr(b, "offset", None):
                raise AssertionError("Cache offsets differ")
        row["passed"] = row["argmax_equal"] and all(
            v["finite"] and v["within_tolerance"]
            for v in [row["logits"]] + row["cache"]
        )
        rows.append(row)
    return rows


def request(model, tokenizer, prompt, mode, count):
    mx.synchronize()
    start = time.perf_counter()
    times, tokens, text = [], [], []
    with select_gdn(model, mode):
        for r in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=count,
            sampler=make_sampler(temp=0),
            prefill_step_size=512,
        ):
            times.append(time.perf_counter() - start)
            tokens.append(r.token)
            text.append(r.text)
    mx.synchronize()
    total = time.perf_counter() - start
    return {
        "mode": mode,
        "tokens": tokens,
        "text": "".join(text),
        "total_s": total,
        "ttft_s": times[0],
        "delivery_times_s": times,
        "useful_tokens": sum(t not in tokenizer.eos_token_ids for t in tokens),
        "prompt_tokens": r.prompt_tokens,
        "prompt_tps": r.prompt_tps,
        "decode_tps": (
            (len(tokens) - 1) / (times[-1] - times[0]) if len(tokens) > 1 else None
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", default="direct,fused")
    p.add_argument("--max-tokens", type=int, default=32)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Use a fresh output path")
    report = {
        "model": str(args.model),
        "tolerances": TOLERANCES,
        "performance_qualified": False,
        "results": [],
    }

    def write():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))

    write()
    try:
        start = time.perf_counter()
        model, tokenizer = load(str(args.model))
        report["load_s"] = time.perf_counter() - start
        report["device"] = mx.device_info(mx.gpu)
        report["gdn_geometry"] = [
            {
                "layer": i,
                "key_heads": l.linear_attn.num_k_heads,
                "value_heads": l.linear_attn.num_v_heads,
                "dtype": str(l.linear_attn.conv1d.weight.dtype),
                "projection_bits": getattr(l.linear_attn.in_proj_qkv, "bits", None),
            }
            for i, l in enumerate(model.layers)
            if l.is_linear
        ]
        raw = "Explain how a hash table resolves collisions. Give a concrete example and discuss time complexity."
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt)
        report["prompt"] = prompt
        report["prompt_ids"] = ids
        print("Loaded", args.model.name, "prompt tokens", len(ids), flush=True)
        for mode in args.modes.split(","):
            row = {"mode": mode, "parity": parity(model, ids, mode)}
            report["results"].append(row)
            write()
            print(
                mode,
                "parity",
                all(r["passed"] for r in row["parity"]),
                "max_logit_error",
                max(r["logits"]["max_abs"] for r in row["parity"]),
                flush=True,
            )
            if not all(r["passed"] for r in row["parity"]):
                continue
            request(model, tokenizer, prompt, "reference", 4)
            request(model, tokenizer, prompt, mode, 4)
            row["requests"] = [
                request(model, tokenizer, prompt, selected, args.max_tokens)
                for selected in ("reference", mode, mode, "reference")
            ]
            row["generated_tokens_equal"] = all(
                r["tokens"] == row["requests"][0]["tokens"] for r in row["requests"]
            )
            print(
                mode,
                "tokens_equal",
                row["generated_tokens_equal"],
                "request_s",
                [round(r["total_s"], 3) for r in row["requests"]],
                flush=True,
            )
            write()
        report["completed"] = True
        write()
    except Exception as exc:
        report["error"] = repr(exc)
        write()
        raise


if __name__ == "__main__":
    main()
