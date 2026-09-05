"""Full requests through MLX-LM's scheduler, with paired timing and state gates."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from gdn import select_gdn
from mlx.utils import tree_flatten
from mlx_gdn_prep.timing import calibration_ok, summarize
from pilot import TOLERANCES, compare

from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(*args):
    p = subprocess.run(args, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def memory():
    vm = command("vm_stat")
    values = {
        k.strip(): int(v) for k, v in re.findall(r"^([^:]+):\s+(\d+)\.", vm, re.M)
    }
    pressure = command("memory_pressure", "-Q")
    free = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if free is None:
        raise RuntimeError("Cannot establish memory-pressure headroom")
    return {
        "swapins": values.get("Swapins"),
        "swapouts": values.get("Swapouts"),
        "pageouts": values.get("Pageouts"),
        "swap": command("sysctl", "vm.swapusage"),
        "pressure": pressure,
        "free_percent": int(free[1]),
    }


def model_fingerprint(path):
    files = sorted(
        set(path.glob("model*.safetensors"))
        | set(path.glob("*.json"))
        | set(path.glob("*.jinja"))
    )
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}


def texts(tokenizer, target, concurrency):
    context = (
        "A service stores user records in a hash table. Collisions occur when different keys map to the same bucket. "
        "Separate chaining stores a list in each bucket; open addressing probes other slots. Resizing lowers the load factor. "
    )
    result = []
    for i in range(concurrency):
        content = (
            context * max(1, target // 40)
            + f"\nExplain these tradeoffs and give example {i + 1}."
        )
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Keep the chat suffix intact and deliberately vary the batched lengths.
        ids = tokenizer.encode(full)
        goal = max(16, target - 7 * i)
        if len(ids) > goal:
            ids = ids[: goal - 14] + ids[-14:]
        result.append(tokenizer.decode(ids))
    return result


def request(model, tokenizer, prompts, mode, count, prefill_step, capture=False):
    before = memory()
    if before["free_percent"] < 15:
        raise RuntimeError("Insufficient memory-pressure headroom for timing")
    mx.synchronize()
    start = time.perf_counter_ns()
    with select_gdn(model, mode):
        ids = [tokenizer.encode(p) for p in prompts]
        detokenizers = [tokenizer.detokenizer for _ in prompts]
        gen = BatchGenerator(
            model,
            max_tokens=count,
            sampler=make_sampler(temp=0),
            stop_tokens=[[t] for t in tokenizer.eos_token_ids],
            prefill_batch_size=len(prompts),
            completion_batch_size=len(prompts),
            prefill_step_size=prefill_step,
        )
        uids = gen.insert(ids)
        tokens = {u: [] for u in uids}
        times = {u: [] for u in uids}
        first_text = {u: None for u in uids}
        finished = {}
        caches, logprobs = {}, {u: [] for u in uids}
        try:
            while responses := gen.next_generated():
                for r in responses:
                    if r.finish_reason != "stop":
                        tokens[r.uid].append(r.token)
                        detok = detokenizers[r.uid]
                        detok.add_token(r.token)
                        if detok.last_segment and first_text[r.uid] is None:
                            first_text[r.uid] = (time.perf_counter_ns() - start) / 1e6
                        times[r.uid].append((time.perf_counter_ns() - start) / 1e6)
                    if capture:
                        logprobs[r.uid].append(r.logprobs)
                    if r.finish_reason is not None:
                        finished[r.uid] = {
                            "reason": r.finish_reason,
                            "latency_ms": (time.perf_counter_ns() - start) / 1e6,
                        }
                        if capture:
                            caches[r.uid] = r.prompt_cache
            for d in detokenizers:
                d.finalize()
        finally:
            gen.close()
    mx.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    after = memory()
    swap_delta = {k: after[k] - before[k] for k in ("swapins", "swapouts", "pageouts")}
    if len(finished) != len(prompts) or any(not t for t in times.values()):
        raise RuntimeError("Incomplete request or no useful token delivered")
    useful = sum(map(len, tokens.values()))
    first = statistics.mean(t[0] for t in times.values())
    last = max(t[-1] for t in times.values())
    return {
        "mode": mode,
        "elapsed_ms": elapsed_ms,
        "mean_ttft_ms": first,
        "first_text_ms": first_text,
        "useful_tokens": useful,
        "useful_tps": 1000 * useful / elapsed_ms,
        "decode_tps": (
            1000 * (useful - len(prompts)) / (last - first)
            if useful > len(prompts)
            else None
        ),
        "tokens": tokens,
        "texts": [d.text for d in detokenizers],
        "delivery_ms": times,
        "finished": finished,
        "prompt_tokens": list(map(len, ids)),
        "peak_memory_bytes": mx.get_peak_memory(),
        "swap_delta": swap_delta,
        "memory_valid": all(v == 0 for v in swap_delta.values())
        and after["free_percent"] >= 15,
        "memory_before": before,
        "memory_after": after,
    }, (caches, logprobs)


def parity(a, b):
    rows = []
    ac, al = a
    bc, bl = b
    if ac.keys() != bc.keys() or al.keys() != bl.keys():
        raise AssertionError("Request ids differ")
    for uid in al:
        if len(al[uid]) != len(bl[uid]):
            raise AssertionError("Generated lengths differ")
        for i, (x, y) in enumerate(zip(al[uid], bl[uid])):
            rows.append(
                dict(uid=uid, kind="logprobs", token=i, **compare(x, y, "logits"))
            )
        if len(ac[uid]) != len(bc[uid]):
            raise AssertionError("Cache layer count differs")
        for layer, (ca, cb) in enumerate(zip(ac[uid], bc[uid])):
            xa, xb = tree_flatten(ca.state), tree_flatten(cb.state)
            if len(xa) != len(xb):
                raise AssertionError("Cache structure differs")
            for (name, x), (name_b, y) in zip(xa, xb):
                if name != name_b:
                    raise AssertionError("Cache names differ")
                if isinstance(x, mx.array) and x.size:
                    rows.append(
                        dict(
                            uid=uid,
                            kind="cache",
                            layer=layer,
                            name=name,
                            **compare(x, y, "cache"),
                        )
                    )
            if ca.meta_state != cb.meta_state:
                raise AssertionError("Cache metadata differs")
    return {
        "passed": all(r["finite"] and r["within_tolerance"] for r in rows),
        "bitwise_equal": all(r["bitwise_equal"] for r in rows),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("direct", "fused"), default="fused")
    p.add_argument("--tokens", default="32,2048")
    p.add_argument("--concurrency", default="1,4")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prefill-step-size", type=int, default=512)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    if args.rounds < 5 or args.max_tokens < 2:
        p.error("At least five rounds and two generated tokens are required")
    if args.output.exists():
        p.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    status = {
        "complete": False,
        "performance_qualified": False,
        "cells": [],
        "tolerances": TOLERANCES,
        "scope": "Warm full requests: tokenization, continuous batching, prefill, greedy sampling, decode, detokenization, cache and synchronization. Model load excluded.",
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }

    def write():
        (args.output / "report.json").write_text(
            json.dumps(status, indent=2, allow_nan=False)
        )

    write()
    try:
        if (
            sum(f.stat().st_size for f in args.model.glob("model*.safetensors"))
            > 90_000_000_000
        ):
            raise RuntimeError("Checkpoint exceeds the qualification weight budget")
        if memory()["free_percent"] < 20:
            raise RuntimeError(
                "Insufficient memory-pressure headroom before model load"
            )
        status["model_fingerprint"] = model_fingerprint(args.model)
        status["environment"] = {
            "device": mx.device_info(mx.gpu),
            "os": platform.platform(),
            "python": sys.version,
            "head": command("git", "rev-parse", "HEAD"),
            "diff_sha256": hashlib.sha256(command("git", "diff").encode()).hexdigest(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "metal": command("xcrun", "metal", "--version"),
            "xcode": command("xcodebuild", "-version"),
        }
        sources = (
            list((root / "mlx_lm").rglob("*.py"))
            + list((root / "experiments/m5_inference").glob("*.py"))
            + list((root / "experiments/gdn_preprocessing/mlx_gdn_prep").glob("*.py"))
        )
        status["source_sha256"] = {str(f.relative_to(root)): sha256(f) for f in sources}
        for source in (root / "experiments/m5_inference").glob("*.py"):
            destination = args.output / "source" / source.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(source.read_bytes())
        binary_dir = Path(mx.__file__).resolve().parent
        binaries = (
            [Path(mx.__file__)]
            + list(binary_dir.rglob("*.dylib"))
            + list(binary_dir.rglob("*.metallib"))
        )
        status["binary_sha256"] = {str(f): sha256(f) for f in binaries}
        mx.set_memory_limit(90_000_000_000)
        mx.set_cache_limit(2_000_000_000)
        model, tokenizer = load(str(args.model))
        for target in map(int, args.tokens.split(",")):
            for concurrency in map(int, args.concurrency.split(",")):
                prompts = texts(tokenizer, target, concurrency)
                cell = {
                    "target_prompt_tokens": target,
                    "concurrency": concurrency,
                    "prompts": prompts,
                    "samples": [],
                    "correctness": [],
                    "automatic_enable": False,
                }
                status["cells"].append(cell)

                def run(mode, capture=False):
                    result, tensors = request(
                        model,
                        tokenizer,
                        prompts,
                        mode,
                        args.max_tokens,
                        args.prefill_step_size,
                        capture,
                    )
                    return result, tensors

                # Warm up both paths, then inspect complete generated logits and final caches.
                for mode in ("reference", args.mode):
                    run(mode)

                def gate():
                    a, at = run("reference", True)
                    b, bt = run(args.mode, True)
                    state = parity(at, bt)
                    state["tokens_equal"] = a["tokens"] == b["tokens"]
                    state["texts_equal"] = a["texts"] == b["texts"]
                    state["passed"] &= state["tokens_equal"] and state["texts_equal"]
                    cell["correctness"].append(state)
                    if not state["passed"]:
                        write()
                        raise RuntimeError("Pretrained correctness gate failed")
                    return a["tokens"]

                golden = gate()

                def collect(label, order):
                    pairs = {key: [] for key in ("elapsed_ms", "mean_ttft_ms")}
                    for i in range(args.rounds):
                        modes = (
                            order
                            if i % 2 == 0 or len(order) == 2
                            else [order[1], order[0], order[0], order[1]]
                        )
                        observations = []
                        for j, mode in enumerate(modes):
                            r, _ = run(mode)
                            if r["tokens"] != golden:
                                raise RuntimeError(
                                    "Generated tokens changed during timing"
                                )
                            r.update(phase=label, round=i, position=j)
                            cell["samples"].append(r)
                            observations.append(r)
                            write()
                        if len(order) == 4:
                            arms = (
                                [observations[0], observations[3]],
                                [observations[1], observations[2]],
                            )
                            if i % 2:
                                arms = arms[::-1]
                        else:
                            arms = ([observations[0]], [observations[1]])
                        for key in pairs:
                            pairs[key].append(
                                tuple(
                                    statistics.mean(r[key] for r in arm) for arm in arms
                                )
                            )
                    return {key: summarize(values) for key, values in pairs.items()}

                cell["aa_before"] = collect("aa_before", ["reference", "reference"])
                cell["ab"] = collect(
                    "ab", ["reference", args.mode, args.mode, "reference"]
                )
                cell["aa_after"] = collect("aa_after", ["reference", "reference"])
                gate()
                drift = (
                    cell["aa_after"]["elapsed_ms"]["reference_median_ms"]
                    / cell["aa_before"]["elapsed_ms"]["reference_median_ms"]
                )
                cell["drift_ratio"] = drift
                cell["valid"] = (
                    all(
                        calibration_ok(cell[arm][key])
                        for arm in ("aa_before", "aa_after")
                        for key in ("elapsed_ms", "mean_ttft_ms")
                    )
                    and 0.95 <= drift <= 1.05
                    and all(r["memory_valid"] for r in cell["samples"])
                )
                cell["meets_5pct_throughput_target"] = (
                    cell["valid"] and cell["ab"]["elapsed_ms"]["ci95"][0] >= 1.05
                )
                write()
                print(
                    "cell",
                    target,
                    concurrency,
                    "valid",
                    cell["valid"],
                    "latency",
                    cell["ab"]["elapsed_ms"],
                    flush=True,
                )
        status["complete"] = True
        write()
    except Exception as exc:
        status["error"] = repr(exc)
        write()
        raise


if __name__ == "__main__":
    main()
