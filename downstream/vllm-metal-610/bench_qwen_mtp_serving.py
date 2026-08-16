#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request

from transformers import AutoTokenizer


def post_stream(url, payload, tokenizer):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    text = ""
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            choice = obj.get("choices", [{}])[0]
            piece = choice.get("text")
            if piece is None:
                piece = choice.get("delta", {}).get("content", "")
            if piece:
                if first is None:
                    first = time.perf_counter()
                text += piece
    end = time.perf_counter()
    if first is None:
        first = end
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return {
        "ttft_s": first - start,
        "latency_s": end - start,
        "decode_s": max(end - first, 1e-9),
        "output_tokens": n_tokens,
    }


def scrape_metrics(base_url):
    try:
        with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
            return response.read().decode()
    except Exception:
        return ""


def counter(text, stem):
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(stem) or "_created" in line:
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except Exception:
            pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--shared-prefix-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = "Apple Silicon inference benchmark. Explain carefully and precisely. " * 180
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    seed = tokenizer.decode(seed_ids[: args.shared_prefix_tokens])
    prompts = [
        seed
        + f"\nCase {i}: give a concise technical explanation of speculative decoding."
        for i in range(args.requests)
    ]

    endpoint = args.base_url + "/v1/completions"
    common = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }

    # Warm kernels and populate the common prefix. Warmup is deliberately unscored.
    post_stream(endpoint, dict(common, prompt=prompts[0], max_tokens=16), tokenizer)
    before = scrape_metrics(args.base_url)

    rows = []
    wall_start = time.perf_counter()
    for prompt in prompts:
        rows.append(post_stream(endpoint, dict(common, prompt=prompt), tokenizer))
    wall_s = time.perf_counter() - wall_start
    after = scrape_metrics(args.base_url)

    total_out = sum(row["output_tokens"] for row in rows)
    total_decode_s = sum(row["decode_s"] for row in rows)
    drafted = max(
        0.0,
        counter(after, "vllm:spec_decode_num_draft_tokens_total")
        - counter(before, "vllm:spec_decode_num_draft_tokens_total"),
    )
    accepted = max(
        0.0,
        counter(after, "vllm:spec_decode_num_accepted_tokens_total")
        - counter(before, "vllm:spec_decode_num_accepted_tokens_total"),
    )

    result = {
        "label": args.label,
        "model": args.model,
        "requests": args.requests,
        "shared_prompt_tokens": len(tokenizer.encode(seed, add_special_tokens=False)),
        "max_output_tokens": args.max_tokens,
        "output_tokens": total_out,
        "wall_s": wall_s,
        "output_tok_s": total_out / total_decode_s if total_decode_s else 0.0,
        "end_to_end_tok_s": total_out / wall_s if wall_s else 0.0,
        "mean_ttft_ms": 1000 * statistics.mean(row["ttft_s"] for row in rows),
        "median_ttft_ms": 1000 * statistics.median(row["ttft_s"] for row in rows),
        "mean_latency_ms": 1000 * statistics.mean(row["latency_s"] for row in rows),
        "mtp_drafted_tokens": drafted,
        "mtp_accepted_tokens": accepted,
        "mtp_acceptance_pct": (100.0 * accepted / drafted) if drafted else None,
        "rows": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("BENCH_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
