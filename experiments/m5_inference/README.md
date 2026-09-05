# Physical M5 inference qualification

This is active development alongside the immutable handoff in
`experiments/gdn_preprocessing` and `workbench`. It loads real checkpoints and
selects the handoff preprocessing kernels on one model instance at a time.
It changes no installed inference service, quantization, model file or default
MLX-LM path. The reference mode uses the original model layer. Selection restores
the original layers on normal exit and on exceptions; projection arrays are shared.

## Runtime and provenance

Initial physical run: Apple M5 Max (40 GPU cores), 128 GiB, macOS 27.0
build 26A5425a, Xcode 27.0 build 27A5237l, Metal 32023.921, Python 3.13.14,
MLX/MLX-Metal 0.32.2 and checkout MLX-LM 0.32.0. The isolated environment is
`.venv` in this checkout. Xcode's missing Metal toolchain was installed before
compilation. No normal inference installation was replaced.

Handoff base: `5db58608fa5d21d2613985cefe94a243420ccd63`.
Current upstream inspected on 2026-09-05:

- MLX: `2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99`.
- MLX-LM: `6d21ce4b065a2e163fa6de76a9936c61aeb5784a`.
- The checkout's `qwen3_5.py`, `gated_delta.py` and `switch_layers.py` match
  that MLX-LM upstream revision byte for byte.

MLX-LM already implements packed GDN recurrence and compiled SwiGLU. Related
work includes open core-MLX PRs [#4020](https://github.com/ml-explore/mlx/pull/4020)
and [#4409](https://github.com/ml-explore/mlx/pull/4409), and the closed, unmerged
paired projection proposal [MLX-LM #1676](https://github.com/ml-explore/mlx-lm/pull/1676).
This experiment does not claim those mechanisms as new work.

## Reproduction

From the existing checkout, create the isolated environment once:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python 'mlx==0.32.2' -e .
python3 workbench/verify_inventory.py
```

Run the original inspected qualifier from its own directory, with a fresh output:

```sh
cd experiments/gdn_preprocessing
../../.venv/bin/python tools/qualify_m5.py \
  --tokens 1,8,128,2048 --dtype bfloat16 --batch 1 \
  --output-dir ../../work/m5-qualification/gdn-original-NEW
```

From the checkout root, inspect the actual checkpoint and run the integration:

```sh
.venv/bin/python experiments/m5_inference/inventory.py /path/to/model \
  --output work/m5-qualification/model-inventory-NEW.json
.venv/bin/python experiments/m5_inference/test_inventory.py
.venv/bin/python experiments/m5_inference/test_gdn.py
.venv/bin/python experiments/m5_inference/pilot.py --model /path/to/model \
  --output work/m5-qualification/pilot-NEW.json
.venv/bin/python experiments/m5_inference/benchmark_requests.py \
  --model /path/to/model --mode fused --tokens 32,2048 --concurrency 1,4 \
  --max-tokens 64 --rounds 5 \
  --output work/m5-qualification/requests-NEW
```

Use separate output directories and run GPU workloads sequentially. The full
request runner includes tokenization, the upstream continuous-batching scheduler,
prefill, greedy sampling, single-token decode, streaming detokenization and final
synchronization. It reports warm loaded-model service time; loading is excluded.
Concurrency four uses four distinct, ragged prompts with the same scheduler and
sampling configuration in both arms. Prompt lengths and text are retained.

The runner warms both paths, compares all generated log-probabilities and final
cache tensors, runs A/A controls before and after five alternating ABBA/BAAB
rounds, then repeats the correctness check. Every timed output must retain the
same greedy tokens. It records individual request/token timings, file hashes,
runtime hashes, memory use and swap counters. Swap activity, A/A intervals outside
0.95-1.05, or drift beyond 5% invalidate a cell. Existing allocated swap alone is
not classified as new benchmark-induced swapping.

Arithmetic tolerances are declared before pretrained execution in `pilot.py`:
logits use absolute 0.05 / relative 0.01; cache arrays use absolute 0.005 /
relative 0.02. Bitwise equality is independently reported. Native component
history-copy tests require exact payload bits. No tolerance was changed to pass
the runs. The pilot additionally exercises chunked prompt processing,
multi-token continuation and subsequent single-token steps using separate caches.

## Model admission and limits

The explicit adapter accepts only the inspected `qwen3_5.GatedDeltaNet` class,
single-device inference and 128-wide key/value heads. It rejects unsupported
modules before changing any layer. Dense 27B Hk/Hv=16/48 and MoE Hk/Hv=16/32
have been exercised with pretrained checkpoints. Quantized projections stay
unchanged, including the dense 6-bit checkpoint.

The installed Qwen3.8-Flash-Next MTPLX package is a different architecture. Its
complete store contains about 80.4 GB of body tensors, a separate 32 GB packed
n-gram table, and vision/MTP extras. Its `qwen4_exp_text` identifier alone does
not admit it. The open upstream [Flash PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788),
inspected at `21968365476c7b6add62c2259e0cc4f86ef70704`, does not implement this
package's `ngram-table.safetensors` sidecar loader. Flash remains unqualified.

Speculative drafting/recovery is not modified. These baseline caches do not
establish a qualified speculative path merely because MTP sidecars exist.
Multi-token forward parity is not reported as speculative committed-token speed.
Single-sort routing and indirect NAX remain independent experiments until their
own physical qualification passes. No automatic fast path is enabled.

Raw local evidence is stored under `work/m5-qualification`; final reviewed
results and their hashes should accompany each completed implementation increment.
The first short-prompt pretrained pilots found matching tokens and bitwise
logits/cache states but no useful end-to-end speed gain. Component-only speed
is not a model-performance result.
