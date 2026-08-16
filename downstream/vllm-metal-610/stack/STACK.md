# vLLM Metal #610 merge-ready stack

This directory is generated from five tested commits based on:

`vllm-project/vllm-metal@083f581f048b1b4460f166b31ec18dc6ec000e17`

Each patch contains exactly one phase and can be reviewed independently. Apply
in the order listed below; phases 1-4 are complete prerequisites and phase 5 is
the final scheduler-owned MTP prefix-cache transaction.

| Phase | Review document | Commit | Patch |
|---:|---|---|---|
| 1 | [Hybrid GDN speculative state-chain planning](PR-01.md) | `62850fd729c1` | `0001-hybrid-gdn-speculative-state-chain.patch` |
| 2 | [Per-token GDN convolution and recurrent snapshots](PR-02.md) | `37bccdfa877a` | `0002-gdn-per-token-state-snapshots.patch` |
| 3 | [Verifier-selected GDN state promotion](PR-03.md) | `8d8a022dfe86` | `0003-verifier-promotes-gdn-state.patch` |
| 4 | [Native Qwen MTP proposer and hidden-state contract](PR-04.md) | `0c7e677098ec` | `0004-native-qwen-mtp-proposer.patch` |
| 5 | [Scheduler-owned paged Qwen MTP prefix transaction](PR-05.md) | `43e83bd80432` | `0005-paged-qwen-mtp-prefix-transaction.patch` |

## Apply the complete series

```bash
git checkout 083f581f048b1b4460f166b31ec18dc6ec000e17
while read patch; do
  git am "/path/to/stack/$patch"
done < /path/to/stack/series
```

## Merge order

1. Phase 1 establishes scheduler memory and state-chain metadata.
2. Phase 2 makes every speculative GDN state observable.
3. Phase 3 commits only the verifier-selected state.
4. Phase 4 connects the trained native Qwen MTP head.
5. Phase 5 adds scheduler-owned MTP KV and safe warm-prefix adoption.

The stack deliberately keeps unsupported configurations fail-closed. Real-model
benchmark claims require the separate Qwen checkpoint integration gate; the
unit and scheduler tests here establish the cache/state invariants.
