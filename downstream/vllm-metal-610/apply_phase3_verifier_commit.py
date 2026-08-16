from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def replace_once(path: str, old: str, new: str, label: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    write(path, text.replace(old, new, 1))


runner_path = "vllm_metal/v1/model_runner.py"
replace_once(
    runner_path,
    """        # ---- update decode state ----
        for i, (req_id, state) in enumerate(decode_reqs):
""",
    """        # The target forward has now produced one GDN checkpoint per
        # verification input token, and the verifier has chosen how many output
        # tokens to commit. Promote the matching scheduler-owned recurrent state
        # before request lengths or block ownership advance.
        self._commit_hybrid_speculative_state(
            decode_reqs=decode_reqs,
            decode_segments=decode_segments,
            decode_token_ids=decode_token_ids,
        )

        # ---- update decode state ----
        for i, (req_id, state) in enumerate(decode_reqs):
""",
    "wire verifier result into GDN promotion",
)

replace_once(
    runner_path,
    """    def _validate_spec_decode_supported(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
""",
    """    def _commit_hybrid_speculative_state(
        self,
        *,
        decode_reqs: list[tuple[str, RequestState]],
        decode_segments: tuple[PagedDecodeSegment, ...],
        decode_token_ids: list[list[int]],
    ) -> None:
        \"\"\"Promote verifier-selected state in an align-mode hybrid runtime.

        ``decode_token_ids[i]`` contains the verifier's emitted sequence for
        request ``i``. Its length is the universal recurrent-state selector:
        state snapshot ``len(sampled_ids) - 1``.
        \"\"\"
        runtime = self._paged_attention_runtime
        if runtime is None or not self.is_hybrid:
            return
        if not any(segment.draft_token_ids for segment in decode_segments):
            return
        if self.cache_config.mamba_cache_mode != "align":
            raise RuntimeError(
                "hybrid speculative state promotion requires "
                "mamba_cache_mode='align'"
            )
        if not (
            len(decode_reqs) == len(decode_segments) == len(decode_token_ids)
        ):
            raise RuntimeError("decode speculation metadata length mismatch")

        req_ids: list[str] = []
        state_block_ids: list[list[list[int]]] = []
        step_positions: list[tuple[int, int]] = []
        num_sampled_tokens: list[int] = []
        for (req_id, _), segment, sampled_ids in zip(
            decode_reqs, decode_segments, decode_token_ids, strict=True
        ):
            if not segment.draft_token_ids:
                continue
            if req_id != segment.req_id:
                raise RuntimeError(
                    "hybrid speculative state promotion received mismatched "
                    f"request metadata: {req_id!r} != {segment.req_id!r}"
                )
            sampled = len(sampled_ids)
            if sampled < 1 or sampled > segment.num_query_tokens:
                raise RuntimeError(
                    f"request {req_id!r} emitted {sampled} tokens for a "
                    f"{segment.num_query_tokens}-token verification window"
                )
            try:
                request_state_blocks = self._state_block_ids_by_req[req_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"no scheduler-owned GDN block table for {req_id!r}"
                ) from exc

            req_ids.append(req_id)
            state_block_ids.append(request_state_blocks)
            step_positions.append(
                (segment.cache_start_pos, segment.num_query_tokens)
            )
            num_sampled_tokens.append(sampled)

        if not req_ids:
            return

        commit = getattr(runtime, "commit_speculative_state", None)
        if commit is None:
            raise RuntimeError(
                "hybrid runtime does not implement speculative GDN state promotion"
            )
        commit(
            req_ids=req_ids,
            state_block_ids=state_block_ids,
            step_positions=step_positions,
            num_sampled_tokens=num_sampled_tokens,
        )
        # The promotion copy is a lazy MLX state mutation. Materialize it before
        # the scheduler can reuse, evict, preempt, or copy those blocks.
        runtime.materialize_pending_state()

    def _validate_spec_decode_supported(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
""",
    "add verifier-selected state promotion helper",
)


test_path = "tests/test_v1_model_runner_generate.py"
replace_once(
    test_path,
    """class PoolingForwardBackendStub:
""",
    """class SpeculativeCommitRuntimeStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.materialize_calls = 0

    def commit_speculative_state(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def materialize_pending_state(self) -> None:
        self.materialize_calls += 1


class PoolingForwardBackendStub:
""",
    "add speculative commit runtime test double",
)

insert_marker = """class TestV1MetalModelRunnerGenerate:
"""
new_tests = """class TestHybridSpeculativeStatePromotion:
    @staticmethod
    def _segment(
        req_id: str,
        *,
        start_row: int,
        cache_start_pos: int,
        drafts: tuple[int, ...],
    ) -> PagedDecodeSegment:
        return PagedDecodeSegment(
            req_id=req_id,
            input_token_ids=(9, *drafts),
            start_row=start_row,
            num_query_tokens=1 + len(drafts),
            draft_token_ids=drafts,
            cache_start_pos=cache_start_pos,
            block_ids=((2, 3, 6, 7),),
        )

    def test_commits_only_speculative_requests_with_sampled_lengths(self) -> None:
        runner = make_stub_runner(tokenizer=object())
        runner.model_args = {"full_attention_interval": 2}
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runtime = SpeculativeCommitRuntimeStub()
        runner._paged_attention_runtime = runtime
        runner._state_block_ids_by_req = {
            "spec": [[2, 3, 6, 7]],
            "plain": [[8, 9]],
        }
        spec_segment = self._segment(
            "spec", start_row=0, cache_start_pos=5, drafts=(10, 11)
        )
        plain_segment = self._segment(
            "plain", start_row=3, cache_start_pos=12, drafts=()
        )

        runner._commit_hybrid_speculative_state(
            decode_reqs=[("spec", object()), ("plain", object())],
            decode_segments=(spec_segment, plain_segment),
            decode_token_ids=[[10, 99], [12]],
        )

        assert runtime.materialize_calls == 1
        assert runtime.calls == [
            {
                "req_ids": ["spec"],
                "state_block_ids": [[[2, 3, 6, 7]]],
                "step_positions": [(5, 3)],
                "num_sampled_tokens": [2],
            }
        ]

    def test_rejects_impossible_verifier_output_length(self) -> None:
        runner = make_stub_runner(tokenizer=object())
        runner.model_args = {"full_attention_interval": 2}
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runner._paged_attention_runtime = SpeculativeCommitRuntimeStub()
        runner._state_block_ids_by_req = {"spec": [[2, 3, 6, 7]]}
        segment = self._segment(
            "spec", start_row=0, cache_start_pos=5, drafts=(10, 11)
        )

        with pytest.raises(RuntimeError, match="emitted 0 tokens"):
            runner._commit_hybrid_speculative_state(
                decode_reqs=[("spec", object())],
                decode_segments=(segment,),
                decode_token_ids=[[]],
            )


""" + insert_marker
replace_once(
    test_path,
    insert_marker,
    new_tests,
    "add verifier-to-GDN promotion tests",
)

print("Applied vllm-metal #610 phase-3 verifier promotion patch.")
