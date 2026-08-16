from __future__ import annotations

from pathlib import Path


base_script = Path(__file__).with_name("apply_phase1_state_chain.py")
exec(compile(base_script.read_text(), str(base_script), "exec"), {"__name__": "__main__"})


def replace_once(path: str, old: str, new: str, label: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    target.write_text(text.replace(old, new, 1))


replace_once(
    "vllm_metal/attention/state/align.py",
    """                if num_scheduled > 1:
                    if num_computed <= 0:
""",
    """                is_speculative_verify = (
                    self._num_speculative_blocks > 0
                    and req_idx < ctx.num_decode_requests
                    and num_scheduled > 1
                )
                if is_speculative_verify:
                    if num_computed <= 0:
""",
    "separate speculative verification from multi-token prefill",
)

replace_once(
    "tests/attention/test_align_gdn_state_manager.py",
    """    def _populate(self, manager, req_ids, tables, positions):
        ctx = PagedAttentionContext(slot_mapping=[])
        manager.populate_step_context(
""",
    """    def _populate(self, manager, req_ids, tables, positions):
        # The production batch packs decode requests first. Mirror that
        # ordering so multi-token decode can be distinguished from chunked
        # prefill without changing the ordinary prefill path.
        num_decode_requests = 0
        for num_computed, _ in positions:
            if num_computed <= 0:
                break
            num_decode_requests += 1
        ctx = PagedAttentionContext(
            slot_mapping=[], num_decode_requests=num_decode_requests
        )
        manager.populate_step_context(
""",
    "make test context expose decode-request boundary",
)

print("Separated speculative GDN verification from chunked prefill.")
