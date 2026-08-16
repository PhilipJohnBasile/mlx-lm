from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    file.write_text(text.replace(old, new, 1))


cache_policy_path = "vllm_metal/v1/cache_policy.py"
replace_once(
    cache_policy_path,
    """        per_block_bytes += (
            self._worker.model_runner._cache_policy.qwen_mtp_aux_bytes_per_block()
        )
""",
    """        per_block_bytes += self._qwen_mtp_aux_bytes_per_block()
""",
    "remove private cache-policy reach-through",
)
replace_once(
    cache_policy_path,
    """    def _memory_fraction(self) -> float:
""",
    """    def _qwen_mtp_aux_bytes_per_block(self) -> int:
        \"\"\"Return optional native-Qwen MTP cache bytes for one block.\"\"\"
        reporter = getattr(
            self._worker.model_runner,
            \"qwen_mtp_aux_bytes_per_block\",
            None,
        )
        if not callable(reporter):
            return 0
        aux_bytes = int(reporter())
        if aux_bytes < 0:
            raise ValueError(
                \"Qwen MTP auxiliary cache bytes per block cannot be negative\"
            )
        return aux_bytes

    def _memory_fraction(self) -> float:
""",
    "add optional public MTP memory reporter",
)

runner_path = "vllm_metal/v1/model_runner.py"
replace_once(
    runner_path,
    """    def linear_cache_bytes_per_slot(self) -> int:
""",
    """    def qwen_mtp_aux_bytes_per_block(self) -> int:
        \"\"\"Return native-Qwen MTP KV and boundary-hidden bytes per block.\"\"\"
        return self._cache_policy.qwen_mtp_aux_bytes_per_block()

    def linear_cache_bytes_per_slot(self) -> int:
""",
    "expose Qwen MTP memory contract on the runner",
)

print("Hardened phase-5 Qwen MTP worker memory accounting.")
