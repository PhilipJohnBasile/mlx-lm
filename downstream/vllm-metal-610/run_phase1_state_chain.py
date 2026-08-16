from __future__ import annotations

from pathlib import Path


script_path = Path(__file__).with_name("apply_phase1_state_chain.py")
source = script_path.read_text()

replacements = [
    (
        '''"""            mamba_cache_mode=self._runner.cache_config.mamba_cache_mode,
            turboquant=config.turboquant,
"""''',
        '''"""            mamba_cache_mode=mamba_cache_mode,
            sdpa_backend=self._runner.sdpa_backend,
"""''',
    ),
    (
        '''"""            mamba_cache_mode=self._runner.cache_config.mamba_cache_mode,
            num_speculative_blocks=self._num_speculative_blocks(),
            turboquant=config.turboquant,
"""''',
        '''"""            mamba_cache_mode=mamba_cache_mode,
            num_speculative_blocks=self._num_speculative_blocks(),
            sdpa_backend=self._runner.sdpa_backend,
"""''',
    ),
]

for old, new in replacements:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            "phase-1 runtime-constructor compatibility rewrite expected "
            f"one match, found {count}"
        )
    source = source.replace(old, new, 1)

exec(compile(source, str(script_path), "exec"), {"__name__": "__main__"})
