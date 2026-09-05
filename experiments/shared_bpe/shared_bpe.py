"""Explicit immutable BPE token-map reuse for frozen, loaded tokenizers."""

import hashlib
import json
import time
from contextlib import contextmanager

from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper


class SharedBPEStreamingDetokenizer(BPEStreamingDetokenizer):
    def __init__(self, tokenmap):
        self.tokenmap = tokenmap
        self.reset()
        self.make_byte_decoder()


class SharedBPE:
    """Prepare once for a frozen tokenizer; every stream owns fresh decode state.

    Vocabulary or tokenizer backend replacement requires a new preparation.
    This opt-in experiment does not intercept tokenizer mutation APIs.
    """

    def __init__(self, tokenizer):
        if type(tokenizer) is not TokenizerWrapper:
            raise ValueError("The inspected TokenizerWrapper is required")
        if tokenizer._detokenizer_class is not BPEStreamingDetokenizer:
            raise ValueError("Only the original BPE streaming detokenizer is supported")
        start = time.perf_counter_ns()
        self.tokenizer = tokenizer
        self.original = tokenizer._detokenizer_class
        self.backend = tokenizer._tokenizer
        self.vocab_size = len(self.backend)
        self.tokenmap = tuple(self.original(tokenizer).tokenmap)
        self.metadata = {
            "vocab_size": self.vocab_size,
            "tokenmap_sha256": hashlib.sha256(
                json.dumps(self.tokenmap, ensure_ascii=False).encode()
            ).hexdigest(),
            "scope": "Frozen tokenizer; immutable vocabulary shared, stream state independent. Preparation is once per loaded tokenizer and excluded from warm request latency.",
        }

        self.metadata["preparation_ms"] = (time.perf_counter_ns() - start) / 1e6

    def new_stream(self, tokenizer):
        return SharedBPEStreamingDetokenizer(self.tokenmap)

    @contextmanager
    def select(self, model, mode):
        if mode not in ("reference", "shared-bpe"):
            raise ValueError("Expected reference or shared-bpe")
        if (
            self.tokenizer._tokenizer is not self.backend
            or len(self.backend) != self.vocab_size
            or self.tokenizer._detokenizer_class is not self.original
        ):
            raise ValueError("Tokenizer changed or selection is nested; prepare again")
        if mode == "reference":
            yield
            return
        try:
            self.tokenizer._detokenizer_class = self.new_stream
            yield
        finally:
            self.tokenizer._detokenizer_class = self.original


def make_selector(model, tokenizer):
    prepared = SharedBPE(tokenizer)
    return prepared.select, prepared.metadata
