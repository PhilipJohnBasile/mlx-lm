"""Streaming state isolation and exact prefix equivalence for immutable BPE maps."""

import unittest

from shared_bpe import SharedBPE
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    TokenizerWrapper,
)


def tokenizer():
    BPEStreamingDetokenizer.make_byte_decoder()
    vocab = dict(BPEStreamingDetokenizer._byte_decoder)
    backend = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.BPE(vocab, [])))
    return TokenizerWrapper(backend, BPEStreamingDetokenizer, eos_token_ids=[])


class TestSharedBPE(unittest.TestCase):
    def test_prefixes_reset_and_interleaved_streams(self):
        tok = tokenizer()
        prepared = SharedBPE(tok)
        sequences = [
            list(text.encode())
            for text in ("  hello world ", "\n中文🙂 test\n", "", "é é\tend")
        ] + [[0xF0, 0x9F], [0xFF, 32, 65], [256]]
        references = [tok.detokenizer for _ in sequences]
        with prepared.select(None, "shared-bpe"):
            streams = [tok.detokenizer for _ in sequences]
        self.assertTrue(all(s.tokenmap is prepared.tokenmap for s in streams))
        self.assertIsInstance(prepared.tokenmap, tuple)
        self.assertEqual(len({id(s.tokens) for s in streams}), len(streams))
        for repeat in range(2):
            for r, s in zip(references, streams):
                r.reset()
                s.reset()
            for position in range(max(map(len, sequences))):
                for seq, r, s in zip(sequences, references, streams):
                    if position >= len(seq):
                        continue
                    r.add_token(seq[position])
                    s.add_token(seq[position])
                    self.assertEqual(
                        (r.text, r.last_segment, r.tokens),
                        (s.text, s.last_segment, s.tokens),
                    )
            for r, s in zip(references, streams):
                r.finalize()
                s.finalize()
                self.assertEqual(
                    (r.text, r.last_segment, r.tokens),
                    (s.text, s.last_segment, s.tokens),
                )

    def test_reference_and_exception_restore(self):
        tok = tokenizer()
        prepared = SharedBPE(tok)
        with prepared.select(None, "reference"):
            self.assertIs(type(tok.detokenizer), BPEStreamingDetokenizer)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with prepared.select(None, "shared-bpe"):
                raise RuntimeError("injected")
        self.assertIs(tok._detokenizer_class, BPEStreamingDetokenizer)
        self.assertIs(type(tok.detokenizer), BPEStreamingDetokenizer)

    def test_vocabulary_growth_requires_reprepare(self):
        tok = tokenizer()
        prepared = SharedBPE(tok)
        tok.add_tokens(["new vocabulary entry"])
        with self.assertRaisesRegex(ValueError, "changed"):
            with prepared.select(None, "shared-bpe"):
                pass
        replacement = SharedBPE(tok)
        self.assertNotEqual(
            prepared.metadata["tokenmap_sha256"],
            replacement.metadata["tokenmap_sha256"],
        )

    def test_unsupported_and_nested_selection(self):
        tok = tokenizer()
        prepared = SharedBPE(tok)
        with prepared.select(None, "shared-bpe"):
            with self.assertRaisesRegex(ValueError, "nested"):
                with prepared.select(None, "shared-bpe"):
                    pass
        tok._detokenizer_class = NaiveStreamingDetokenizer
        with self.assertRaisesRegex(ValueError, "Only the original"):
            SharedBPE(tok)
        self.assertIs(tok._detokenizer_class, NaiveStreamingDetokenizer)

    def test_two_tokenizers_do_not_share_maps(self):
        a, b = tokenizer(), tokenizer()
        b.add_tokens(["other"])
        pa, pb = SharedBPE(a), SharedBPE(b)
        self.assertIsNot(pa.tokenmap, pb.tokenmap)
        self.assertNotEqual(pa.tokenmap, pb.tokenmap)


if __name__ == "__main__":
    unittest.main()
