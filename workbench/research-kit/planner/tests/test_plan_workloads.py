import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plan_workloads import make_manifest


def example(name):
    return json.loads((ROOT / 'examples' / (name + '.extracted.json')).read_text())


class TestPlanner(unittest.TestCase):
    def test_dense_state_and_kv(self):
        for name in ('Qwen3.6-27B', 'Qwen3.8-27B'):
            m = make_manifest(example(name))
            self.assertEqual(m['one_request']['gdn_state_bytes'], 144 * 2**20)
            self.assertEqual(m['one_request']['kv_bytes_per_token'], 64 * 2**10)
            cell = next(x for x in m['workload_cells'] if x['context_tokens'] == 131072 and x['concurrency'] == 1)
            self.assertEqual(cell['bf16_or_selected_width_kv_bytes'], 8 * 2**30)
            self.assertIsNone(cell['safe_to_run'])

    def test_moe_state_and_shared_expert(self):
        m = make_manifest(example('Qwen3.6-35B-A3B'))
        self.assertEqual(m['one_request']['gdn_state_bytes'], 60 * 2**20)
        self.assertEqual(m['one_request']['kv_bytes_per_token'], 20 * 2**10)
        self.assertEqual(m['geometry']['experts'], 256)
        self.assertEqual(m['geometry']['intermediate'], 512)
        self.assertIn('shared_expert_down', [x['name'] for x in m['projection_shapes']])

    def test_c4_128k_dense(self):
        m = make_manifest(example('Qwen3.8-27B'))
        cell = m['workload_cells'][-1]
        self.assertEqual(cell['bf16_or_selected_width_kv_bytes'], 32 * 2**30)
        self.assertIsNone(cell['fits_caller_estimate_only'])

    def test_budget_is_not_safety_claim(self):
        m = make_manifest(example('Qwen3.8-27B'), budget_bytes=1,
                          resident_weight_bytes=1, workspace_reserve_bytes=1)
        self.assertTrue(all(not c['fits_caller_estimate_only'] for c in m['workload_cells']))
        self.assertTrue(all(c['safe_to_run'] is None for c in m['workload_cells']))

    def test_element_width_change_explicit(self):
        m = make_manifest(example('Qwen3.8-27B'), state_bytes=2, kv_bytes=4)
        self.assertEqual(m['one_request']['gdn_state_bytes'], 72 * 2**20)
        self.assertEqual(m['one_request']['kv_bytes_per_token'], 128 * 2**10)

    def test_actual_layer_types_win(self):
        raw = example('Qwen3.8-27B')
        raw['text_config']['layer_types'] = ['full_attention'] * 64
        m = make_manifest(raw)
        self.assertEqual(m['geometry']['linear_layers'], 0)
        self.assertEqual(m['one_request']['gdn_state_bytes'], 0)

    def test_missing_geometry_fails(self):
        raw = example('Qwen3.8-27B')
        del raw['text_config']['linear_num_key_heads']
        with self.assertRaises(ValueError):
            make_manifest(raw)

    def test_unsupported_architecture_fails(self):
        raw = example('Qwen3.8-27B')
        raw['text_config']['layer_types'] = ['mla'] * 64
        with self.assertRaises(ValueError):
            make_manifest(raw)

    def test_topk_rejects_impossible_geometry(self):
        raw = example('Qwen3.6-35B-A3B')
        raw['text_config']['num_experts_per_tok'] = 257
        with self.assertRaises(ValueError):
            make_manifest(raw)

    def test_dense_projection_geometry(self):
        m = make_manifest(example('Qwen3.8-27B'))
        shape = next(x for x in m['projection_shapes'] if x['name'] == 'ffn_gate_and_up')
        self.assertEqual((shape['K'], shape['N']), (5120, 17408))
        self.assertIn(9, m['verification_rows'])
        self.assertIn('affine6', m['required_precision_coverage'])
        self.assertNotIn('affine6', m['current_indirect_pilot_coverage'])

    def test_journal_estimate_is_not_full_snapshot(self):
        m = make_manifest(example('Qwen3.8-27B'))
        n = next(x for x in m['state_strategy_accounting'] if x['verified_positions'] == 8)
        self.assertEqual(n['extra_full_state_snapshots_bytes'], 8 * 144 * 2**20)
        self.assertEqual(n['k_v_g_beta_journal_bytes'], 8 * 48 * ((16 * 128 + 48 * 128) * 2 + 2 * 48 * 4))

    def test_root_config_accepted(self):
        raw = example('Qwen3.8-27B')['text_config']
        self.assertEqual(make_manifest(raw)['geometry']['hidden'], 5120)


if __name__ == '__main__':
    unittest.main()
