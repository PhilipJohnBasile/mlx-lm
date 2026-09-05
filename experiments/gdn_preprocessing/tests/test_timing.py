import unittest
import numpy as np
from mlx_gdn_prep.timing import summarize,calibration_ok,payload_equal

class TimingTests(unittest.TestCase):
    def test_known_ratio(self):
        out=summarize([(2.,1.)]*7)
        self.assertEqual(out['ci95'],[2.,2.]);self.assertEqual(out['geomean_speedup'],2.)
    def test_reject_invalid_timings(self):
        for rows in ([(1.,1.)]*4,[(0.,1.)]*7,[(float('inf'),1.)]*7,[(float('nan'),1.)]*7):
            with self.assertRaises(ValueError):summarize(rows)
    def test_aa_interval_not_only_point_estimate(self):
        self.assertTrue(calibration_ok({'ci95':[.99,1.01]}))
        self.assertFalse(calibration_ok({'ci95':[.80,1.20]}))
        self.assertFalse(calibration_ok({'ci95':[float('nan'),1.]}))
    def test_payload_rejects_signed_zero_and_different_shapes(self):
        self.assertFalse(payload_equal(np.array([0.],np.float32),np.array([-0.],np.float32)))
        self.assertFalse(payload_equal(np.ones((1,2)),np.ones((2,))))

if __name__=='__main__':unittest.main(verbosity=2)
