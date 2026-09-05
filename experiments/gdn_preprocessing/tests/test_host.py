import ast
import itertools
import unittest
from pathlib import Path

import numpy as np
from mlx_gdn_prep.geometry import Geometry, window_location
from mlx_gdn_prep.oracle import direct, reference, round_dtype

ROOT = Path(__file__).resolve().parents[1]


class HostTests(unittest.TestCase):
    def test_geometry_dense_and_moe(self):
        self.assertEqual(Geometry(1, 8, 16, 48).channels, 10240)
        self.assertEqual(Geometry(1, 8, 16, 32).channels, 8192)

    def test_invalid_geometry(self):
        for args in ((0,1,1,1), (1,0,1,1), (1,1,2,3), (1,1,1,1,1),
                     (1,1,1,1,9), (1,1,1,1,4,64), (True,1,1,1),
                     (2**30,2,1,1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Geometry(*args)

    def test_large_address_is_not_truncated(self):
        old, offset = window_location(7, 4095, 10239, 8192, 10240, 3)
        self.assertFalse(old)
        self.assertGreater(offset * 4, 2**31)
        self.assertEqual(offset, (7 * 8192 + 4092) * 10240 + 10239)

    def test_address_bounds(self):
        for pos in (-1, 7):
            with self.assertRaises(ValueError):
                window_location(0, pos, 0, 4, 128, 3)

    def test_exact_window_mapping(self):
        for tokens, taps in itertools.product((1,2,3,4,7), (2,3,4,8)):
            C = 5
            state = np.arange(2 * (taps-1) * C).reshape(2,taps-1,C)
            x = (10000 + np.arange(2*tokens*C)).reshape(2,tokens,C)
            joined = np.concatenate((state,x),axis=1)
            for b,t,c in itertools.product(range(2),range(tokens+taps-1),range(C)):
                old, off = window_location(b,t,c,tokens,C,taps-1)
                self.assertEqual((state if old else x).flat[off], joined[b,t,c])

    def test_semantic_paths(self):
        rng = np.random.default_rng(42)
        for dtype, taps, tokens in itertools.product(("float32","float16","bfloat16"),(2,4,8),(1,3,8)):
            geom=Geometry(2,tokens,1,2,taps)
            x=rng.normal(0,.3,(2,tokens,geom.channels)).astype(np.float32)
            w=rng.normal(0,.2,(geom.channels,taps,1)).astype(np.float32)
            st=rng.normal(0,.1,geom.state_shape).astype(np.float32)
            mask=rng.random((2,tokens))>.4
            lengths=np.array([-5,tokens+999],np.int64)
            for m,l in ((None,None),(mask,None),(None,lengths),(mask,lengths)):
                with self.subTest(dtype=dtype,taps=taps,tokens=tokens,masked=m is not None,lengths=l is not None):
                    a=reference(x,w,st,geom,m,l,dtype)
                    b=direct(x,w,st,geom,m,l,dtype)
                    for u,v in zip(a,b):
                        np.testing.assert_array_equal(u.view(np.uint32),v.view(np.uint32))

    def test_history_selects_raw_inputs_not_activations(self):
        g=Geometry(2,2,1,1)
        x=np.full((2,2,g.channels), 2.,np.float32)
        st=np.full(g.state_shape, -3.,np.float32)
        w=np.ones((g.channels,4,1),np.float32)
        out=direct(x,w,st,g,lengths=np.array([0,1]))[-1]
        np.testing.assert_array_equal(out[0],st[0])
        np.testing.assert_array_equal(out[1,-1],x[1,0])
        np.testing.assert_array_equal(out[1,:2],st[1,1:])

    def test_mask_does_not_zero_old_history(self):
        g=Geometry(1,1,1,1)
        x=np.full((1,1,g.channels),np.nan,np.float32)
        st=np.ones(g.state_shape,np.float32)
        w=np.ones((g.channels,4,1),np.float32)
        out=direct(x,w,st,g,mask=np.zeros((1,1),bool))
        self.assertTrue(np.all(np.isfinite(out[2])))
        self.assertTrue(np.all(out[2] > 0))
        self.assertTrue(np.all(out[3][:,-1] == 0))

    def test_split_chunks_preserve_next_history(self):
        rng=np.random.default_rng(7)
        g=Geometry(2,9,1,2)
        x=rng.normal(size=(2,9,g.channels)).astype(np.float32)
        st=rng.normal(size=g.state_shape).astype(np.float32)
        w=rng.normal(size=(g.channels,4,1)).astype(np.float32)
        for dtype in ("float32","float16","bfloat16"):
            full=reference(x,w,st,g,dtype=dtype)
            history=st
            pieces=[[],[],[]]
            for start,end in ((0,1),(1,3),(3,4),(4,9)):
                part=direct(x[:,start:end],w,history,Geometry(2,end-start,1,2),dtype=dtype)
                for i in range(3): pieces[i].append(part[i])
                history=part[3]
            for i in range(3): np.testing.assert_array_equal(np.concatenate(pieces[i],axis=1),full[i])
            np.testing.assert_array_equal(history,full[3])

    def test_round_bfloat_special_values(self):
        x=np.array([0.,-0.,np.inf,-np.inf,np.nan],np.float32)
        y=round_dtype(x,"bfloat16")
        np.testing.assert_array_equal(y[:4].view(np.uint32),x[:4].view(np.uint32))
        self.assertTrue(np.isnan(y[4]))

    def test_import_does_not_require_mlx_or_install_hook(self):
        import sys
        import mlx_gdn_prep
        self.assertNotIn("mlx.core",sys.modules)
        self.assertEqual(mlx_gdn_prep.prepare.__kwdefaults__["mode"],"reference")

    def test_source_has_no_global_mutation(self):
        source=(ROOT/"mlx_gdn_prep"/"__init__.py").read_text()
        tree=ast.parse(source)
        self.assertNotIn("set_default_device",source)
        self.assertNotIn("setattr",source)
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=="prepare" for n in tree.body))


if __name__=="__main__": unittest.main(verbosity=2)
