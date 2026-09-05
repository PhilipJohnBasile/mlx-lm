"""Actual MLX/Metal gates. Running this file without Metal exits nonzero."""
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    if __name__ == '__main__':
        sys.exit('NATIVE NOT RUN: MLX is unavailable')
    raise unittest.SkipTest('MLX unavailable; not a native qualification pass')
if not mx.metal.is_available():
    if __name__ == '__main__':
        sys.exit('NATIVE NOT RUN: no Metal GPU')
    raise unittest.SkipTest('Metal unavailable; not a native qualification pass')

from mlx_gdn_prep import prepare
from mlx_gdn_prep.integration import forward


def bits_equal(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:return False
    integer=mx.uint32 if a.dtype==mx.float32 else mx.uint16
    return bool(mx.array_equal(a.view(integer),b.view(integer)).item())


def assert_outputs(test,a,b,dtype,require_exact=False):
    test.assertEqual(len(a),len(b))
    # Predeclared engineering tolerances; matching payloads are also reported.
    atol,rtol={mx.float32:(3e-6,3e-5),mx.float16:(1e-4,2e-3),mx.bfloat16:(8e-4,1.5e-2)}[dtype]
    for i,(x,y) in enumerate(zip(a,b)):
        test.assertEqual(x.shape,y.shape);test.assertEqual(x.dtype,y.dtype)
        test.assertTrue(bool(mx.all(mx.isfinite(x)).item()))
        test.assertTrue(bool(mx.all(mx.isfinite(y)).item()))
        if i==3 or require_exact:
            test.assertTrue(bits_equal(x,y),f'payload mismatch at output {i}')
        else:
            test.assertTrue(bool(mx.allclose(x,y,atol=atol,rtol=rtol).item()),
                            f'output {i}: max_abs={mx.max(mx.abs(x.astype(mx.float32)-y.astype(mx.float32))).item()}')


class NativeTests(unittest.TestCase):
    def inputs(self,B,S,hk,hv,taps,dtype):
        C=(2*hk+hv)*128
        q=mx.random.normal((B,S,C)).astype(dtype)*.2
        w=mx.random.normal((C,taps,1)).astype(dtype)*.1
        h=mx.random.normal((B,taps-1,C)).astype(dtype)*.2
        mx.eval(q,w,h);return q,w,h

    def test_dtype_geometry_mask_and_lengths(self):
        for dtype,(hk,hv),S,taps in itertools.product((mx.float32,mx.float16,mx.bfloat16),((1,2),(16,32),(16,48)),(1,3,8),(2,4)):
            x,w,h=self.inputs(2,S,hk,hv,taps,dtype)
            mask=mx.arange(S)[None,:] < mx.array([S,max(0,S-1)])[:,None]
            lengths=mx.array([-2,S+999],mx.int64)
            for m,l in ((None,None),(mask,None),(None,lengths),(mask,lengths)):
                with self.subTest(dtype=str(dtype),hk=hk,hv=hv,S=S,taps=taps,mask=m is not None,lengths=l is not None):
                    args=dict(key_heads=hk,value_heads=hv,mask=m,lengths=l,stream=mx.gpu)
                    ref=prepare(x,w,h,**args)
                    for mode in ('direct','fused'):
                        got=prepare(x,w,h,mode=mode,**args)
                        mx.eval(*ref,*got);assert_outputs(self,ref,got,dtype)

    def test_noncontiguous_and_broadcast(self):
        for dtype in (mx.float16,mx.bfloat16,mx.float32):
            x,w,h=self.inputs(2,6,1,2,4,dtype)
            x=x[:,::2,:];w=w[:,::-1,:];h=mx.broadcast_to(h[:1],h.shape)
            lengths=mx.array([2**40,-2**40],mx.int64)
            ref=prepare(x,w,h,key_heads=1,value_heads=2,lengths=lengths,stream=mx.gpu)
            for mode in ('direct','fused'):
                out=prepare(x,w,h,key_heads=1,value_heads=2,lengths=lengths,mode=mode,stream=mx.gpu)
                mx.eval(*out,*ref);assert_outputs(self,ref,out,dtype)

    def test_masked_nan_never_enters_convolution(self):
        x,w,h=self.inputs(1,1,1,2,4,mx.float32)
        x=mx.full(x.shape,float('nan'));mask=mx.array([[False]])
        ref=prepare(x,w,h,key_heads=1,value_heads=2,mask=mask,stream=mx.gpu)
        for mode in ('direct','fused'):
            out=prepare(x,w,h,key_heads=1,value_heads=2,mask=mask,mode=mode,stream=mx.gpu)
            mx.eval(*out,*ref);assert_outputs(self,ref,out,mx.float32)

    def test_chunked_history_then_next_token(self):
        x,w,h=self.inputs(2,13,1,2,4,mx.bfloat16)
        ref=prepare(x,w,h,key_heads=1,value_heads=2,stream=mx.gpu)
        for mode in ('direct','fused'):
            history=h;pieces=[[],[],[]]
            for lo,hi in ((0,1),(1,4),(4,12),(12,13)):
                out=prepare(x[:,lo:hi],w,history,key_heads=1,value_heads=2,mode=mode,stream=mx.gpu)
                for i in range(3):pieces[i].append(out[i])
                history=out[3]
            joined=tuple(mx.concatenate(p,axis=1) for p in pieces)+(history,)
            mx.eval(*ref,*joined);assert_outputs(self,ref,joined,mx.bfloat16)

    def test_every_supported_tap_count(self):
        for taps in range(2,9):
            x,w,h=self.inputs(1,2,1,1,taps,mx.float32)
            ref=prepare(x,w,h,key_heads=1,value_heads=1,stream=mx.gpu)
            for mode in ('direct','fused'):
                out=prepare(x,w,h,key_heads=1,value_heads=1,mode=mode,stream=mx.gpu)
                mx.eval(*out,*ref);assert_outputs(self,ref,out,mx.float32)

    def test_recurrent_layer_and_next_token(self):
        from mlx_lm.models.qwen3_5 import GatedDeltaNet
        class Cache:
            def __init__(self):self.data=[None,None];self.lengths=None;self.offset=0
            def __getitem__(self,k):return self.data[k]
            def __setitem__(self,k,v):self.data[k]=v
            def advance(self,n):self.offset+=n
        for hk,hv in ((1,2),(16,32),(16,48)):
            config=SimpleNamespace(hidden_size=64,linear_num_value_heads=hv,
                linear_num_key_heads=hk,linear_key_head_dim=128,linear_value_head_dim=128,
                linear_conv_kernel_dim=4,rms_norm_eps=1e-6)
            mod=GatedDeltaNet(config);mod.eval()
            x=mx.random.normal((1,5,64))*.1
            for mode in ('direct','fused'):
                a,b=Cache(),Cache()
                for lo,hi in ((0,3),(3,4),(4,5)):
                    ref=mod(x[:,lo:hi],cache=a)
                    got=forward(mod,x[:,lo:hi],cache=b,mode=mode)
                    mx.eval(ref,got,*a.data,*b.data)
                    self.assertTrue(bool(mx.allclose(ref,got,atol=2e-5,rtol=1e-4).item()))
                    self.assertTrue(bits_equal(a[0],b[0]))
                    self.assertTrue(bool(mx.allclose(a[1],b[1],atol=2e-5,rtol=1e-4).item()))
                    self.assertEqual(a.offset,b.offset)


if __name__=='__main__':
    mx.set_default_device(mx.gpu)
    mx.random.seed(20260904)
    print('Actual device:',mx.device_info(mx.gpu),flush=True)
    unittest.main(verbosity=2)
