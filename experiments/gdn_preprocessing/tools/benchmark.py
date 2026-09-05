#!/usr/bin/env python3
"""Warm preprocessing-only Metal A/B. Not a full-model tokens/sec benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlx_gdn_prep import prepare
from mlx_gdn_prep.timing import summarize,calibration_ok


def gate(ref,out,mx):
    if len(ref)!=len(out):raise RuntimeError('Different number of outputs')
    rows=[]
    dtype=ref[0].dtype
    tol={mx.float32:(3e-6,3e-5),mx.float16:(1e-4,2e-3),mx.bfloat16:(8e-4,1.5e-2)}[dtype]
    for i,(a,b) in enumerate(zip(ref,out)):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('Shape/dtype mismatch')
        itype=mx.uint32 if dtype==mx.float32 else mx.uint16
        exact=bool(mx.array_equal(a.view(itype),b.view(itype)).item())
        finite=bool((mx.all(mx.isfinite(a)) & mx.all(mx.isfinite(b))).item())
        close=bool(mx.allclose(a,b,atol=tol[0],rtol=tol[1]).item())
        if not finite or not close or (i==3 and not exact):
            raise RuntimeError(f'Correctness failed at output {i}')
        rows.append({'output':i,'bitwise_equal':exact,'finite':finite,'within_tolerance':close,
                     'max_abs':float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item())})
    return rows


def collect(a,b,reps,rounds,mx):
    def measure(f):
        mx.synchronize()
        start=time.perf_counter_ns()
        for _ in range(reps):
            out=f();mx.eval(*out)
        return (time.perf_counter_ns()-start)/1e6/reps
    pairs=[]
    for i in range(rounds):
        if i%2==0:
            a1,b1,b2,a2=measure(a),measure(b),measure(b),measure(a)
        else:
            b1,a1,a2,b2=measure(b),measure(a),measure(a),measure(b)
        pairs.append(((a1+a2)/2,(b1+b2)/2))
    return summarize(pairs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',default='dense27b,moe35b')
    p.add_argument('--tokens',default='1,8,128,2048')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--dtype',choices=('float32','float16','bfloat16'),default='bfloat16')
    p.add_argument('--rounds',type=int,default=9);p.add_argument('--reps',type=int,default=20)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.rounds<5 or a.reps<1 or a.batch<1: p.error('rounds>=5, reps>=1, batch>=1 required')
    import mlx.core as mx
    if not mx.metal.is_available():raise RuntimeError('No Metal device; no timings recorded')
    info=mx.device_info(mx.gpu);dtype=getattr(mx,a.dtype)
    mx.set_default_device(mx.gpu);mx.random.seed(774)
    root=Path(__file__).resolve().parents[1]
    report={'scope':'warm preprocessing only; no model weights, no tok/s claim',
            'device':info,'platform':platform.platform(),'python':sys.version,
            'mlx':importlib.metadata.version('mlx'),
            'source_hashes':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in (root/'mlx_gdn_prep').glob('*.py')},'cells':[]}
    for profile in a.profiles.split(','):
        if profile not in ('dense27b','moe35b'):p.error('Unknown profile')
        hk,hv=16,(48 if profile=='dense27b' else 32);C=(2*hk+hv)*128
        for S in map(int,a.tokens.split(',')):
            if S<1:p.error('Token counts must be positive')
            x=(mx.random.normal((a.batch,S,C))*.2).astype(dtype)
            w=(mx.random.normal((C,4,1))*.1).astype(dtype)
            history=(mx.random.normal((a.batch,3,C))*.2).astype(dtype)
            mx.eval(x,w,history)
            f={m:(lambda mode=m:prepare(x,w,history,key_heads=hk,value_heads=hv,mode=mode,stream=mx.gpu)) for m in ('reference','direct','fused')}
            checks={m:gate(f['reference'](),f[m](),mx) for m in ('direct','fused')}
            for fun in f.values():
                for _ in range(6):mx.eval(*fun())
            pre=collect(f['reference'],f['reference'],a.reps,a.rounds,mx)
            timing={m:collect(f['reference'],f[m],a.reps,a.rounds,mx) for m in ('direct','fused')}
            post=collect(f['reference'],f['reference'],a.reps,a.rounds,mx)
            final={m:gate(f['reference'](),f[m](),mx) for m in ('direct','fused')}
            drift=post['reference_median_ms']/pre['reference_median_ms']
            stable=calibration_ok(pre) and calibration_ok(post) and .95<=drift<=1.05
            cell={'profile':profile,'batch':a.batch,'tokens':S,'dtype':a.dtype,
                  'before_checks':checks,'after_checks':final,'aa_before':pre,'aa_after':post,
                  'drift_ratio':drift,'stable':stable,'timings':timing,
                  'latency_candidate':{m:stable and timing[m]['ci95'][0]>1.03 for m in timing},
                  'automatic_enable':False}
            report['cells'].append(cell)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
            print(profile,S,a.dtype,{m:timing[m]['geomean_speedup'] for m in timing},'stable=',stable,flush=True)
    print(a.output)

if __name__=='__main__':main()
