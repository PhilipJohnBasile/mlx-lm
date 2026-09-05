"""Paired timing summaries. The independent sample is one ABBA/BAAB round."""
import math
import numpy as np


def summarize(pairs, *, seed=4):
    values=np.asarray(pairs,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=2 or len(values)<5:
        raise ValueError('At least five positive (reference,candidate) timing pairs required')
    if not np.all(np.isfinite(values)) or np.any(values<=0):
        raise ValueError('Invalid/nonfinite timing observations')
    logs=np.log(values[:,0]/values[:,1])
    rng=np.random.default_rng(seed)
    resamples=rng.integers(0,len(logs),(4000,len(logs)))
    ci=np.exp(np.quantile(logs[resamples].mean(axis=1),(.025,.975)))
    return {'geomean_speedup':float(np.exp(logs.mean())), 'ci95':ci.tolist(),
            'reference_median_ms':float(np.median(values[:,0])),
            'candidate_median_ms':float(np.median(values[:,1])),
            'pairs_ms':values.tolist()}


def calibration_ok(summary,tolerance=.05):
    lo,hi=summary['ci95']
    return math.isfinite(lo) and math.isfinite(hi) and 1-tolerance<=lo<=hi<=1+tolerance


def payload_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
