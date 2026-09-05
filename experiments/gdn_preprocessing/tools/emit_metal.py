#!/usr/bin/env python3
"""Emit the actual kernel bodies in standalone MSL translation units."""
import argparse
import hashlib
import urllib.request
import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlx_gdn_prep.kernels import HEADER,FUSED,DIRECT


HEADERS = {
    "bf16.h": "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb",
    "bf16_math.h": "1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496",
}
UPSTREAM = "https://raw.githubusercontent.com/ml-explore/mlx/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/mlx/backend/metal/kernels/"


def read_header(vendor, name):
    """Use bundled headers, or fetch and verify the pinned public source once."""
    path = vendor / name
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(UPSTREAM + name, timeout=20) as response:
            data = response.read(1_048_577)
        if len(data) > 1_048_576:
            raise ValueError("Unexpectedly large upstream header")
    if hashlib.sha256(data).hexdigest() != HEADERS[name]:
        raise ValueError(f"Pinned header hash mismatch: {name}")
    if not path.exists():
        vendor.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data.decode("utf-8").replace("#pragma once", "")


def emit():
    vendor = Path(__file__).resolve().parents[1] / 'reference' / 'vendor'
    source=['#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n']
    source += [read_header(vendor, name) for name in HEADERS]
    source.append(HEADER)
    count=0
    for mode,mask_space,length_space in itertools.product(("direct","fused"),("constant","device"),("constant","device")):
        name=f'gdn_prepare_{mode}_{mask_space[0]}m_{length_space[0]}l'
        outputs = ['out_q','out_k','out_v','next_history'] if mode=='fused' else ['conv_out','next_history']
        argtypes=['const device T*','const constant int*','const device T*','const device T*',
                  f'const {mask_space} bool*',f'const {length_space} int*','const constant float*']
        names=['qkv','qkv_shape','weight','history','mask','lengths','scales']
        params=[f'{typ} {arg} [[buffer({i})]]' for i,(typ,arg) in enumerate(zip(argtypes,names))]
        params += [f'device T* {arg} [[buffer({7+i})]]' for i,arg in enumerate(outputs)]
        argtypes += ['device T*']*len(outputs)
        params += ['uint3 thread_position_in_grid [[thread_position_in_grid]]',
                   'uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]',
                   'uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]']
        argtypes += ['uint3','uint3','uint']
        source.append('template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>\n[[kernel]] void '+name+'('+',\n'.join(params)+') {\n'+(FUSED if mode=='fused' else DIRECT)+'\n}\n')
        for dtype,hv,taps,masked,lengths in itertools.product(('half','bfloat','float'),(32,48),(2,4,8),(False,True),(False,True)):
            tag=f'{name}_{dtype}_h{hv}_t{taps}_m{int(masked)}_l{int(lengths)}'
            templ=f'{dtype},16,{hv},{taps},{str(masked).lower()},{str(lengths).lower()}'
            types=', '.join(x.replace('T*',dtype+'*') for x in argtypes)
            source.append(f'template [[host_name("{tag}")]] [[kernel]] void {name}<{templ}>({types});\n')
            count+=1
    return '\n'.join(source),count


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();source,count=emit()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(source)
    print(f'Emitted {count} kernel instantiations to {args.output}')
