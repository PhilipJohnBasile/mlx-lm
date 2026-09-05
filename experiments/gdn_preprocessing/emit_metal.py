"""Compile-only export of the same direct and fused kernel bodies."""
import itertools
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
headers = Path(sys.argv[2])
kernels = runpy.run_path(str(root / 'kernels.py'))
source = ['#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n']
source += [(headers / name).read_text().replace('#pragma once', '') for name in ('bf16.h', 'bf16_math.h')]
source.append(kernels['HEADER'])
count = 0
for mode, mask_space, length_space in itertools.product(('direct', 'fused'), ('constant', 'device'), ('constant', 'device')):
    name = f'gdn_prepare_{mode}_{mask_space[0]}m_{length_space[0]}l'
    outputs = ['out_q', 'out_k', 'out_v', 'next_history'] if mode == 'fused' else ['conv_out', 'next_history']
    types = ['const device T*', 'const constant int*', 'const device T*', 'const device T*', f'const {mask_space} bool*', f'const {length_space} int*', 'const constant float*']
    names = ['qkv', 'qkv_shape', 'weight', 'history', 'mask', 'lengths', 'scales']
    params = [f'{typ} {arg} [[buffer({i})]]' for i, (typ, arg) in enumerate(zip(types, names))]
    params += [f'device T* {arg} [[buffer({7+i})]]' for i, arg in enumerate(outputs)]
    types += ['device T*'] * len(outputs)
    params += ['uint3 thread_position_in_grid [[thread_position_in_grid]]', 'uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]', 'uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]']
    types += ['uint3', 'uint3', 'uint']
    source.append('template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>\n[[kernel]] void ' + name + '(' + ',\n'.join(params) + ') {\n' + kernels[mode.upper()] + '\n}\n')
    for dtype, hv, taps, masked, lengths in itertools.product(('half', 'bfloat', 'float'), (32, 48), (2, 4, 8), (False, True), (False, True)):
        tag = f'{name}_{dtype}_h{hv}_t{taps}_m{int(masked)}_l{int(lengths)}'
        templ = f'{dtype},16,{hv},{taps},{str(masked).lower()},{str(lengths).lower()}'
        signature = ', '.join(x.replace('T*', dtype+'*') for x in types)
        source.append(f'template [[host_name("{tag}")]] [[kernel]] void {name}<{templ}>({signature});\n')
        count += 1
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(source))
print(f'Emitted {count} kernel instantiations to {out}')
