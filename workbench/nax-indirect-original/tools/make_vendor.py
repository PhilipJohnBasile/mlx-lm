"""Assemble the pinned MIT-licensed MLX helpers needed by the experiment."""
import hashlib
import json
from pathlib import Path
import re
import sys

src = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[1]
kernels = root / 'mlx_nax_indirect' / 'kernels'
seen = set()
provenance = {}

def record(path):
    data = (src / path).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    provenance[path] = {'git_blob_sha1': blob, 'sha256': hashlib.sha256(data).hexdigest()}
    return data.decode()

def flatten(path):
    if path in seen:
        return ''
    seen.add(path)
    text = record(path)
    text = re.sub(r'^#pragma once\s*$', '', text, flags=re.M)
    def include(m):
        child = m.group(1)
        return flatten(child)
    return '\n// BEGIN ' + path + '\n' + re.sub(r'^#include "([^"]+)"\s*$', include, text, flags=re.M) + '\n// END ' + path + '\n'

nax = 'mlx/backend/metal/kernels/steel/gemm/nax.h'
quant = 'mlx/backend/metal/kernels/quantized_nax.h'
header = flatten(nax)
qt = record(quant)
assert provenance[quant]['git_blob_sha1'] == 'bef2634736838a3f38602a2e0fed7d6b951e16bb'
assert (src / 'mlx/ops.cpp').is_file()
# No function constants: the pilot specializes alignments as template arguments.
qt = qt[:qt.index('template <typename T>\nMETAL_FUNC void adjust_matrix_offsets')]
qt = re.sub(r'^constant bool align_[MNK].*\n', '', qt, flags=re.M)
header += '\n// BEGIN pinned quantized load helpers\n' + qt + '\n'
(kernels / 'upstream_nax.h').write_text(header)
# A separate flattened utils preamble is used only by offline xcrun compilation.
# mx.fast.metal_kernel already supplies this preamble at runtime.
seen.clear()
(kernels / 'offline_utils.h').write_text(flatten('mlx/backend/metal/kernels/utils.h'))
(root / 'LICENSE.upstream').write_text((src / 'LICENSE').read_text())
(root / 'UPSTREAM.json').write_text(json.dumps({
    'repository': 'ml-explore/mlx',
    'commit': 'b6368984b8e02a3fb3ee7986846c0fb85e1fccf7',
    'files': provenance,
}, indent=2) + '\n')
print('Wrote pinned NAX helpers:', len(header), 'bytes')
