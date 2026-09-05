"""Emit a standalone Metal translation unit for offline Xcode compilation."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from mlx_nax_indirect import kernel_sources

header, body = kernel_sources()
utils = (root / 'mlx_nax_indirect/kernels/offline_utils.h').read_text()
signature = '''
template<typename T, int GROUP_SIZE, int BITS, int BM, bool ALIGNED_M, bool INDIRECT>
[[kernel]] void pilot(
  const device T* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const device uint32_t* w [[buffer(2)]],
  const constant int* w_shape [[buffer(3)]],
  const device T* scales [[buffer(4)]],
  const device T* biases [[buffer(5)]],
  const device uint32_t* indices [[buffer(6)]],
  const constant int* indices_shape [[buffer(7)]],
  const device uint32_t* rows [[buffer(8)]],
  device T* y [[buffer(9)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''
arguments = '''(const device {t}*, const constant int*, const device uint32_t*,
 const constant int*, const device {t}*, const device {t}*, const device uint32_t*,
 const constant int*, const device uint32_t*, device {t}*, uint3, uint, uint);'''
source = utils + '\n' + header + signature + body + '\n}\n'
for dtype in ('float16_t', 'bfloat16_t'):
    for group in (64, 128):
        for bits in (4, 8):
            for bm in (32, 64):
                for aligned in ('true', 'false'):
                    for indirect in ('true', 'false'):
                        name = f'pilot_{dtype}_{group}_{bits}_{bm}_{aligned}_{indirect}'
                        source += f'\ntemplate [[host_name("{name}")]] [[kernel]] void pilot<{dtype},{group},{bits},{bm},{aligned},{indirect}>' + arguments.format(t=dtype) + '\n'
output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'results/pilot.metal'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(source)
print(output)
