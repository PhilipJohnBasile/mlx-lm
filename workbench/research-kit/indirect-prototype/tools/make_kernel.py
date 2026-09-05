"""Derive the pilot body from the pinned upstream kernel, with checked edits."""
from pathlib import Path
import sys

src = Path(sys.argv[1])
root = Path(__file__).resolve().parents[1]
text = (src / 'mlx/backend/metal/kernels/quantized_nax.h').read_text()
start = text.index('[[kernel]] void affine_gather_qmm_rhs_nax(')
body = text[text.index(' {', start)+2:]
# The sorted-RHS kernel is the final function in this pinned file.
assert body.rstrip().endswith('}')
body = body.rstrip()[:-1]

def replace(old, new, count=1):
    global body
    found = body.count(old)
    assert found == count, (old[:60], found, count)
    body = body.replace(old, new)

replace('  x += y_row_long * K;', '''  const device T* x_base = x;
  if constexpr (!INDIRECT) {
    x += y_row_long * K;
  }''')
replace('  // Do as many matmuls as necessary', '''  mlx_nax_route::IndirectRows<TM> row_map;
  if constexpr (INDIRECT) {
    row_map.initialize(rows, y_row + tm, M, source_rows, K, simd_lane_id);
  }

  // Do as many matmuls as necessary''')
replace('    const device T* xn = x + tm * K;', '''    // Poison an invalid expert instead of reading outside the weight tensor.
    if (index >= uint32_t(expert_count)) {
      for (short i = 0; i < Dtile.kElemsPerTile; ++i) {
        Dtile.elems()[i] = NAN;
      }
      Dtile.store_slice(y + tm * N + tn, N,
                       short2(0, m_lo_lim), short2(SN, m_hi_lim));
      continue;
    }

    const device T* xn = x;
    if constexpr (!INDIRECT) {
      xn += tm * K;
    }''')
replace('''              if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }''', '''              if constexpr (INDIRECT) {
                row_map.load(Atile, x_base, k * BK + kk1);
              } else if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }''')
replace('          xn += BK;', '''          if constexpr (!INDIRECT) {
            xn += BK;
          }''')
# The pilot admits only K,N multiples of 64. Remove the unreachable tail
# to avoid carrying the separate upstream ragged-K defect into this experiment.
lo = body.index('        if (!align_K) {')
hi = body.index('\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        // Store', lo)
body = body[:lo] + body[hi:]
preamble = '''// Derived from MLX b6368984b, affine_gather_qmm_rhs_nax. See LICENSE.upstream.
// The only experimental path is INDIRECT=true. K and N must be multiples of 64.
constexpr int group_size = GROUP_SIZE;
constexpr int bits = BITS;
constexpr int BN = 64, BK = 64, WM = 2, WN = 2;
constexpr bool transpose = true, align_N = true, align_M = ALIGNED_M;
const int K = x_shape[1], M = indices_shape[0], N = w_shape[1];
const int source_rows = x_shape[0], expert_count = w_shape[0];
const uint3 tid = threadgroup_position_in_grid;
const uint simd_group_id = simdgroup_index_in_threadgroup;
const uint simd_lane_id = thread_index_in_simdgroup;
'''
(root / 'mlx_nax_indirect/kernels/gather_body.metal').write_text(preamble + body + '\n')
print('Wrote pilot kernel body')
