// Derived from MLX b6368984b, affine_gather_qmm_rhs_nax. See LICENSE.upstream.
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

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  const device T* x_base = x;
  if constexpr (!INDIRECT) {
    x += y_row_long * K;
  }
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = align_M ? SM : min(int(SM), max(0, M - (y_row + tm)));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  mlx_nax_route::IndirectRows<TM> row_map;
  if constexpr (INDIRECT) {
    row_map.initialize(rows, y_row + tm, M, source_rows, K, simd_lane_id);
  }

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    // Poison an invalid expert instead of reading outside the weight tensor.
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
    }

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              if constexpr (INDIRECT) {
                row_map.load(Atile, x_base, k * BK + kk1);
              } else if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }

          if constexpr (!INDIRECT) {
            xn += BK;
          }
          loader_w.next();
        }


        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }

