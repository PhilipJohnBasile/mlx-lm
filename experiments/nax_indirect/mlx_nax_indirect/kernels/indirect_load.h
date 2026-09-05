// Copyright (c) 2026 Philip John Basile. MIT License.
#pragma once

namespace mlx_nax_route {

template <short TileRows>
struct IndirectRows {
  RowAddress addresses[TileRows * 2];
  uint32_t lane;

  METAL_FUNC void initialize(
      const device uint32_t* rows,
      uint32_t first_route,
      uint32_t route_count,
      uint32_t source_count,
      uint32_t stride,
      uint32_t lane_id) thread {
    lane = lane_id;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short h = 0; h < 2; ++h) {
        addresses[2 * i + h] = route_address(
            rows, uint64_t(first_route) + fragment_row(lane, i, h),
            route_count, source_count, stride);
      }
    }
  }

  template <typename T, short TileColumns>
  METAL_FUNC void load(
      thread mlx::steel::NAXTile<T, TileRows, TileColumns>& tile,
      const device T* source,
      uint32_t k_offset) const thread {
    static_assert(mlx::steel::BaseNAXFrag::kFragRows == 16);
    static_assert(mlx::steel::BaseNAXFrag::kFragCols == 16);
    static_assert(mlx::steel::BaseNAXFrag::kElemsPerFrag == 8);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TileColumns; ++j) {
        thread auto& frag = tile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; ++h) {
          const auto address = addresses[2 * i + h];
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 4; ++e) {
            if (address.valid) {
              frag[4 * h + e] = source[
                  address.offset + uint64_t(k_offset) + fragment_column(lane, j, e)];
            } else {
              frag[4 * h + e] = address.active ? T(NAN) : T(0);
            }
          }
        }
      }
    }
  }
};

} // namespace mlx_nax_route
