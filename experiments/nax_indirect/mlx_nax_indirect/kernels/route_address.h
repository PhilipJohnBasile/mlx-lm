// Copyright (c) 2026 Philip John Basile. MIT License.
#pragma once

#ifdef __METAL_VERSION__
#define ROUTE_DEVICE device
#define ROUTE_INLINE METAL_FUNC
#else
#include <cstdint>
#define ROUTE_DEVICE
#define ROUTE_INLINE inline
#endif

namespace mlx_nax_route {

// Layout of the pinned MLX 16x16 cooperative-tensor fragment.
ROUTE_INLINE uint32_t fragment_row(uint32_t lane, uint32_t fragment_index, uint32_t row_half) {
  const uint32_t qid = lane >> 2;
  return 16 * fragment_index + (qid & 4) + ((lane >> 1) & 3) + 8 * row_half;
}

ROUTE_INLINE uint32_t fragment_column(uint32_t lane, uint32_t fragment_index, uint32_t element) {
  const uint32_t qid = lane >> 2;
  return 16 * fragment_index + ((qid & 2) | (lane & 1)) * 4 + element;
}

struct RowAddress {
  uint64_t offset;
  bool active;
  bool valid;
};

ROUTE_INLINE RowAddress route_address(
    const ROUTE_DEVICE uint32_t* rows,
    uint64_t route,
    uint32_t route_count,
    uint32_t source_count,
    uint32_t stride) {
  if (route >= route_count) {
    return {0, false, false};
  }
  const uint32_t source = rows[route];
  if (source >= source_count) {
    return {0, true, false};
  }
  return {uint64_t(source) * uint64_t(stride), true, true};
}

} // namespace mlx_nax_route

#undef ROUTE_DEVICE
#undef ROUTE_INLINE
