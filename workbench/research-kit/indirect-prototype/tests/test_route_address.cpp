// Copyright (c) 2026 Philip John Basile. MIT License.
#include "mlx_nax_indirect/kernels/route_address.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace mlx_nax_route;
static uint64_t checks = 0;

void require(bool ok) {
  ++checks;
  if (!ok) throw std::runtime_error("route-address regression");
}

void test_fragment_layout() {
  std::array<unsigned, 256> coverage{};
  for (uint32_t lane = 0; lane < 32; ++lane) {
    for (uint32_t i = 0; i < 8; ++i) {
      // Literal pinned upstream BaseNAXFrag::get_coord(idx).
      const uint32_t qid = lane >> 2;
      const uint32_t expected_row = ((qid & 4) | ((lane >> 1) & 3)) + (i >> 2) * 8;
      const uint32_t expected_col = ((qid & 2) | (lane & 1)) * 4 + i % 4;
      const auto r = fragment_row(lane, 0, i / 4);
      const auto c = fragment_column(lane, 0, i % 4);
      require(r == expected_row && c == expected_col);
      require(r < 16 && c < 16);
      coverage.at(r * 16 + c)++;
    }
  }
  for (auto count : coverage) require(count == 1);
}

void test_bounds() {
  const auto padding = route_address(nullptr, 8, 8, 1, 64);
  require(!padding.active && !padding.valid);
  const std::array<uint32_t, 3> rows{0, 3, 0xffffffffu};
  require(route_address(rows.data(), 0, 3, 3, 64).valid);
  for (auto i : {1u, 2u}) {
    const auto invalid = route_address(rows.data(), i, 3, 3, 64);
    require(invalid.active && !invalid.valid && invalid.offset == 0);
  }
  const std::array<uint32_t, 1> large{100000u};
  require(route_address(large.data(), 0, 1, 100001, 100000).offset == 10000000000ULL);
}

void test_tiles() {
  std::mt19937 random(0x31415926);
  for (const uint32_t t : {1, 2, 17, 65}) {
    for (const uint32_t count : {8, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
      for (const uint32_t k : {64, 128, 256}) {
        std::vector<uint32_t> x(t * k), rows(count);
        for (auto& v : x) v = random();
        for (auto& row : rows) row = random() % t;
        for (const uint32_t tile_rows : {1, 2}) {
          const auto height = tile_rows * 16;
          for (uint32_t base = 0; base < count; base += height) {
            for (uint32_t offset = 0; offset < k; offset += 32) {
              std::vector<unsigned> coverage(height * 32);
              std::vector<uint32_t> output(height * 32, 0xffffffffu);
              for (uint32_t lane = 0; lane < 32; ++lane) {
                for (uint32_t ir = 0; ir < tile_rows; ++ir) {
                  for (uint32_t h = 0; h < 2; ++h) {
                    const auto r = fragment_row(lane, ir, h);
                    const auto address = route_address(rows.data(), uint64_t(base) + r,
                                                       count, t, k);
                    for (uint32_t ic = 0; ic < 2; ++ic) {
                      for (uint32_t e = 0; e < 4; ++e) {
                        const auto c = fragment_column(lane, ic, e);
                        const auto index = r * 32 + c;
                        require(index < output.size());
                        require(!address.active || address.valid);
                        output.at(index) = address.valid ? x.at(address.offset + offset + c) : 0;
                        coverage.at(index)++;
                      }
                    }
                  }
                }
              }
              for (uint32_t r = 0; r < height; ++r) {
                for (uint32_t c = 0; c < 32; ++c) {
                  const auto expected = base + r < count ? x.at(rows.at(base + r) * k + offset + c) : 0;
                  require(output.at(r * 32 + c) == expected);
                  require(coverage.at(r * 32 + c) == 1);
                }
              }
            }
          }
        }
      }
    }
  }
}

int main() {
  test_fragment_layout();
  test_bounds();
  test_tiles();
  std::cout << "PASS: " << checks << " address/layout checks (CPU, not Metal execution)\n";
}
