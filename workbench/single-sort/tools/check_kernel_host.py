#!/usr/bin/env python3
"""Compile the exact kernel body as C++ and test its indexing, not Metal execution."""

import argparse
import ast
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "mlx_lm/models/_single_sort_moe.py"

HEADER = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using uint = uint32_t;
struct Position { uint x, y, z; };

template<class T> struct Input {
  std::vector<T> data;
  long origin = 0;
  T operator[](long offset) const {
    long loc = origin + offset;
    if (loc < 0 || size_t(loc) >= data.size()) throw std::runtime_error("input OOB");
    return data[size_t(loc)];
  }
};
template<class T> struct Output {
  std::vector<T> data;
  std::vector<unsigned> writes;
  explicit Output(size_t n): data(n), writes(n) {}
  struct Ref {
    Output& o; size_t at;
    void operator=(T value) {
      if (++o.writes.at(at) != 1) throw std::runtime_error("multiple output writers");
      o.data.at(at) = value;
    }
  };
  Ref operator[](size_t at) {
    if (at >= data.size()) throw std::runtime_error("output OOB");
    return {*this, at};
  }
  void complete() const {
    for (auto n : writes) if (n != 1) throw std::runtime_error("unwritten output");
  }
};

template<int TopK, class X, class E, class O, class P, class S, class I>
void kernel(const X& x, const E& experts, const O& order,
            P& packed, S& sorted_experts, I& inverse,
            const int* x_shape, const long* x_strides,
            const int* order_shape, const long* order_strides,
            const long* experts_strides, Position thread_position_in_grid) {
'''

FOOTER = r'''
}

size_t cases = 0, checked_elements = 0;
template<int TopK, class Word, class Expert>
void run_case(int tokens, int width, int layout, int id_stride,
              bool same_expert, std::mt19937& rng) {
  const int routes = tokens * TopK;
  long row_stride = width, column_stride = 1;
  if (layout == 1) { row_stride = width * 2 + 3; column_stride = 2; }
  if (layout == 2) row_stride = -width;
  if (layout == 3) column_stride = -1;
  if (layout == 4) row_stride = 0;
  if (layout == 5) { row_stride = 1; column_stride = tokens; }
  if (layout == 6) { row_stride = -width; column_stride = -1; }
  long lo = std::min(0L, (tokens-1L)*row_stride) + std::min(0L, (width-1L)*column_stride);
  long hi = std::max(0L, (tokens-1L)*row_stride) + std::max(0L, (width-1L)*column_stride);
  Input<Word> x;
  x.origin = -lo;
  x.data.resize(size_t(hi-lo+1));
  // For uint16, odd multiplication visits every 16-bit bit pattern.
  for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = Word(i*0x9e3779b1ULL);
  Input<Expert> experts;
  experts.origin = id_stride < 0 ? long(routes-1)*-id_stride : 0;
  experts.data.resize(size_t(routes-1)*size_t(std::abs(id_stride))+1);
  for (int r=0; r<routes; ++r) {
    const Expert value = same_expert ? Expert(7) : Expert(rng()%17);
    experts.data[size_t(experts.origin + long(r)*id_stride)] = value;
  }
  Input<uint32_t> order;
  order.data.resize(routes);
  std::iota(order.data.begin(), order.data.end(), 0);
  std::shuffle(order.data.begin(), order.data.end(), rng);
  std::stable_sort(order.data.begin(), order.data.end(), [&](auto a, auto b) {
    return experts[long(a)*id_stride] < experts[long(b)*id_stride];
  });
  std::vector<uint32_t> expected_inverse(routes);
  std::iota(expected_inverse.begin(), expected_inverse.end(), 0);
  std::sort(expected_inverse.begin(), expected_inverse.end(), [&](auto a, auto b) {
    return order[a] < order[b];
  });
  Output<Word> packed(size_t(routes)*width);
  Output<Expert> sorted_experts(routes);
  Output<uint32_t> inverse(routes);
  const int x_shape[] = {tokens, 1, width}, order_shape[] = {routes};
  const long x_strides[] = {row_stride, 0, column_stride};
  const long order_strides[] = {1}, experts_strides[] = {id_stride};
  const int grid_width = (width+63)/64*64;
  // Reverse traversal; metadata may execute after all other columns.
  for (int r=routes+2; r>=0; --r) {
    for (int c=grid_width-1; c>=0; --c) {
      kernel<TopK>(x, experts, order, packed, sorted_experts, inverse,
          x_shape, x_strides, order_shape, order_strides, experts_strides,
          Position{uint(c),uint(r),0});
    }
  }
  packed.complete(); sorted_experts.complete(); inverse.complete();
  if (inverse.data != expected_inverse) throw std::runtime_error("inverse mismatch");
  for (int r=0; r<routes; ++r) {
    if (sorted_experts.data[r] != experts[long(order[r])*id_stride])
      throw std::runtime_error("expert mismatch");
    for (int c=0; c<width; ++c) {
      Word expect = x[long(order[r]/TopK)*row_stride + long(c)*column_stride];
      if (packed.data[size_t(r)*width+c] != expect)
        throw std::runtime_error("activation bits mismatch");
    }
  }
  ++cases; checked_elements += packed.data.size() + 2*routes;
}

template<int K, class W, class E>
void sweep(std::mt19937& rng) {
  for (int t : {1,2,7,8,17,65})
    for (int d : {1,3,31,32,33,65,257})
      for (int layout : {0,1,2,3,4,5,6})
        run_case<K,W,E>(t,d,layout,(t%3 == 0 ? -1 : (t%2+1)),t==1,rng);
}
struct VirtualInput {
  uint64_t value;
  uint64_t operator[](long offset) const { return value ^ uint64_t(offset); }
};
struct ConstantInput {
  uint32_t value;
  uint32_t operator[](long) const { return value; }
};
struct Recorder {
  std::map<size_t,uint64_t> values;
  uint64_t& operator[](size_t i) { return values[i]; }
};
void large_offset() {
  const uint row=600000, width=8192;
  const int xs[]={600001,1,int(width)}, os[]={600001};
  const long strides[]={long(width),0,1}, one[]={1};
  VirtualInput x{0xfedcba98}; ConstantInput experts{9}, order{row};
  Recorder p,s,i;
  kernel<1>(x,experts,order,p,s,i,xs,strides,os,one,one,{0,row,0});
  size_t off=size_t(row)*width;
  if (off <= UINT32_MAX || p.values.size()!=1 || p.values.at(off)!=x[long(off)] ||
      s.values.at(row)!=9 || i.values.at(row)!=row)
    throw std::runtime_error("64-bit offset was narrowed");
  ++cases;
}
int main() {
  try {
    std::mt19937 rng(399);
    sweep<1,uint16_t,int32_t>(rng);
    sweep<2,uint16_t,uint32_t>(rng);
    sweep<3,uint32_t,int32_t>(rng);
    sweep<4,uint32_t,uint32_t>(rng);
    sweep<8,uint16_t,int64_t>(rng);
    sweep<16,uint32_t,uint64_t>(rng);
    for (int r : {32767,32768,32769,65535,65536,65537})
      run_case<1,uint16_t,int32_t>(r,3,0,-1,false,rng);
    run_case<3,uint32_t,uint64_t>(10923,33,5,2,false,rng);
    run_case<1,uint16_t,int32_t>(1024,64,0,1,false,rng);
    large_offset();
    std::cout << "PASS " << cases << " host kernel cases; " << checked_elements
              << " output elements checked; exact source body; ASan/UBSan enabled\n";
    std::cout << "NOT a Metal compile, GPU execution, or performance measurement\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n'; return 1;
  }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    args = parser.parse_args()
    tree = ast.parse(MODULE.read_text())
    body = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PACK_SOURCE" for t in n.targets)
    )
    out = ROOT / "validation"
    out.mkdir(exist_ok=True)
    cpp = out / "kernel_host.cpp"
    binary = out / "kernel_host"
    cpp.write_text(HEADER + body + FOOTER)
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise SystemExit(f"C++ compiler not found: {args.cxx}")
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
         "-fno-omit-frame-pointer", str(cpp), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    print(result.stdout, end="")
    print(result.stderr, end="")
    (out / "kernel_host.log").write_text(result.stdout + result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
