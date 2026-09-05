#!/usr/bin/env python3
"""Execute the actual kernel bodies as C++ with a host SIMD collective shim.

This checks scalar addressing/arithmetic and memory bounds for float32 only.
It cannot validate Metal lowering, GPU scheduling, half/bfloat math or speed.
"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlx_gdn_prep.kernels import HEADER,FUSED,DIRECT

PREAMBLE=r'''
#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
using uint=unsigned int;
using ulong=unsigned long;
struct uint3 { uint x,y,z; };
namespace metal { using std::abs; using std::exp; using std::clamp;
namespace precise { inline float rsqrt(float x) {return 1.0f/std::sqrt(x);} } }
struct Simd { std::barrier<> barrier{32}; std::array<float,32> values{}; };
thread_local Simd* simd;
thread_local int host_lane;
float simd_sum(float x) {
 simd->values[host_lane]=x;simd->barrier.arrive_and_wait();
 for(int bit=1;bit<32;bit*=2) {
  float sum=simd->values[host_lane]+simd->values[host_lane^bit];
  simd->barrier.arrive_and_wait();simd->values[host_lane]=sum;
  simd->barrier.arrive_and_wait();
 }
 return simd->values[host_lane];
}
'''
FUSED_SIGNATURE=r'''
template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
void fused(const T* qkv,const int* qkv_shape,const T* weight,const T* history,
 const bool* mask,const int* lengths,const float* scales,T* out_q,T* out_k,
 T* out_v,T* next_history,uint3 threadgroup_position_in_grid,uint thread_index_in_simdgroup) {
'''
DIRECT_SIGNATURE=r'''
template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
void direct(const T* qkv,const int* qkv_shape,const T* weight,const T* history,
 const bool* mask,const int* lengths,const float* scales,T* conv_out,T* next_history,
 uint3 thread_position_in_grid) {
 (void)scales;
'''
HARNESS=r'''
size_t elements=0;int cases=0;
void check(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
template<int HK,int HV,int TAPS,bool MASK,bool LENGTHS>
void one(int B,int S,int variant) {
 constexpr int H=2*HK+HV,C=H*128,R=TAPS-1;
 std::minstd_rand rng(1200+variant);std::uniform_real_distribution<float> dist(-.4f,.4f);
 std::vector<float>x(size_t(B)*S*C),w(C*TAPS),old(size_t(B)*R*C);
 for(auto&v:x)v=dist(rng);for(auto&v:w)v=dist(rng);for(auto&v:old)v=dist(rng);
 std::unique_ptr<bool[]> mask(new bool[B*S]);std::vector<int> lengths(B);
 for(int b=0;b<B;++b) {lengths[b]=(variant==0?-5:variant==1?S+9:std::max(0,S-b));
 for(int t=0;t<S;++t)mask[b*S+t]=(t+b)%3!=0;}
 if(MASK)for(int b=0;b<B;++b)for(int t=0;t<S;++t)if(!mask[b*S+t])
  for(int c=0;c<C;++c)x[(size_t(b)*S+t)*C+c]=std::numeric_limits<float>::quiet_NaN();
 // Independent materialized-concatenation oracle.
 std::vector<float>joined(size_t(B)*(R+S)*C),ref(x.size()),expected_old(old.size());
 for(int b=0;b<B;++b)for(int p=0;p<R+S;++p)for(int c=0;c<C;++c)
  joined[(size_t(b)*(R+S)+p)*C+c]=p<R?old[(size_t(b)*R+p)*C+c]:
     MASK&&!mask[b*S+p-R]?0.0f:x[(size_t(b)*S+p-R)*C+c];
 for(int b=0;b<B;++b)for(int t=0;t<S;++t)for(int c=0;c<C;++c) {
  float acc=0;for(int j=0;j<TAPS;++j)acc+=joined[(size_t(b)*(R+S)+t+j)*C+c]*w[c*TAPS+j];
  ref[(size_t(b)*S+t)*C+c]=acc;
 }
 for(int b=0;b<B;++b)for(int j=0;j<R;++j)for(int c=0;c<C;++c)
  expected_old[(size_t(b)*R+j)*C+c]=joined[(size_t(b)*(R+S)+(LENGTHS?std::clamp(lengths[b],0,S):S)+j)*C+c];
 std::vector<float>conv(x.size(),NAN),hist(old.size(),NAN);
 int shape[3]={B,S,C};float scale[2]={float(std::pow(128.,-.5)*std::pow(128.,-.5)),float(std::pow(128.,-.5))};
 for(int b=0;b<B;++b)for(int t=0;t<S;++t)for(int c=0;c<C+3;++c)
  direct<float,HK,HV,TAPS,MASK,LENGTHS>(x.data(),shape,w.data(),old.data(),mask.get(),lengths.data(),scale,conv.data(),hist.data(),{uint(c),uint(t),uint(b)});
 check(std::memcmp(ref.data(),conv.data(),ref.size()*sizeof(float))==0,"direct convolution mismatch");
 check(std::memcmp(expected_old.data(),hist.data(),hist.size()*sizeof(float))==0,"direct history mismatch");
 elements+=conv.size()+hist.size();
 std::vector<float>q(size_t(B)*S*HK*128,NAN),k(q.size(),NAN),v(size_t(B)*S*HV*128,NAN);
 std::fill(hist.begin(),hist.end(),NAN);Simd context;std::vector<std::thread>workers;
 for(int lane=0;lane<32;++lane)workers.emplace_back([&,lane]{
  host_lane=lane;simd=&context;
  for(int row=0;row<B*S*H;++row) {
   fused<float,HK,HV,TAPS,MASK,LENGTHS>(x.data(),shape,w.data(),old.data(),mask.get(),lengths.data(),scale,q.data(),k.data(),v.data(),hist.data(),{0,uint(row),0},uint(lane));
   context.barrier.arrive_and_wait();
  }
 });
 for(auto&t:workers)t.join();
 for(auto&x:ref) {float y=1.0f/(1.0f+std::exp(std::abs(x)));x*=x<0?y:1.0f-y;}
 for(int b=0;b<B;++b)for(int t=0;t<S;++t)for(int h=0;h<H;++h) {
  float sum=0;for(int i=0;i<128;++i) {float a=ref[((size_t(b)*S+t)*H+h)*128+i];sum+=a*a;}
  float inv=1.0f/std::sqrt(sum/128.0f+1e-6f);
  for(int i=0;i<128;++i) {
   float expected=ref[((size_t(b)*S+t)*H+h)*128+i];float got;
   if(h<HK){expected=(expected*inv)*scale[0];got=q[((size_t(b)*S+t)*HK+h)*128+i];}
   else if(h<2*HK){expected=(expected*inv)*scale[1];got=k[((size_t(b)*S+t)*HK+h-HK)*128+i];}
   else {got=v[((size_t(b)*S+t)*HV+h-2*HK)*128+i];}
   check(std::isfinite(got)&&std::abs(expected-got)<=2e-6f+2e-6f*std::abs(expected),"fused preparation mismatch");++elements;
  }
 }
 check(std::memcmp(expected_old.data(),hist.data(),hist.size()*sizeof(float))==0,"fused history mismatch");elements+=hist.size();++cases;
}
template<int TAP>void taps(){for(int S:{1,2,7})for(int v=0;v<3;++v){
 one<1,2,TAP,false,false>(2,S,v);one<1,2,TAP,true,false>(2,S,v);
 one<1,2,TAP,false,true>(2,S,v);one<1,2,TAP,true,true>(2,S,v);}}
int main(){taps<2>();taps<4>();taps<8>();
 one<16,32,4,true,true>(2,3,2);one<16,48,4,true,true>(2,3,2);
 std::cout<<"PASS: "<<cases<<" float32 host cases, "<<elements<<" output elements checked; ASan/UBSan. Not Metal execution.\n";}
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    path=a.output_dir/'kernel_body_test.cpp'
    path.write_text(PREAMBLE+HEADER+FUSED_SIGNATURE+FUSED+'}\n'+DIRECT_SIGNATURE+DIRECT+'}\n'+HARNESS)
    binary=a.output_dir/'kernel_body_test'
    subprocess.run(['clang++','-std=c++20','-O1','-g','-pthread','-ffp-contract=off',
       '-fsanitize=address,undefined','-fno-omit-frame-pointer',str(path),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=40)

if __name__=='__main__':main()
