
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

namespace gdn_prep {
template <typename T>
inline T sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
}

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
void fused(const T* qkv,const int* qkv_shape,const T* weight,const T* history,
 const bool* mask,const int* lengths,const float* scales,T* out_q,T* out_k,
 T* out_v,T* next_history,uint3 threadgroup_position_in_grid,uint thread_index_in_simdgroup) {

const uint lane = thread_index_in_simdgroup;
const uint row = threadgroup_position_in_grid.y;
const int S = qkv_shape[1];
constexpr int H = 2 * HK + HV;
constexpr int C = H * 128;
constexpr int HISTORY = TAPS - 1;
const int head = row % H;
const int t = (row / H) % S;
const int b = (row / H) / S;
T activated[4];
float sumsq = 0.0f;
for (int i = 0; i < 4; ++i) {
    const int c = head * 128 + lane * 4 + i;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; ++tap) {
        const int pos = t + tap;
        T value;
        if (pos < HISTORY) {
            value = history[(ulong(b) * HISTORY + pos) * C + c];
        } else {
            const int src_t = pos - HISTORY;
            bool valid = true;
            if constexpr (HAS_MASK) valid = mask[ulong(b) * S + src_t];
            value = valid ? qkv[(ulong(b) * S + src_t) * C + c] : T(0);
        }
        acc += float(value) * float(weight[ulong(c) * TAPS + tap]);
    }
    const T conv = T(acc);
    const T sig = gdn_prep::sigmoid(conv);
    activated[i] = T(conv * sig);
    const float a = float(activated[i]);
    sumsq += a * a;

    // Only t=0 writes history; output history holds pre-convolution inputs.
    if (t == 0) {
        int end = S;
        if constexpr (HAS_LENGTHS) end = metal::clamp(int(lengths[b]), 0, S);
        for (int j = 0; j < HISTORY; ++j) {
            const int pos = end + j;
            T value;
            if (pos < HISTORY) {
                value = history[(ulong(b) * HISTORY + pos) * C + c];
            } else {
                const int src_t = pos - HISTORY;
                bool valid = true;
                if constexpr (HAS_MASK) valid = mask[ulong(b) * S + src_t];
                value = valid ? qkv[(ulong(b) * S + src_t) * C + c] : T(0);
            }
            next_history[(ulong(b) * HISTORY + j) * C + c] = value;
        }
    }
}
if (head < 2 * HK) {
    const float total = simd_sum(sumsq);
    const float inv = metal::precise::rsqrt(total / 128.0f + 1.0e-6f);
    const T scale = T(scales[head < HK ? 0 : 1]);
    const int h = head < HK ? head : head - HK;
    for (int i = 0; i < 4; ++i) {
        const T normalized = T(float(activated[i]) * inv);
        const T value = T(normalized * scale);
        const ulong offset = ((ulong(b) * S + t) * HK + h) * 128 + lane * 4 + i;
        if (head < HK) out_q[offset] = value;
        else out_k[offset] = value;
    }
} else {
    const int h = head - 2 * HK;
    for (int i = 0; i < 4; ++i) {
        const ulong offset = ((ulong(b) * S + t) * HV + h) * 128 + lane * 4 + i;
        out_v[offset] = activated[i];
    }
}
}

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
void direct(const T* qkv,const int* qkv_shape,const T* weight,const T* history,
 const bool* mask,const int* lengths,const float* scales,T* conv_out,T* next_history,
 uint3 thread_position_in_grid) {
 (void)scales;

const uint c = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const uint b = thread_position_in_grid.z;
const int S = qkv_shape[1], C = qkv_shape[2];
constexpr int HISTORY = TAPS - 1;
if (c >= uint(C) || t >= uint(S) || b >= uint(qkv_shape[0])) return;
float acc = 0.0f;
for (int tap = 0; tap < TAPS; ++tap) {
    const int pos = t + tap;
    T value;
    if (pos < HISTORY) {
        value = history[(ulong(b) * HISTORY + pos) * C + c];
    } else {
        const int src_t = pos - HISTORY;
        bool valid = true;
        if constexpr (HAS_MASK) valid = mask[ulong(b) * S + src_t];
        value = valid ? qkv[(ulong(b) * S + src_t) * C + c] : T(0);
    }
    acc += float(value) * float(weight[ulong(c) * TAPS + tap]);
}
conv_out[(ulong(b) * S + t) * C + c] = T(acc);
if (t == 0) {
    int end = S;
    if constexpr (HAS_LENGTHS) end = metal::clamp(int(lengths[b]), 0, S);
    for (int j = 0; j < HISTORY; ++j) {
        const int pos = end + j;
        T value;
        if (pos < HISTORY) {
            value = history[(ulong(b) * HISTORY + pos) * C + c];
        } else {
            const int src_t = pos - HISTORY;
            bool valid = true;
            if constexpr (HAS_MASK) valid = mask[ulong(b) * S + src_t];
            value = valid ? qkv[(ulong(b) * S + src_t) * C + c] : T(0);
        }
        next_history[(ulong(b) * HISTORY + j) * C + c] = value;
    }
}
}

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
