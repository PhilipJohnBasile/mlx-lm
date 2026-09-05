#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Copyright © 2023 Apple Inc.



#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}

// Copyright © 2023 Apple Inc.



///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int& exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(                                   \
    itype, otype, ctype, itype_to_ctype, ctype_to_otype)                     \
                                                                             \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {    \
    return ctype_to_otype(                                                   \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));    \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {           \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));           \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {             \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));             \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {      \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));      \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {        \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));        \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));               \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));               \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

namespace metal {

instantiate_metal_simd_comm_funcs(
    bfloat16_t,
    bfloat16_t,
    uint16_t,
    bfloat16_to_uint16,
    uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal


namespace gdn_prep {
template <typename T>
inline T sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
}

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_direct_cm_cl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const constant bool* mask [[buffer(4)]],
const constant int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* conv_out [[buffer(7)]],
device T* next_history [[buffer(8)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_cl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_cl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_direct_cm_dl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const constant bool* mask [[buffer(4)]],
const device int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* conv_out [[buffer(7)]],
device T* next_history [[buffer(8)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_cm_dl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_cm_dl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_direct_dm_cl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const device bool* mask [[buffer(4)]],
const constant int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* conv_out [[buffer(7)]],
device T* next_history [[buffer(8)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_cl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_cl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_direct_dm_dl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const device bool* mask [[buffer(4)]],
const device int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* conv_out [[buffer(7)]],
device T* next_history [[buffer(8)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_direct_dm_dl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_direct_dm_dl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_fused_cm_cl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const constant bool* mask [[buffer(4)]],
const constant int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* out_q [[buffer(7)]],
device T* out_k [[buffer(8)]],
device T* out_v [[buffer(9)]],
device T* next_history [[buffer(10)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_cl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_cl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_fused_cm_dl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const constant bool* mask [[buffer(4)]],
const device int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* out_q [[buffer(7)]],
device T* out_k [[buffer(8)]],
device T* out_v [[buffer(9)]],
device T* next_history [[buffer(10)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const constant bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const constant bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_cm_dl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_cm_dl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const constant bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_fused_dm_cl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const device bool* mask [[buffer(4)]],
const constant int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* out_q [[buffer(7)]],
device T* out_k [[buffer(8)]],
device T* out_v [[buffer(9)]],
device T* next_history [[buffer(10)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const constant int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const constant int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_cl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_cl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const constant int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template<typename T,int HK,int HV,int TAPS,bool HAS_MASK,bool HAS_LENGTHS>
[[kernel]] void gdn_prepare_fused_dm_dl(const device T* qkv [[buffer(0)]],
const constant int* qkv_shape [[buffer(1)]],
const device T* weight [[buffer(2)]],
const device T* history [[buffer(3)]],
const device bool* mask [[buffer(4)]],
const device int* lengths [[buffer(5)]],
const constant float* scales [[buffer(6)]],
device T* out_q [[buffer(7)]],
device T* out_k [[buffer(8)]],
device T* out_v [[buffer(9)]],
device T* next_history [[buffer(10)]],
uint3 thread_position_in_grid [[thread_position_in_grid]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

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

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,32,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,2,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,2,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,2,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,2,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,4,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,4,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,4,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,4,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,8,false,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,8,false,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,8,true,false>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_half_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<half,16,48,8,true,true>(const device half*, const constant int*, const device half*, const device half*, const device bool*, const device int*, const constant float*, device half*, device half*, device half*, device half*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,32,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,2,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,2,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,2,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,2,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,4,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,4,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,4,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,4,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,8,false,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,8,false,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,8,true,false>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_bfloat_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<bfloat,16,48,8,true,true>(const device bfloat*, const constant int*, const device bfloat*, const device bfloat*, const device bool*, const device int*, const constant float*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h32_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,32,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t2_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,2,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t2_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,2,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t2_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,2,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t2_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,2,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t4_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,4,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t4_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,4,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t4_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,4,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t4_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,4,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t8_m0_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,8,false,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t8_m0_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,8,false,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t8_m1_l0")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,8,true,false>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);

template [[host_name("gdn_prepare_fused_dm_dl_float_h48_t8_m1_l1")]] [[kernel]] void gdn_prepare_fused_dm_dl<float,16,48,8,true,true>(const device float*, const constant int*, const device float*, const device float*, const device bool*, const device int*, const constant float*, device float*, device float*, device float*, device float*, uint3, uint3, uint);
