"""Metal sources. Native compilation/execution is a separate qualification gate."""

# Keep arithmetic boundaries from MLX b6368984b and MLX-LM 32bb4e687.
HEADER = r'''
namespace gdn_prep {
template <typename T>
inline T sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
}
'''

# Each lane owns four adjacent channels. One SIMD group covers one 128-wide head.
# MASK and LENGTHS are accessed directly, permitting MLX's small constant buffers.
FUSED = r'''
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
'''

# Conservative control: preserve upstream SiLU and normalization as separate ops.
DIRECT = r'''
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
'''
