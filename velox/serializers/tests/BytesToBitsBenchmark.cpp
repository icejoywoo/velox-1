#include "velox/common/base/BitUtil.h"

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <cinttypes>
#include <random>

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#if defined(__AVX512F__) || defined(__AVX512BW__) || defined(__AVX__) || \
    defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace velox = facebook::velox;

/// https://github.com/ClickHouse/ClickHouse/blob/facbd89d4867b246762d8cb82cde2691b2f90200/src/Columns/ColumnsCommon.h#L27
/// Transform 64-byte mask to 64-bit mask
inline uint64_t bytes64MaskToBits64Mask(const uint8_t * bytes64)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i vbytes = _mm512_loadu_si512(reinterpret_cast<const void *>(bytes64));
    uint64_t res = _mm512_testn_epi8_mask(vbytes, vbytes);
#elif defined(__AVX__) && defined(__AVX2__)
    const __m256i zero32 = _mm256_setzero_si256();
    uint64_t res =
        (static_cast<uint64_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bytes64)), zero32))) & 0xffffffff)
        | (static_cast<uint64_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bytes64+32)), zero32))) << 32);
#elif defined(__SSE2__)
    const __m128i zero16 = _mm_setzero_si128();
    uint64_t res =
        (static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64)), zero16))) & 0xffff)
        | ((static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 16)), zero16))) << 16) & 0xffff0000)
        | ((static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
        _mm_loadu_si128(uint64_t<const __m128i *>(bytes64 + 32)), zero16))) << 32) & 0xffff00000000)
        | ((static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 48)), zero16))) << 48) & 0xffff000000000000);
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const uint8x16_t bitmask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    const auto * src = reinterpret_cast<const unsigned char *>(bytes64);
    const uint8x16_t p0 = vceqzq_u8(vld1q_u8(src));
    const uint8x16_t p1 = vceqzq_u8(vld1q_u8(src + 16));
    const uint8x16_t p2 = vceqzq_u8(vld1q_u8(src + 32));
    const uint8x16_t p3 = vceqzq_u8(vld1q_u8(src + 48));
    uint8x16_t t0 = vandq_u8(p0, bitmask);
    uint8x16_t t1 = vandq_u8(p1, bitmask);
    uint8x16_t t2 = vandq_u8(p2, bitmask);
    uint8x16_t t3 = vandq_u8(p3, bitmask);
    uint8x16_t sum0 = vpaddq_u8(t0, t1);
    uint8x16_t sum1 = vpaddq_u8(t2, t3);
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    uint64_t res = vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
#else
    uint64_t res = 0;
    for (size_t i = 0; i < 64; ++i)
        res |= static_cast<uint64_t>(0 == bytes64[i]) << i;
#endif
    return ~res;
}

// Usage example.
// int main() {
//   uint8_t bytes64[64] = {
//     0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
//     1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
//     1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
// };
//   auto result = bytes64MaskToBits64Mask(bytes64);
//   printf("%lu\n", result);
//   return 0;
// }

constexpr int n = 10'000'000;
std::string input;
std::string output(n, 0);

void init() {
  const size_t numBits = 8 * n; // 定义要生成的二进制位数，8位为1字节
  // 初始化随机数生成器
  std::random_device rd; // 用于获得真随机数种子
  std::mt19937 gen(rd()); // 标准的 mersenne_twister_engine
  std::uniform_int_distribution<> dis(0, 1); // 均匀分布的随机数，范围在0和1之间

  // 生成随机0和1，并构建二进制字符串
  for (size_t i = 0; i < numBits; ++i) {
    input += dis(gen) ? '1' : '0';
  }
}

void bytesToBits(std::string_view bytes, uint8_t* output) {
  static constexpr uint8_t BYTE_MASK = 0b1111'1111;

  size_t count = bytes.size();
  int idx = 0;

  int size = (count & ~0b111);
  for (auto i = 0; i < size; i += 8) {
    uint8_t value = 0xFF;
    value &= bytes[i] == 1 ? 0b1111'1110 : BYTE_MASK;
    value &= bytes[i + 1] == 1 ? 0b1111'1101 : BYTE_MASK;
    value &= bytes[i + 2] == 1 ? 0b1111'1011 : BYTE_MASK;
    value &= bytes[i + 3] == 1 ? 0b1111'0111 : BYTE_MASK;
    value &= bytes[i + 4] == 1 ? 0b1110'1111 : BYTE_MASK;
    value &= bytes[i + 5] == 1 ? 0b1101'1111 : BYTE_MASK;
    value &= bytes[i + 6] == 1 ? 0b1011'1111 : BYTE_MASK;
    value &= bytes[i + 7] == 1 ? 0b0111'1111 : BYTE_MASK;
    output[idx++] = value;
  }

  // Write last null bits
  if ((count & 0b111) > 0) {
    uint8_t value = 0xFF;
    uint8_t bitMask = 0x01;
    for (auto i = count & ~0b111; i < count; ++i) {
      value &= bytes[i] == 1 ? (~bitMask) : BYTE_MASK;
      bitMask <<= 1;
    }
    output[idx] = value;
  }
}

void bytesToBitsSimd(std::string_view bytes, uint8_t* output) {
  static constexpr uint8_t BYTE_MASK = 0b1111'1111;

  size_t count = bytes.size();
  int idx = 0;

  int size64 = (count & ~0xFFFFFFFFFFFFFFFFULL);
  for (auto i = 0; i < size64; i += 64) {
    *(uint64_t*) &output[idx] = bytes64MaskToBits64Mask((const uint8_t*)(&bytes[i]));
    idx += 8;
  }

  int size8 = ((count - size64) & ~0b111);
  for (auto i = 0; i < size8; i += 8) {
    uint8_t value = 0xFF;
    value &= bytes[i] == 1 ? 0b1111'1110 : BYTE_MASK;
    value &= bytes[i + 1] == 1 ? 0b1111'1101 : BYTE_MASK;
    value &= bytes[i + 2] == 1 ? 0b1111'1011 : BYTE_MASK;
    value &= bytes[i + 3] == 1 ? 0b1111'0111 : BYTE_MASK;
    value &= bytes[i + 4] == 1 ? 0b1110'1111 : BYTE_MASK;
    value &= bytes[i + 5] == 1 ? 0b1101'1111 : BYTE_MASK;
    value &= bytes[i + 6] == 1 ? 0b1011'1111 : BYTE_MASK;
    value &= bytes[i + 7] == 1 ? 0b0111'1111 : BYTE_MASK;
    output[idx++] = value;
  }

  // Write last null bits
  if ((count & 0b111) > 0) {
    uint8_t value = 0xFF;
    uint8_t bitMask = 0x01;
    for (auto i = count & ~0b111; i < count; ++i) {
      value &= bytes[i] == 1 ? (~bitMask) : BYTE_MASK;
      bitMask <<= 1;
    }
    output[idx] = value;
  }
}

void check() {
  std::string output1(n, 0);
  bytesToBits(input, reinterpret_cast<uint8_t*>(output1.data()));
  std::string output3(n, 0);
  bytesToBitsSimd(input, reinterpret_cast<uint8_t*>(output3.data()));
  if (output1 !=  output3) {
    throw std::runtime_error("encodeAsBitsV1 != encodeAsBitsV3");
  }
}

BENCHMARK(encode_v1) {
  bytesToBits(input, reinterpret_cast<uint8_t*>(output.data()));
}

BENCHMARK_RELATIVE(encode_v3) {
  bytesToBitsSimd(input, reinterpret_cast<uint8_t*>(output.data()));
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  init();
  check();
  folly::runBenchmarks();
  return 0;
}

