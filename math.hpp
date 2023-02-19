#ifndef SCALABLE_VIDEO_CODEC_MATH_HPP
#define SCALABLE_VIDEO_CODEC_MATH_HPP

#include <cassert>
#include <cmath>
#include <numeric>

#include "types.hpp"

inline uint Pow2(uint exp) {
  uint res = 1;
  for (uint i = 0; i < exp; ++i) {
    res <<= 1;
  }
  return res;
}

inline int RoundFloatToInt(float a) {
  int res = std::round(a);
  return res;
}

inline float Abs(float a) {
  float res = std::abs(a);
  return res;
}

inline float Log(float a) {
  float res = std::log(a);
  return res;
}

inline float Ceil(float a) {
  float res = std::ceil(a);
  return res;
}

inline float Pow(float base, float exp) {
  float res = std::pow(base, exp);
  return res;
}

inline float Sqrt(float a) {
  float res = std::sqrt(a);
  return res;
}

inline float Sqr(float a) {
  float res = a * a;
  return res;
}

inline int Sqr(int a) {
  int result = a * a;
  return result;
}

inline uint Sqr(uint a) {
  uint result = a * a;
  return result;
}

inline uchar AbsDiff(uchar x, uchar y) {
  if (x > y) {
    return x - y;
  }
  return y - x;
}

inline float Max(float a, float b) {
  float result = a;
  if (a < b) {
    result = b;
  }
  return result;
}

inline float Min(float a, float b) {
  float result = a;
  if (a > b) {
    result = b;
  }
  return result;
}

inline int Max(int a, int b) {
  int result = a;
  if (a < b) {
    result = b;
  }
  return result;
}

inline int Min(int a, int b) {
  int result = a;
  if (a > b) {
    result = b;
  }
  return result;
}

inline uint Max(uint a, uint b) {
  uint result = a;
  if (a < b) {
    result = b;
  }
  return result;
}

inline uint Min(uint a, uint b) {
  uint result = a;
  if (a > b) {
    result = b;
  }
  return result;
}

struct Vec2i {
  int x;
  int y;
  int& operator[](uint i) { return (&x)[i]; }
};

inline Vec2i operator+(Vec2i a, Vec2i b) {
  Vec2i result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  return result;
}

inline Vec2i operator+=(Vec2i& a, Vec2i b) {
  a = a + b;
  return a;
}

inline Vec2i operator-(Vec2i a, Vec2i b) {
  Vec2i result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

inline Vec2i operator-=(Vec2i& a, Vec2i b) {
  a = a - b;
  return a;
}

inline Vec2i operator*(Vec2i a, int s) {
  Vec2i result;
  result.x = a.x * s;
  result.y = a.y * s;
  return result;
}

inline Vec2i operator*(int s, Vec2i a) {
  Vec2i result;
  result.x = a.x * s;
  result.y = a.y * s;
  return result;
}

inline int Dot(Vec2i a, Vec2i b) {
  int result = a.x * b.x + a.y * b.y;
  return result;
}

struct Vec2ui {
  uint x;
  uint y;
  uint& operator[](uint i) { return (&x)[i]; }
};

inline Vec2ui operator-(Vec2ui a, Vec2ui b) {
  Vec2ui result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

struct Vec2f {
  float x;
  float y;
  float& operator[](uint i) { return (&x)[i]; }
};

inline Vec2f operator+(Vec2f a, Vec2f b) {
  Vec2f result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  return result;
}

inline Vec2f operator+=(Vec2f& a, Vec2f b) {
  a = a + b;
  return a;
}

inline Vec2f operator-(Vec2f a, Vec2f b) {
  Vec2f result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

inline Vec2f operator-=(Vec2f& a, Vec2f b) {
  a = a - b;
  return a;
}

inline Vec2f operator*(Vec2f a, float s) {
  Vec2f result;
  result.x = a.x * s;
  result.y = a.y * s;
  return result;
}

inline Vec2f operator*(float s, Vec2f a) {
  Vec2f result;
  result.x = a.x * s;
  result.y = a.y * s;
  return result;
}

inline Vec2f operator*=(Vec2f& a, float s) {
  a = a * s;
  return a;
}

inline float Dot(Vec2f a, Vec2f b) {
  float result = a.x * b.x + a.y * b.y;
  return result;
}

inline float Magnitude(Vec2f a) {
  float result = std::sqrt(Dot(a, a));
  return result;
}

inline Vec2f Normalize(Vec2f a) {
  Vec2f result = a * (1.0f / Magnitude(a));
  return result;
}

inline Vec2i Vec2fToVec2i(Vec2f a) {
  Vec2i result;
  result.x = RoundFloatToInt(a.x);
  result.y = RoundFloatToInt(a.y);
  return result;
}

inline Vec2f Vec2iToVec2f(Vec2i a) {
  Vec2f result;
  result.x = (float)a.x;
  result.y = (float)a.y;
  return result;
}

inline Vec2ui Vec2iToVec2ui(Vec2i a) {
  Vec2ui result;
  result.x = a.x;
  result.y = a.y;
  return result;
}

inline Vec2i Vec2uiToVec2i(Vec2ui a) {
  Vec2i result;
  result.x = a.x;
  result.y = a.y;
  return result;
}

inline Vec2f Vec2uiToVec2f(Vec2ui a) {
  Vec2f result;
  result.x = (float)a.x;
  result.y = (float)a.y;
  return result;
}

inline int ClosestLargerDivisible(int a, int x, int y) {
  assert(x != 0);
  assert(y != 0);

  int lcm_x_y = std::lcm(x, y);
  int quotient = (a + lcm_x_y - 1) / lcm_x_y;
  return quotient * lcm_x_y;
}

struct Vec4f {
  float w;
  float x;
  float y;
  float z;
  float& operator[](uint i) { return (&x)[i]; }
};

#endif  // SCALABLE_VIDEO_CODEC_MATH_HPP