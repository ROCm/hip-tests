/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <catch.hpp>

template <typename T, typename Matcher> class ValidatorBase : public Catch::MatcherBase<T> {
 public:
  template <typename... Ts>
  ValidatorBase(T target, Ts&&... args) : matcher_{std::forward<Ts>(args)...}, target_{target} {}

  bool match(const T& val) const override {
    if (std::isnan(target_)) {
      return std::isnan(val);
    }

    return matcher_.match(val);
  }

  virtual std::string describe() const override {
    if (std::isnan(target_)) {
      return "is not NaN";
    }

    return matcher_.describe();
  }

 private:
  Matcher matcher_;
  T target_;
  bool nan = false;
};

template <typename T> auto ULPValidatorBuilderFactory(int64_t ulps) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinUlpsMatcher>{
        target, Catch::WithinULP(target, ulps)};
  };
};

template <typename T> auto AbsValidatorBuilderFactory(double margin) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinAbsMatcher>{
        target, Catch::WithinAbs(target, margin)};
  };
}

template <typename T> auto RelValidatorBuilderFactory(T margin) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinRelMatcher>{
        target, Catch::WithinRel(target, margin)};
  };
}

template <typename T, typename ValidatorBuilder>
class PairValidator : public Catch::MatcherBase<std::pair<T, T>> {
 public:
  PairValidator(const std::pair<T, T>& target, const ValidatorBuilder& vb)
      : sin_matcher_{vb(target.first)}, cos_matcher_{vb(target.second)} {}

  bool match(const std::pair<T, T>& val) const override {
    return sin_matcher_.match(val.first) && cos_matcher_.match(val.second);
  }

  virtual std::string describe() const override {
    return "<" + sin_matcher_.describe() + ", " + cos_matcher_.describe() + ">";
  }

 private:
  decltype(std::declval<ValidatorBuilder>()(std::declval<T>())) sin_matcher_;
  decltype(std::declval<ValidatorBuilder>()(std::declval<T>())) cos_matcher_;
};

template <typename T, typename ValidatorBuilder>
auto PairValidatorBuilderFactory(const ValidatorBuilder& vb) {
  return [&](const std::pair<T, T>& t) { return PairValidator<T, ValidatorBuilder>(t, vb); };
}
