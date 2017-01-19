/*
Author : Tim Lenertz
Date : May 2016

Copyright (c) 2016, Université libre de Bruxelles

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files to deal in the Software without restriction, including the rights to use, copy, modify, merge,
publish the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MF_UTILITY_MISC_H_
#define MF_UTILITY_MISC_H_

#include <string>
#include <vector>
#include <random>
#include <memory>
#include <type_traits>

namespace mf {
	
using default_random_engine = std::mt19937;
	
template<typename...> using void_t = void;

/// Check if a file at `filename` exists.
bool file_exists(const std::string& filename);

/// Compute square of a number.
template<typename Numeric> Numeric sq(Numeric n) { return n * n; }

/// Clamp `value` between `minimum` and `maximum` value.
template<typename T> T clamp(T value, T minimum, T maximum);

/// Compute greatest common divisor of `a` and `b`.
template<typename T> T gcd(T a, T b);

/// Compute least common multiple of `a` and `b`.
template<typename T> T lcm(T a, T b);

/// Check if `x` is a power of 2.
template<typename T>
bool is_power_of_two(T x) {
	return (x != 0) && !(x & (x - 1));
}

/// Check if `x` is a multiple of `base`, including zero.
template<typename T, typename T2>
bool is_multiple_of(T x, T2 base) {
	return (x % base == 0);
}

/// Check if `x` is a non-zero multiple of `base`.
template<typename T, typename T2>
bool is_nonzero_multiple_of(T x, T2 base) {
	return (x != 0) && is_multiple_of(x, base);
}

/// Check if `x` is odd.
template<typename T>
bool is_odd(T x) { return (x % 2) != 0; }

/// Check if `x` is even.
template<typename T>
bool is_even(T x) { return (x % 2) == 0; }


default_random_engine& random_engine();


template<typename T>
T randint(T a, T b) {
	default_random_engine& engine = random_engine();
	std::uniform_int_distribution<T> dist(a, b);
	return dist(engine);
}



/// Get `shared_ptr` to new object copy- or move- constructed from \a t.
/** Returns `std::shared_ptr<T>` (with `T` decayed). */
template<typename T>
auto forward_make_shared(T&& t) {
	using decayed_t = std::decay_t<T>;
	return std::make_shared<decayed_t>(std::forward<T>(t));
}


/// Get `shared_ptr` to new const object copy- or move- constructed from \a t.
/** Returns `std::shared_ptr<const T>` (with `T` decayed). */
template<typename T>
auto forward_make_shared_const(T&& t) {
	using decayed_t = std::decay_t<T>;
	return std::make_shared<const decayed_t>(std::forward<T>(t));
}


void sleep(unsigned int seconds);
void sleep_ms(unsigned int milliseconds);
void sleep_us(unsigned int microseconds);


}

#include "misc.tcc"

#endif
