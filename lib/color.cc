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

#include "color.h"
#include "utility/misc.h"
#include <cstdint>

namespace mf {

const rgb_color rgb_color::black(0, 0, 0);
const rgb_color rgb_color::white(255, 255, 255);


template<>
rgb_color color_convert(const ycbcr_color& in) {
	real y  = static_cast<real>(in.y) + 0.5;

	real cr = static_cast<real>(in.cr) - 127.0;
	real cb = static_cast<real>(in.cb) - 127.0;

	real r = clamp(y            + 1.402*cr, 0.0, 255.0);
	real g = clamp(y - 0.344*cb - 0.714*cr, 0.0, 255.0);
	real b = clamp(y + 1.772*cb,            0.0, 255.0);
		
	return rgb_color(static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b));
}


template<>
ycbcr_color color_convert(const rgb_color& in) {
	real r = static_cast<real>(in.r);
	real g = static_cast<real>(in.g);
	real b = static_cast<real>(in.b);

	real y  = clamp( 0.29900 * r + 0.58700 * g + 0.11400 * b        , 0.0, 255.0);
	real cb = clamp(-0.16874 * r - 0.33126 * g + 0.50000 * b + 128.0, 0.0, 255.0);
	real cr = clamp( 0.50000 * r - 0.41869 * g - 0.08131 * b + 128.0, 0.0, 255.0);

	return ycbcr_color(static_cast<std::uint8_t>(y), static_cast<std::uint8_t>(cb), static_cast<std::uint8_t>(cr));
}


rgb_color color_blend(const rgb_color& a, const rgb_color& b) {
	// TODO generalize to similar operations, use SIMD
	using sum_type = std::int_fast16_t;
	sum_type r_sum = static_cast<sum_type>(a.r) + static_cast<sum_type>(b.r);
	sum_type g_sum = static_cast<sum_type>(a.g) + static_cast<sum_type>(b.g);
	sum_type b_sum = static_cast<sum_type>(a.b) + static_cast<sum_type>(b.b);
	rgb_color result;
	result.r = static_cast<std::uint8_t>(r_sum / 2);
	result.g = static_cast<std::uint8_t>(g_sum / 2);
	result.b = static_cast<std::uint8_t>(b_sum / 2);
	return result;
}


rgb_color color_blend(const rgb_color& a, real a_weight, const rgb_color& b, real b_weight) {
	using sum_type = real;
	real r_sum = a_weight * static_cast<sum_type>(a.r) + b_weight * static_cast<sum_type>(b.r);
	real g_sum = a_weight * static_cast<sum_type>(a.g) + b_weight * static_cast<sum_type>(b.g);
	real b_sum = a_weight * static_cast<sum_type>(a.b) + b_weight * static_cast<sum_type>(b.b);
	real weight_sum = a_weight + b_weight;
	rgb_color result;
	result.r = static_cast<std::uint8_t>(clamp(r_sum / weight_sum, 0.0, 255.0));
	result.g = static_cast<std::uint8_t>(clamp(g_sum / weight_sum, 0.0, 255.0));
	result.b = static_cast<std::uint8_t>(clamp(b_sum / weight_sum, 0.0, 255.0));
	return result;
}


bool operator==(const rgb_color& a, const rgb_color& b) {
	return (a.r == b.r) && (a.g == b.g) && (a.b == b.b);
}

bool operator!=(const rgb_color& a, const rgb_color& b) {
	return (a.r != b.r) || (a.g != b.g) || (a.b != b.b);
}

bool operator==(const ycbcr_color& a, const ycbcr_color& b) {
	return (a.y == b.y) && (a.cr == b.cr) && (a.cb == b.cb);
}

bool operator!=(const ycbcr_color& a, const ycbcr_color& b) {
	return (a.y != b.y) || (a.cr != b.cr) || (a.cb != b.cb);
}


}
