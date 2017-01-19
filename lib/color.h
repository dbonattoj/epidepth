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

#ifndef MF_COLOR_H_
#define MF_COLOR_H_

#include <cstdint>
#include "common.h"

namespace mf {


/// RGB color, 8 bit.
struct rgb_color {
	rgb_color() = default;
	rgb_color(std::uint8_t nr, std::uint8_t ng, std::uint8_t nb) :
		r(nr), g(ng), b(nb) { }
	
	rgb_color(const rgb_color&) = default;
	rgb_color& operator=(const rgb_color&) = default;
	
	
	std::uint8_t r; // red
	std::uint8_t g; // green
	std::uint8_t b; // blue
	
	const static rgb_color black;
	const static rgb_color white;
};

bool operator==(const rgb_color& a, const rgb_color& b);
bool operator!=(const rgb_color& a, const rgb_color& b);


/// YCbCr color, 8 bit.
struct alignas(4) ycbcr_color {
	ycbcr_color() = default;
	ycbcr_color(std::uint8_t ny, std::uint8_t ncr, std::uint8_t ncb) :
		y(ny), cr(ncr), cb(ncb) { }
	
	ycbcr_color(const ycbcr_color&) = default;
	ycbcr_color& operator=(const ycbcr_color&) = default;

	std::uint8_t y;  // luma, Y'
	std::uint8_t cr; // chroma-red, U
	std::uint8_t cb; // chroma-blue, V
};

bool operator==(const ycbcr_color& a, const ycbcr_color& b);
bool operator!=(const ycbcr_color& a, const ycbcr_color& b);



/// Color conversion, specialized for different color formats.
template<typename Output, typename Input>
Output color_convert(const Input&);

template<> rgb_color color_convert(const ycbcr_color&);
template<> ycbcr_color color_convert(const rgb_color&);


/// Color blend.
rgb_color color_blend(const rgb_color& a, const rgb_color& b);
rgb_color color_blend(const rgb_color& a, real a_weight, const rgb_color& b, real b_weight);

}


#include "nd.h"

namespace tff {

/// Color `elem_traits` specializations.
template<>
struct elem_traits<mf::rgb_color> :
	elem_traits_base<mf::rgb_color, std::uint8_t, 3, false> { };
	
template<>
struct elem_traits<mf::ycbcr_color> :
	elem_traits_base<mf::ycbcr_color, std::uint8_t, 3, false> { };

}

#endif
