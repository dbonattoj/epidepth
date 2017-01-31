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


struct rgba_color {
	rgba_color() = default;
	rgba_color(std::uint8_t nr, std::uint8_t ng, std::uint8_t nb, std::uint8_t na = 255) :
		r(nr), g(ng), b(nb), a(na) { }
	
	rgba_color(const rgba_color&) = default;
	rgba_color& operator=(const rgba_color&) = default;
	
	
	std::uint8_t r; // red
	std::uint8_t g; // green
	std::uint8_t b; // blue
	std::uint8_t a; // alpha
	
	const static rgba_color black;
	const static rgba_color white;
};


inline bool operator==(const rgba_color& a, const rgba_color& b) {
	return (a.r == b.r) && (a.g == b.g) && (a.b == b.b) && (a.a == b.a);
}

inline bool operator!=(const rgba_color& a, const rgba_color& b) {
	return (a.r != b.r) || (a.g != b.g) || (a.b != b.b) || (a.a != b.a);
}

rgba_color color_blend(const rgba_color& a, const rgba_color& b, real k);

}


#endif
