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

const rgba_color rgba_color::black(0, 0, 0);
const rgba_color rgba_color::white(255, 255, 255);

rgba_color color_blend(const rgba_color& a, const rgba_color& b, real k) {
	real ik = 1.0 - k;
	rgba_color ab;
	ab.r = ik*a.r + k*b.r;
	ab.g = ik*a.g + k*b.g;
	ab.b = ik*a.b + k*b.b;
	ab.a = 1.0;
	return ab;
}

}
