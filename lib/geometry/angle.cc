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

#include "angle.h"
#include "math_constants.h"
#include <istream>
#include <ostream>

namespace mf {

std::ostream& operator<<(std::ostream& str, angle a) {
	str << a.get_degrees() << "°";
	return str;
}

std::istream& operator>>(std::istream& str, angle& a) {
	real deg;
	str >> deg;
	a.set_degrees(deg);
	return str;
}

angle angle::smallest_between(angle a, angle b) {
	real diff = b - a;
	while(diff < -pi) diff += pi;
	while(diff > +pi) diff -= pi;
	return diff;
}

}
