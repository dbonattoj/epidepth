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

#include <sstream>
#include <algorithm>
#include <cmath>
#include "../common.h"


namespace mf {


template<typename T>
inline T clamp(T value, T minimum, T maximum) {
	return std::min(std::max(value, minimum), maximum);
}


template<typename T>
T gcd(T a, T b) {
	Assert_crit(a > 0 && b > 0);
	if(a < b) std::swap(a, b);
	while(b > 0) {
		T c = a % b;
		a = b;
		b = c;
	}
	return a;
}

template<typename T>
T lcm(T a, T b) {
	if(a == 0 || b == 0) return 0;
	else return (a * b) / gcd(a, b);
}


}
