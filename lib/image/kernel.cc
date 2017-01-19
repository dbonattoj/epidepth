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

#include "../utility/misc.h"
#include "kernel.h"
#include <algorithm>
#include <iostream>

namespace mf {

bool_image_kernel disk_image_kernel(std::size_t diameter) {
	bool_image_kernel kernel(make_ndsize(diameter, diameter));
	auto center = make_ndptrdiff(diameter / 2, diameter / 2);
	std::size_t max_radius_sq = sq(diameter / 2);
	for(auto it = kernel.begin(); it < kernel.end(); ++it) {
		auto coord = it.coordinates();
		std::size_t radius_sq = sq(coord[0] - center[0]) + sq(coord[1] - center[1]);
		*it = (radius_sq <= max_radius_sq);
	}
	return kernel;
}


bool_image_kernel box_image_kernel(std::size_t diameter) {
	bool_image_kernel kernel(make_ndsize(diameter, diameter));
	std::fill(kernel.begin(), kernel.end(), true);
	return kernel;
}


}
