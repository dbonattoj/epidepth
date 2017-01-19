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

#include "image_camera.h"

namespace mf {

image_camera::image_camera(const ndsize<2>& sz) :
	image_size_(sz) { }


void image_camera::set_image_size_(ndsize<2> sz) {
	image_size_ = sz;
}


ndsize<2> image_camera::scaled_image_size(ndsize<2> shape, real factor) {
	return make_ndsize(
		factor * shape[0],
		factor * shape[1]
	);
}


real image_camera::image_aspect_ratio() const {
	return static_cast<real>(image_size_[0]) / static_cast<real>(image_size_[1]);
}

void image_camera::flip_pixel_coordinates() {
	image_size_ = flip(image_size_);
	flipped_ = ! flipped_;
}



}
