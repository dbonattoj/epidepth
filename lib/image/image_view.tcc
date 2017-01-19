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

#include <utility>

namespace mf {

template<typename Pixel>
image_view<Pixel>::image_view() { }


template<typename Pixel>
image_view<Pixel>::image_view(const ndarray_view_type& vw) :
	mat_(to_opencv(vw)) { }


template<typename Pixel>
image_view<Pixel>::image_view(cv_mat_qualified_type& mat) :
	mat_(mat) { }


template<typename Pixel>
image_view<Pixel>::image_view(const image_view& im) :
	mat_(im.cv_mat()) { }

template<typename Pixel>
auto image_view<Pixel>::shape() const -> shape_type {
	return make_ndsize(mat_.size[0], mat_.size[1]);
}


template<typename Pixel>
auto image_view<Pixel>::array_view() const -> ndarray_view_type {
	if(is_null()) return ndarray_view_type::null();
	else return to_ndarray_view(mat_);
}


template<typename Pixel>
void image_view<Pixel>::reset(const image_view& im) {
	mat_ = im.mat_;
}


}
