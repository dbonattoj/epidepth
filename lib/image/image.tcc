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

namespace mf {

template<typename Pixel>
image<Pixel>::image(const shape_type& shape) :
	mat_(shape[0], shape[1]) { }


template<typename Pixel>
image<Pixel>::image(const cv_mat_type& mat) :
	mat_()
{
	mat.copyTo(mat_);
}


template<typename Pixel>
image<Pixel>::image(const const_ndarray_view_type& ndvw) :
	mat_()
{
	copy_to_opencv(ndvw, mat_);
}


template<typename Pixel>
image<Pixel>::image(const const_view_type& vw) :
	mat_()
{
	vw.cv_mat().copyTo(mat_);
}


template<typename Pixel>
image<Pixel>::image(image&& im) :
	mat_(im.mat_) { }
	

template<typename Pixel>
image<Pixel>::image(const image& im) :
	image(im.cview()) { }


template<typename Pixel>
auto image<Pixel>::operator=(const const_view_type& vw) -> image& {
	vw.cv_mat().copyTo(mat_);
	return *this;
}


template<typename Pixel>
auto image<Pixel>::operator=(image&& im) -> image& {
	mat_ = im.mat_;
	return *this;
}


template<typename Pixel>
auto image<Pixel>::operator=(const image& im) -> image& {
	im.mat_.copyTo(mat_);
	return *this;
}


template<typename Pixel>
auto image<Pixel>::shape() const -> shape_type {
	return make_ndsize(mat_.size[0], mat_.size[1]);
}


template<typename Pixel>
auto image<Pixel>::view() -> view_type {
	view_type vw(mat_);
	return vw;
}


}
