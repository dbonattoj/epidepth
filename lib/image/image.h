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

#ifndef MF_IMAGE_H_
#define MF_IMAGE_H_

#include "image_view.h"
#include <type_traits>

namespace mf {

using namespace tff;

template<typename Pixel>
class image {
	static_assert(! std::is_const<Pixel>::value, "image Pixel must be non-const");
	
public:
	using pixel_type = Pixel;
	using shape_type = ndsize<2>;
	using view_type = image_view<pixel_type>;
	using const_view_type = image_view<const pixel_type>;
	using ndarray_view_type = ndarray_view<2, pixel_type>;
	using const_ndarray_view_type = ndarray_view<2, const pixel_type>;
	using cv_mat_type = cv::Mat_<pixel_type>;

private:
	cv_mat_type mat_;

public:
	explicit image(const shape_type&);
	explicit image(const cv_mat_type&);
	explicit image(const const_ndarray_view_type&);

	image(const const_view_type&);
	image(const image&);
	image(image&&);
	
	image& operator=(const const_view_type&);
	image& operator=(const image&);
	image& operator=(image&&);
	
	shape_type shape() const;
	
	view_type view();
	const_view_type view() const { return cview(); }
	const_view_type cview() const { return const_view_type(mat_); }
	
	operator view_type () { return view();  }
	operator const_view_type () const { return cview(); }
	
	ndarray_view_type array_view() { return view().array_view(); }
	const_ndarray_view_type array_view() const { return cview().array_view(); }
	
	cv_mat_type& cv_mat() { return mat_; }
	const cv_mat_type& cv_mat() const { return mat_; }
};


}

#include "image.tcc"

#endif
