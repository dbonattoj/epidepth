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

#ifndef MF_IMAGE_VIEW_H_
#define MF_IMAGE_VIEW_H_

#include "../opencv.h"
#include "../nd.h"
#include <type_traits>

namespace mf {

using namespace tff;

/// View to two-dimensional image.
template<typename Pixel>
class image_view {
public:
	using shape_type = ndsize<2>;
	using pixel_type = Pixel;
	
	using ndarray_view_type = ndarray_view<2, pixel_type>;
	
	using cv_mat_type = cv::Mat_<std::remove_const_t<pixel_type>>;
	using cv_mat_qualified_type = std::conditional_t<
		std::is_const<pixel_type>::value, const cv_mat_type, cv_mat_type
	>;

protected:
	using const_view = image_view<std::add_const_t<pixel_type>>;
	mutable cv_mat_type mat_;

public:
	image_view();
	explicit image_view(const ndarray_view_type&);
	explicit image_view(cv_mat_qualified_type&);
	image_view(const image_view&);
	virtual ~image_view() = default;
	
	image_view& operator=(const image_view&) = delete;

	void reset(const image_view&);
	operator const_view () const { return const_view(mat_); }

	bool is_null() const { return mat_.empty(); }
	explicit operator bool () const { return ! is_null(); }
			
	shape_type shape() const;
	
	ndarray_view_type array_view() const;
	cv_mat_qualified_type& cv_mat() const { return mat_; }
};


template<typename Pixel>
image_view<Pixel> make_image_view(const ndarray_view<2, Pixel>& vw) {
	return image_view<Pixel>(vw);
}


}

#include "image_view.tcc"

#endif
