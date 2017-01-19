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

#ifndef MF_MASKED_IMAGE_VIEW_H_
#define MF_MASKED_IMAGE_VIEW_H_

#include "image_view.h"

namespace mf {

/// View to two-dimensional image with mask.
/** Image and mask represented as two separate OpenCV Mat.
 ** \a Pixel may be const or non-const. \a Mask must be non-const, but the actually used `mask_type` will get the
 ** same constness as `pixel_type`. */
template<typename Pixel, typename Mask = byte>
class masked_image_view : public image_view<Pixel> {
	static_assert(! std::is_const<Mask>::value, "masked_image_view Mask must not be const");
	using base = image_view<Pixel>;

public:
	using typename base::shape_type;
	using typename base::pixel_type;
	using typename base::ndarray_view_type;
	using typename base::cv_mat_type;
	using typename base::cv_mat_qualified_type;
	
	using mask_type = std::conditional_t<std::is_const<Pixel>::value, const Mask, Mask>;
	using mask_ndarray_view_type = ndarray_view<2, mask_type>;

	using cv_mask_mat_type = cv::Mat_<std::remove_const_t<mask_type>>;
	using cv_mask_mat_qualified_type = std::conditional_t<
		std::is_const<mask_type>::value, const cv_mask_mat_type, cv_mask_mat_type
	>;

private:
	using const_view = typename base::const_view;
	using const_masked_view = masked_image_view<std::add_const_t<pixel_type>, Mask>;

	mutable cv_mask_mat_type mask_mat_;

public:
	masked_image_view();
	masked_image_view(const masked_image_view&);
	masked_image_view(const ndarray_view_type&, const mask_ndarray_view_type&);
	masked_image_view(cv_mat_qualified_type&, cv_mask_mat_qualified_type&);

	masked_image_view& operator=(const masked_image_view&) = delete;

	void reset(const masked_image_view&);
	operator const_masked_view () const { return const_masked_view(base::mat_, mask_mat_); }
	operator const_view () const { return const_view(base::mat_); }

	mask_ndarray_view_type mask_array_view() const;
	cv_mask_mat_qualified_type& cv_mask_mat() const { return mask_mat_; }
};


template<typename Pixel, typename Mask>
masked_image_view<Pixel, Mask> make_masked_image_view(const ndarray_view<2, Pixel>& vw, const ndarray_view<2, Mask>& mask_vw) {
	return masked_image_view<Pixel, Mask>(vw, mask_vw);
}

}

#include "masked_image_view.tcc"

#endif
