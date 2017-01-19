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

template<typename Pixel, typename Mask>
masked_image_view<Pixel, Mask>::masked_image_view() { }


template<typename Pixel, typename Mask>
masked_image_view<Pixel, Mask>::masked_image_view(const masked_image_view& im) :
	base(im),
	mask_mat_(im.mask_mat_) { }


template<typename Pixel, typename Mask>
masked_image_view<Pixel, Mask>::masked_image_view(const ndarray_view_type& vw, const mask_ndarray_view_type& mask_vw) :
	base(vw),
	mask_mat_(to_opencv(mask_vw))
{
	Assert(vw.shape() == mask_vw.shape(), "image and mask must have same shape");
}


template<typename Pixel, typename Mask>
masked_image_view<Pixel, Mask>::masked_image_view(cv_mat_qualified_type& mat, cv_mask_mat_qualified_type& mask_mat) :
	base(mat),
	mask_mat_(mask_mat)
{
	Assert(mat.size == mask_mat.size, "image and mask must have same shape");
}


template<typename Pixel, typename Mask>
void masked_image_view<Pixel, Mask>::reset(const masked_image_view& im) {
	base::reset(im);
	mask_mat_ = im.mask_mat_;
}


template<typename Pixel, typename Mask>
auto masked_image_view<Pixel, Mask>::mask_array_view() const -> mask_ndarray_view_type {
	if(base::is_null()) return mask_ndarray_view_type::null();
	else return to_ndarray_view(mask_mat_);
}

}
