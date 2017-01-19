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

#ifndef MF_OPENCV_H_
#define MF_OPENCV_H_

#include <opencv2/opencv.hpp>
#include <algorithm>

#include "color.h"
#include "nd/ndcoord.h"
#include "nd/ndarray.h"

namespace cv { // in OpenCV namespace
	template<>
	class DataType<::mf::rgb_color> {
	public:
		using value_type = mf::rgb_color;
		using work_type = int;
		using channel_type = uchar;
		enum {
			generic_type = 0,
			depth = DataDepth<channel_type>::value,
			channels = 3,
			fmt = ((channels - 1)<<8) + DataDepth<channel_type>::fmt,
			type = CV_MAKETYPE(depth, channels)
		};
		using vec_type = Vec<channel_type, channels>;
	};
	
	template<>
	class DataType<::mf::ycbcr_color> {
	public:
		using value_type = mf::ycbcr_color;
		using work_type = int;
		using channel_type = uchar;
		enum {
			generic_type = 0,
			depth = DataDepth<channel_type>::fmt,
			channels = 3,
			fmt = ((channels - 1)<<8) + DataDepth<channel_type>::fmt,
			type = CV_MAKETYPE(depth, channels)
		};
		using vec_type = Vec<channel_type, channels>;
	};
}


namespace mf {


/// Copy the data in \a vw into the OpenCV Mat \a mat.
/** \a mat is created and allocated as needed using `cv::Mat::create()`, and \a mat owns the new copy of the data.
 ** `Elem` can not be a `masked_elem` type. */
template<std::size_t Dim, typename Elem>
void copy_to_opencv(const ndarray_view<Dim, const Elem>& vw, cv::Mat_<Elem>& mat) {
	int sizes[Dim];
	for(std::ptrdiff_t i = 0; i < Dim; ++i) sizes[i] = vw.shape()[i];

	mat.create(Dim, sizes);

	/// TODO check order of iterations
	std::transform(
		vw.begin(),
		vw.end(),
		mat.begin(),
		[](const Elem& elem) -> Elem { return elem; }
	);
}



/// Create `cv::Mat` header pointing at the same data as \a vw.
/** If `Elem` is a const type, a `const_cast` is done and the returned type is
 ** `const cv::Mat_<std::remove_const_t<Elem>>`. The called must ensure that the returned `cv::Mat_` does not get
 ** written into.
 ** \a vw must have default strides without padding, and `Elem` cannot be a `masked_elem` type. */
template<std::size_t Dim, typename Elem>
auto to_opencv(const ndarray_view<Dim, Elem>& vw) {	
	Assert(vw.has_default_strides_without_padding());

	using elem_type = std::remove_const_t<Elem>;
	using opencv_type = cv::DataType<elem_type>;
	using mat_type = cv::Mat_<elem_type>;
	using qualified_mat_type = std::conditional_t<std::is_const<Elem>::value, const mat_type, mat_type>;
	using channel_type = typename opencv_type::channel_type;

	int sizes[Dim];
	for(std::ptrdiff_t i = 0; i < Dim; ++i) sizes[i] = vw.shape()[i];

	cv::Mat mat(
		Dim,
		sizes,
		opencv_type::type,
		reinterpret_cast<void*>(const_cast<elem_type*>(vw.start()))
	);
	cv::Mat_<elem_type> mat_(mat);

	return mat_;
}


/// Create \ref ndarray_view pointing at same data as OpenCV matrix \a mat.
/** Dimension must be specified if different from 2, because the OpenCV \a mat knows its dimension only runtime. */
template<std::size_t Dim = 2, typename Elem>
ndarray_view<Dim, Elem> to_ndarray_view(cv::Mat_<Elem>& mat) {
	if(mat.dims != Dim) throw std::invalid_argument("OpenCV matrix in to_ndarray_view has incorrect dimension");
	Elem* start = reinterpret_cast<Elem*>(mat.data);
	ndcoord<Dim, int> shape(mat.size.p, mat.size.p + Dim);
	ndcoord<Dim, std::size_t> strides(mat.step.p, mat.step.p + Dim);
	return ndarray_view<Dim, Elem>(start, shape, strides);
}


/// Create \ref ndarray_view pointing at same data as OpenCV matrix \a mat.
/** Dimension must be specified if different from 2, because the OpenCV \a mat knows its dimension only runtime. */
template<std::size_t Dim = 2, typename Elem>
ndarray_view<Dim, const Elem> to_ndarray_view(const cv::Mat_<Elem>& mat) {
	if(mat.dims != Dim) throw std::invalid_argument("OpenCV matrix in to_ndarray_view has incorrect dimension");
	const Elem* start = reinterpret_cast<const Elem*>(mat.data);
	ndcoord<Dim, int> shape(mat.size.p, mat.size.p + Dim);
	ndcoord<Dim, std::size_t> strides(mat.step.p, mat.step.p + Dim);
	return ndarray_view<Dim, const Elem>(start, shape, strides);
}



/// Copy OpenCV Mat \a mat into the memory pointed to by \a vw.
/** \a vw and the memory if points to must have been created beforehand, and \a vw must have the same shape as \a mat.
 ** `Elem` cannot be a `masked_elem` type. */
template<std::size_t Dim, typename Elem>
void copy_to_ndarray_view(cv::Mat_<Elem>& mat, const ndarray_view<Dim, Elem>& vw) {
	ndarray_view<Dim, Elem> tmp_vw = to_ndarray_view<Dim>(mat);
	if(tmp_vw.shape() != vw.shape()) throw std::invalid_argument("vw in copy_to_ndarray_view has incorrect shape");
	vw.assign(tmp_vw);
}



}

#endif
