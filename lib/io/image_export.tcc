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

#include "../opencv.h"

namespace mf {

namespace detail {

inline void imwrite_cv(const std::string& filename, const cv::Mat& mat) {
	bool success = false;
	try {
		success = cv::imwrite(filename, mat);
	} catch(...) { }
	if(! success) throw std::runtime_error("imwrite failed");
}

}
	
inline void image_export(const image_view<rgba_color>& img, const std::string& filename) {
	cv::Mat output_img;
	cv::cvtColor(img.cv_mat(), output_img, CV_RGBA2BGR);
	detail::imwrite_cv(filename, output_img);
}


template<typename Mask>
void image_export(const masked_image_view<rgba_color, Mask>& img, const std::string& filename) {
	cv::Mat output_img, holes;
	cv::cvtColor(img.cv_mat(), output_img, CV_RGBA2BGRA);
	cv::bitwise_not(img.cv_mask_mat(), holes);
	output_img.setTo(cv::Scalar::all(0), holes);
	detail::imwrite_cv(filename, output_img);
}


template<typename Mask>
void image_export(const masked_image_view<rgba_color, Mask>& img, const std::string& filename, const rgba_color& background) {
	cv::Scalar cv_background(background.b, background.g, background.r, background.a);
	cv::Mat output_img, holes;
	cv::cvtColor(img.cv_mat(), output_img, CV_RGBA2BGRA);
	cv::bitwise_not(img.cv_mask_mat(), holes);
	output_img.setTo(cv_background, holes);
	detail::imwrite_cv(filename, output_img);
}


template<typename Scalar, typename Mask>
std::enable_if_t<std::is_arithmetic<Scalar>::value> image_export
	(const masked_image_view<Scalar, Mask>& img, const std::string& filename, Scalar min_value, Scalar max_value, const rgba_color& background)
{
	cv::Mat output_img;
	double alpha = 255.0 / (max_value - min_value);
	double beta = alpha * min_value;
	img.cv_mat().convertTo(output_img, CV_8U, alpha, beta);
	
	cv::Scalar cv_background(background.b, background.g, background.r, background.a);
	cv::Mat output_rgb_img, holes;
	cv::cvtColor(output_img, output_rgb_img, CV_GRAY2RGBA);
	cv::bitwise_not(img.cv_mask_mat(), holes);
	output_rgb_img.setTo(cv_background, holes);
	detail::imwrite_cv(filename, output_rgb_img);

}


template<typename Scalar, typename Mask>
std::enable_if_t<std::is_arithmetic<Scalar>::value> image_export
	(const masked_image_view<Scalar, Mask>& img, const std::string& filename, const rgba_color& background)
{
	cv::Point min_pos, max_pos;
	cv::minMaxLoc(img.cv_mat(), NULL, NULL, &min_pos, &max_pos, img.cv_mask_mat());
	image_export(img, filename, img.cv_mat()(min_pos), img.cv_mat()(max_pos), background);
}


template<typename Scalar, typename Mask>
std::enable_if_t<std::is_arithmetic<Scalar>::value> image_export
	(const masked_image_view<Scalar, Mask>& img, const std::string& filename, Scalar min_value, Scalar max_value)
{
	cv::Mat output_img;
	double alpha = 255.0 / (max_value - min_value);
	double beta = alpha * min_value;
	img.cv_mat().convertTo(output_img, CV_8U, alpha, beta);
	
	cv::Mat output_rgba_img, holes;
	cv::cvtColor(output_img, output_rgba_img, CV_GRAY2RGBA);
	cv::bitwise_not(img.cv_mask_mat(), holes);
	output_rgba_img.setTo(cv::Scalar::all(0), holes);
	
	detail::imwrite_cv(filename, output_rgba_img);
}


template<typename Scalar, typename Mask>
std::enable_if_t<std::is_arithmetic<Scalar>::value> image_export
	(const masked_image_view<Scalar, Mask>& img, const std::string& filename)
{
	cv::Point min_pos, max_pos;
	cv::minMaxLoc(img.cv_mat(), NULL, NULL, &min_pos, &max_pos, img.cv_mask_mat());
	image_export(img, filename, img.cv_mat()(min_pos), img.cv_mat()(max_pos));
}


template<typename Scalar>
std::enable_if_t<std::is_arithmetic<Scalar>::value> image_export
	(const image_view<Scalar>& img, const std::string& filename, Scalar min_value, Scalar max_value)
{
	cv::Mat output_img;
	double alpha = 255.0 / (max_value - min_value);
	double beta = alpha * min_value;
	img.cv_mat().convertTo(output_img, CV_8U, alpha, beta);
	
	detail::imwrite_cv(filename, output_img);
}


template<typename Scalar>
std::enable_if_t<std::is_arithmetic<Scalar>::value> image_export
	(const image_view<Scalar>& img, const std::string& filename)
{
	cv::Point min_pos, max_pos;
	cv::minMaxLoc(img.cv_mat(), NULL, NULL, &min_pos, &max_pos);
	image_export(img, filename, img.cv_mat()(min_pos), img.cv_mat()(max_pos));
}



}
