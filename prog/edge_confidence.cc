#include "edge_confidence.h"
#include "utility.h"
#include "global.h"

#include "utility/misc.h"
#include "opencv.h"


namespace mf {

ndarray<2, real> edge_confidence(const ndarray_view<2, rgb_color>& epi, std::ptrdiff_t radius) {
	ndarray<2, real> conf(epi_shp);
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray_view<1, rgb_color> epi_line = epi.slice(s, 1);
		for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
			std::ptrdiff_t u_begin = (u > radius ? u - radius : u);
			std::ptrdiff_t u_end = (u < epi.shape()[0]-radius ? u + radius : epi.shape()[0]);
			real diff_sum = 0.0;
			for(std::ptrdiff_t u2 = u_begin; u2 < u_end; ++u2)
				diff_sum += color_diff(epi_line[u], epi_line[u2]);
			conf[u][s] = sq(diff_sum);
		}
	}
	return conf;
}


ndarray<2, std::uint8_t> edge_confidence_mask(const ndarray_view<2, real>& conf, real threshold) {
	ndarray<2, std::uint8_t> mask(conf.shape());

	cv::Mat_<real> conf_cv = to_opencv(conf);
	cv::Mat_<uchar> mask_cv = to_opencv(mask.view());
	
	cv::Mat_<uchar> kernel_cv = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
	cv::morphologyEx((conf_cv > threshold), mask_cv, cv::MORPH_ERODE, kernel_cv);
	
	return mask;
}


}