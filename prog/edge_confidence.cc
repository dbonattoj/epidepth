#include "edge_confidence.h"
#include "utility.h"
#include "global.h"

#include "utility/misc.h"
#include "opencv.h"

#include <fstream>
#include <mutex>


namespace mf {

namespace {
	std::ofstream log_;
	std::mutex mut_;

	void log_edge_confidence_(real conf) {
		std::lock_guard<std::mutex> lock(mut_);
		if(! log_.is_open()) log_.open("../output/edge_conf.dat");
		log_ << conf << '\n';
	}
}

ndarray<2, real> edge_confidence(const ndarray_view<2, rgb_color>& epi) {
	ndarray<2, real> conf(epi_shp);
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray_view<1, rgb_color> epi_line = epi.slice(s, 1);
		for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
			std::ptrdiff_t u_begin = (u > edge_confidence_rad ? u - edge_confidence_rad : u);
			std::ptrdiff_t u_end = (u < epi.shape()[0]-edge_confidence_rad ? u + edge_confidence_rad : epi.shape()[0]);
			real diff_sum = 0.0;
			for(std::ptrdiff_t u2 = u_begin; u2 < u_end; ++u2)
				diff_sum += color_diff(epi_line[u], epi_line[u2]);
			conf[u][s] = diff_sum / (u_end - u_begin);
			//log_edge_confidence_(conf[u][s]);
		}
	}
	return conf;
}


ndarray<2, std::uint8_t> edge_confidence_mask(const ndarray_view<2, real>& conf) {
	ndarray<2, std::uint8_t> mask(conf.shape());

	cv::Mat_<real> conf_cv = to_opencv(conf);
	cv::Mat_<uchar> mask_cv = to_opencv(mask.view());
	
	cv::Mat_<uchar> kernel_cv = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, edge_confidence_mask_width));
	cv::morphologyEx((conf_cv > edge_confidence_mask_min_threshold), mask_cv, cv::MORPH_DILATE, kernel_cv);
	//mask_cv = (conf_cv > edge_confidence_mask_min_threshold);
	
	return mask;
}


}