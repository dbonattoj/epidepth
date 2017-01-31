#include "disparity_estimator_native.h"
#include "global.h"
#include "utility.h"

#include "utility/misc.h"
#include "opencl.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <streambuf>
#include "io/image_export.h"

namespace mf {

namespace {
	std::ofstream confidences_log_;
	std::ofstream scores_log_;
	std::mutex mut_;

	void log_confidence_(real conf) {
		std::lock_guard<std::mutex> lock(mut_);
		if(! confidences_log_.is_open()) confidences_log_.open("../output/conf.dat");
		confidences_log_ << conf << '\n';
	}

	void log_score_(real sc) {
		std::lock_guard<std::mutex> lock(mut_);
		if(! scores_log_.is_open()) scores_log_.open("../output/score.dat");
		scores_log_ << sc << '\n';
	}
}


real disparity_estimator_native::depth_score_
(const ndarray_view<2, rgba_color>& epi, std::ptrdiff_t s, std::ptrdiff_t u, real d) {
	std::vector<rgba_color> colors;
	colors.reserve(s_sz);
		
	for(std::ptrdiff_t s2 = 0; s2 < epi.shape()[1]; ++s2) {
		real dv = d * scale_down_factor;
		std::ptrdiff_t u2 = u  + (s - s2)*dv;
		if(u2 < 0 || u2 >= u_sz) continue;
		colors.push_back(epi[u2][s2]);
	}

	auto kernel = [](real x) -> real {
		const real h = depth_score_color_max_threshold;
		real q = x / h;
		if(q <= 1.0) return sq(1.0 - q);
		else return 0.0;
	};
	
	real score = 0.0;
	
	rgba_color col = epi[u][s];
	
	for(const rgba_color& col2 : colors) score += kernel(color_diff(col, col2));
	score /= colors.size();
	
	//log_score_(score);
	
	return score;
};


disparity_estimator_native::disparity_estimator_native(
	std::ptrdiff_t v_min,
	std::ptrdiff_t v_max,
	const ndarray_view<3, rgba_color>& epi,
	const ndarray_view<3, real>& edge_conf,
	const ndarray_view<3, real>& min_disp,
	const ndarray_view<3, real>& max_disp
) :
	v_min_(v_min),
	v_max_(v_max),
	epi_(epi),
	edge_conf_(edge_conf),
	min_disp_(min_disp),
	max_disp_(max_disp) { }


epi_line_disparity_result disparity_estimator_native::estimate_epi_line_disparity
(std::ptrdiff_t v, std::ptrdiff_t s, const ndarray_view<1, uchar>& mask) {
	ndarray_view<2, rgba_color> epi = epi_[v - v_min_];
	ndarray_view<1, real> conf = edge_conf_[v - v_min_].slice(s, 1);
	ndarray_view<1, real> min_disparity = min_disp_[v - v_min_].slice(s, 1);
	ndarray_view<1, real> max_disparity = max_disp_[v - v_min_].slice(s, 1);
	
	epi_line_disparity_result result;

	#pragma omp parallel for
	for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
		if(! mask[u]) {
			result.confidence[u] = 0.0;
			result.disparity[u] = NAN;
			continue;
		}
		
		real min_d = min_disparity[u], max_d = max_disparity[u];
		real d_incr = (max_d - min_d) / disparity_steps;
		Assert_crit(! std::isnan(min_d));
		Assert_crit(! std::isnan(max_d));
		
		real max_score = 0.0;
		real avg_score = 0.0;
		real max_score_d = 0;
		real n_scores = 0;
		for(real d = min_d; d < max_d; d += d_incr) {
			real score = depth_score_(epi, s, u, d);
			avg_score += score;
			n_scores += 1.0;
			if(score > max_score) {
				max_score = score;
				max_score_d = d;
			}
		}
		avg_score /= n_scores;
	/*
		static bool first=true;
		if(first && max_score_d > 3.5 && same(epi, epis[100])) {
			static ndarray<2, rgb_color> epi_copy(epi);
		//	for(std::ptrdiff_t s2 = 0; s2 < epi.shape()[1]; ++s2) {
		//		real dv = max_score_d * scale_down_factor;
		//		std::ptrdiff_t u2 = u  + (s - s2)*dv;
		//			std::cout << make_ndptrdiff(u2,s2) << std::endl;
		//		if(u2 < 0 || u2 >= u_sz) continue;
		//		//epi[u2][s2] = rgb_color(255,0,0);
		//	}
			epi_copy[u][s] = rgb_color(0,0,255);
			//first = false;
			std::cout << max_score_d << ": " << max_score << std::endl;
			
			image_export(make_image_view(epi_copy.view()), "../output/epi.png");
			//scores_log_.close();
			//std::terminate();
		}
	*/
		result.disparity[u] = max_score_d;
		result.confidence[u] = conf[u] * std::abs(max_score - avg_score);
		//log_confidence_(result.confidence[u]);
	}
	
	return result;
}


}