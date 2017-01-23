#include "disparity_estimation.h"
#include "global.h"
#include "utility.h"

#include "utility/misc.h"
#include "opencl.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <streambuf>

namespace mf {

namespace {
	std::ofstream confidences_log_;
	std::mutex mut_;

	void log_confidence_(real conf) {
		std::lock_guard<std::mutex> lock(mut_);
		if(! confidences_log_.is_open()) confidences_log_.open("../output/conf.dat");
		confidences_log_ << conf << '\n';
	}
}

/*
namespace {
	cl::Context cl_context;

	void setup_cl_context_() {
		cl_context = get_cl_context();
		cl::Program::Sources sources;

		std::ifstream file("../prog/disparity_score.cl");
		std::string file_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		file.close();
		sources.push_back({ file_str.c_str(), file_str.length() });
	
		std::string options = "-DS_SZ=" + std::to_string(s_sz);
		cl::Program program(sources);
		auto res = program.build(options);
		assert(res == CL_SUCCESS);
	}
}
*/

real depth_score(const ndarray_view<2, rgb_color>& epi, std::ptrdiff_t s, std::ptrdiff_t u, real d) {
	std::vector<rgb_color> colors;
	colors.reserve(s_sz);
		
	for(std::ptrdiff_t s2 = 0; s2 < epi.shape()[1]; ++s2) {
		std::ptrdiff_t u2 = u  + (s - s2)*d/scale_down_factor;
		if(u2 < 0 || u2 >= u_sz) continue;
		colors.push_back(epi[u2][s2]);
	}

	auto kernel = [](real x) -> real {
		const real h = depth_score_color_threshold;
		real q = x / h;
		if(q <= 1.0) return sq(1.0 - q);
		else return 0.0;
	};
	
	real score = 0.0;
	
	rgb_color col = epi[u][s];
	
	
	for(const rgb_color& col2 : colors) score += kernel(color_diff(col, col2));
	score /= colors.size();
	
	return score;
};


epi_line_disparity_result estimate_epi_line_disparity(
	std::ptrdiff_t s,
	const ndarray_view<2, rgb_color>& epi,
	const ndarray_view<1, real>& conf,
	const ndarray_view<1, bool>& mask,
	const ndarray_view<1, real>& min_disparity,
	const ndarray_view<1, real>& max_disparity
) {
	epi_line_disparity_result result(u_sz, s);
	
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
			real score = depth_score(epi, s, u, d);
			avg_score += score;
			n_scores += 1.0;
			if(score > max_score) {
				max_score = score;
				max_score_d = d;
			}
		}
		avg_score /= n_scores;
		
		result.disparity[u] = max_score_d;
		result.confidence[u] = conf[u] * std::abs(max_score - avg_score);
		//log_confidence_(result.confidence[u]);
	}
	
	return result;
}


}