#include "disparity_estimation.h"
#include "global.h"
#include "utility.h"

#include "utility/misc.h"
#include <vector>
#include <iostream>

namespace mf {

real depth_score(const ndarray_view<2, rgb_color>& epi, std::ptrdiff_t s, std::ptrdiff_t u, real d) {
	std::vector<rgb_color> colors;
	colors.reserve(s_sz);
		
	for(std::ptrdiff_t s2 = 0; s2 < epi.shape()[1]; ++s2) {
		std::ptrdiff_t u2 = u  + (s - s2)*d;
		if(u2 < 0 || u2 >= u_sz) continue;
		colors.push_back(epi[u2][s2]);
	}

	auto kernel = [](real x) -> real {
		const real h = 0.02;
		real q = x / h;
		if(std::abs(q) <= 1.0) return sq(1.0 - q);
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
	const ndarray_view<1, std::uint8_t>& mask,
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
		
		real d_incr = disparity_increment;
		real min_d = min_disparity[u], max_d = max_disparity[u];
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
	}
	
	return result;
}


}