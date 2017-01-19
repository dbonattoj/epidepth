#include "sparse_epi.h"
#include <algorithm>

namespace mf {

void sparse_epi::add_segment(const sparse_epi_segment& seg) {
	segments_.push_back(seg);
}

ndarray<2, rgb_color> sparse_epi::reconstruct() const {
	auto cmp = [](const sparse_epi_segment& a, const sparse_epi_segment& b) { return (a.d < b.d); };
	std::sort(segments_.begin(), segments_.end(), cmp);
	
	ndarray<2, rgb_color> epi(shape_);
	rgb_color background(0, 0, 0);
	std::fill(epi.begin(), epi.end(), background);
	for(const sparse_epi_segment& segment : segments_) {
		for(std::ptrdiff_t s2 = 0; s2 < shape_[1]; ++s2) {
			std::ptrdiff_t u2 = segment.u  + (segment.s - s2)*segment.d;
			if(u2 < 0 || u2 >= shape_[0]) continue;
			epi[u2][s2] = segment.avg_color;
		}
	}
	return epi;
}


}