#include "sparse_epi.h"
#include "global.h"

#include <algorithm>
#include <cmath>

namespace mf {

void sparse_epi::add_segment(const sparse_epi_segment& seg) {
	segments_.push_back(seg);
}

ndarray<2, rgba_color> sparse_epi::reconstruct() const {
	auto cmp = [](const sparse_epi_segment& a, const sparse_epi_segment& b) { return (a.d < b.d); };
	std::sort(segments_.begin(), segments_.end(), cmp);
	
	ndarray<2, rgba_color> epi(shape_);
	rgba_color background(0, 0, 0);
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


ndarray<1, rgba_color> sparse_epi::reconstruct_line(real s) const {
	auto cmp = [](const sparse_epi_segment& a, const sparse_epi_segment& b) { return (a.d < b.d); };
	std::sort(segments_.begin(), segments_.end(), cmp);
	
	ndarray<1, rgba_color> epi_line(make_ndsize(u_sz));
	rgba_color background(0, 0, 0);
	std::fill(epi_line.begin(), epi_line.end(), background);
	
	for(const sparse_epi_segment& segment : segments_) {
		const rgba_color& col = segment.avg_color;
		
		real uf = segment.u  + (segment.s - s)*segment.d;
		
		real uf_int;
		real uf_fra = std::modf(uf, &uf_int);
		std::ptrdiff_t u = uf_int;
		
		if(u > 0 && u < shape_[0]) epi_line[u] = color_blend(epi_line[u], col, 1.0-uf_fra);
		if(u > 0 && u < shape_[0]-1) epi_line[u+1] = color_blend(epi_line[u+1], col, uf_fra);
	}
	return epi_line;
}


};