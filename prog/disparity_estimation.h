#ifndef EPI_DISPARITY_ESTIMATION_H_
#define EPI_DISPARITY_ESTIMATION_H_

#include <cstddef>
#include <cstdint>
#include "common.h"
#include "nd.h"
#include "color.h"

namespace mf {


struct epi_line_disparity_result {
	ndarray<1, real> disparity;
	ndarray<1, real> confidence;
	std::ptrdiff_t s;
	
	epi_line_disparity_result(std::size_t sz, std::ptrdiff_t s_) :
		disparity(make_ndsize(sz)),
		confidence(make_ndsize(sz)),
		s(s_) { }
};



real depth_score(const ndarray_view<2, rgb_color>& epi, std::ptrdiff_t s, std::ptrdiff_t u, real d);


epi_line_disparity_result estimate_epi_line_disparity(
	std::ptrdiff_t s,
	const ndarray_view<2, rgb_color>& epi,
	const ndarray_view<1, real>& conf,
	const ndarray_view<1, bool>& mask,
	const ndarray_view<1, real>& min_disparity,
	const ndarray_view<1, real>& max_disparity
);


}

#endif
