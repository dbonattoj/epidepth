#ifndef MF_DISPARITY_ESTIMATOR_H_
#define MF_DISPARITY_ESTIMATOR_H_

#include "common.h"
#include "global.h"
#include "nd.h"
#include "opencv.h"

namespace mf {

struct epi_line_disparity_result {
	ndarray<1, real> disparity;
	ndarray<1, real> confidence;
	
	epi_line_disparity_result() :
		disparity(make_ndsize(u_sz)),
		confidence(make_ndsize(u_sz)) { }
};


class disparity_estimator {
public:
	virtual ~disparity_estimator();
	virtual epi_line_disparity_result estimate_epi_line_disparity
		(std::ptrdiff_t v, std::ptrdiff_t s, const ndarray_view<1, uchar>& mask) = 0;
};

};

#endif
