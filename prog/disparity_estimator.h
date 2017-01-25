/*
#ifndef MF_DISPARITY_ESTIMATOR_H_
#define MF_DISPARITY_ESTIMATOR_H_

#include "opencl.h"
#include "nd.h"

namespace mf {

class disparity_estimator {
private:
	cl::CommandQueue& queue_;
	
public:
	explicit disparity_estimator(
		const ndarray_view<2, rgb_color>& epi,
		const ndarray_view<2, real>& conf,
		const ndarray_view<2, bool>& mask,
		const ndarray_view<2, real>& min_disparity,
		const ndarray_view<2, real>& max_disparity,
		
		const ndarray_view<2, real> output_disparity,
		const ndarray_view<2, real> output_confidence
	);
	
	void enqueue_epi_line_disparity(std::ptrdiff_t s);
	
};

};

#endif
*/