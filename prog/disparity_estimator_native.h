#ifndef MF_DISPARITY_ESTIMATOR_NATIVE_H_
#define MF_DISPARITY_ESTIMATOR_NATIVE_H_

#include "disparity_estimator.h"
#include "utility.h"
#include "global.h"

namespace mf {

class disparity_estimator_native : public disparity_estimator {
private:
	std::ptrdiff_t v_min_;
	std::ptrdiff_t v_max_;
	ndarray_view<3, rgba_color> epi_;
	ndarray_view<3, real> edge_conf_;
	ndarray_view<3, uchar> edge_conf_mask_;
	ndarray_view<3, real> min_disp_;
	ndarray_view<3, real> max_disp_;
	
	real depth_score_(const ndarray_view<2, rgba_color>& epi, std::ptrdiff_t s, std::ptrdiff_t u, real d);
	
public:
	disparity_estimator_native(
		std::ptrdiff_t v_min,
		std::ptrdiff_t v_max,
		const ndarray_view<3, rgba_color>& epi,
		const ndarray_view<3, real>& edge_conf,
		const ndarray_view<3, real>& min_disp,
		const ndarray_view<3, real>& max_disp
	);
	
	epi_line_disparity_result estimate_epi_line_disparity
		(std::ptrdiff_t v, std::ptrdiff_t s, const ndarray_view<1, uchar>& mask) override;
};

}

#endif