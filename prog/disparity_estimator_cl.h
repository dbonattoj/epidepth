#ifndef MF_DISPARITY_ESTIMATOR_CL_H_
#define MF_DISPARITY_ESTIMATOR_CL_H_

#include "disparity_estimator.h"
#include "opencl.h"
#include "nd.h"
#include "global.h"

namespace mf {

struct cl_worker;

class disparity_estimator_cl : public disparity_estimator {
private:
	cl_worker& worker_;

	std::ptrdiff_t v_min_;
	std::ptrdiff_t v_max_;
	ndarray_view<3, real> edge_conf_;

	cl::CommandQueue& queue_;
	
	cl::Image2DArray epi_image_array_;

	cl::Buffer min_disp_buffer_;
	cl::Buffer max_disp_buffer_;
	cl::Buffer mask_buffer_;
	
	cl::Buffer output_max_score_buffer_;
	cl::Buffer output_max_score_disp_buffer_;
	cl::Buffer output_avg_score_buffer_;
	
	static cl::Image2DArray get_epi_image_array_(const cl_worker& worker, const ndarray_view<3, rgba_color>& epi);
	
public:
	static void setup_cl();
	static int workers_count();
	
	disparity_estimator_cl(
		int worker_index,
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

};

#endif
