#include "disparity_estimator_cl.h"
#include <streambuf>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace mf {

struct cl_worker {
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel kernel;
};

namespace {

static std::vector<cl_worker> workers_;

std::string program_source_() {
	static std::string source;
	if(source.empty()) {
		std::ifstream stream("../prog/disparity_estimation.cl");
		using iterator_type = std::istreambuf_iterator<char>;
		source = std::string(iterator_type(stream), iterator_type());
	}
	return source;
}

void install_cl_worker_(cl::Platform platform, cl::Device device) {
	cl::Context context(device);
	std::string source = program_source_();
	cl::Program program(context, source);
	std::string program_options = "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable -DS_SZ=" + std::to_string(s_sz);
	try {
		program.build({device}, program_options.c_str());
	} catch(const cl::Error& err) {
		std::string log;
		program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &log);
		std::cerr << "OpenCL program build failed. Log:\n" << log << std::endl;
		std::terminate();
	}
	cl::CommandQueue queue(context, device);
	
	cl::Kernel kernel(program, "estimate_epi_line_disparity");
	
	std::string device_name, device_vendor;
	device.getInfo(CL_DEVICE_NAME, &device_name);
	device.getInfo(CL_DEVICE_VENDOR, &device_vendor);
	std::cout << "cl worker #" << workers_.size() << ": " << device_vendor << " " << device_name << std::endl;

	workers_.emplace_back(cl_worker{
		std::move(platform),
		std::move(device),
		std::move(context),
		std::move(queue),
		std::move(kernel)
	});
};

};

void disparity_estimator_cl::setup_cl() {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	for(cl::Platform& platform : platforms) {
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		for(cl::Device& device : devices) install_cl_worker_(platform, device);
	}
}

int disparity_estimator_cl::workers_count() {
	return workers_.size();
}


cl::Image2DArray disparity_estimator_cl::get_epi_image_array_
(const cl_worker& worker, const ndarray_view<3, rgba_color>& epi) {
	cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
	return cl::Image2DArray(
		worker.context,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		format,
		epi.shape()[0], // v
		epi.shape()[2], // s -> width
		epi.shape()[1], // u -> height
		epi.strides()[1],
		epi.strides()[0],
		epi.start()
	);
}


disparity_estimator_cl::disparity_estimator_cl(
	int worker_index,
	std::ptrdiff_t v_min,
	std::ptrdiff_t v_max,
	const ndarray_view<3, rgba_color>& epi,
	const ndarray_view<3, real>& edge_conf,
	const ndarray_view<3, real>& min_disp,
	const ndarray_view<3, real>& max_disp
) :
	worker_(workers_.at(worker_index)),
	v_min_(v_min),
	v_max_(v_max),
	edge_conf_(edge_conf),
	queue_(worker_.queue),
	epi_image_array_(get_epi_image_array_(worker_, epi))
{
	std::size_t elem_count = epi.size();
	Assert(epi.has_default_strides_without_padding());
	static_assert(std::is_same<real, double>::value, "mf::real must be double");
	min_disp_buffer_ = cl::Buffer(worker_.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, elem_count*sizeof(double), min_disp.start());
	max_disp_buffer_ = cl::Buffer(worker_.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, elem_count*sizeof(double), max_disp.start());
	mask_buffer_ = cl::Buffer(worker_.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, u_sz*sizeof(double));
	output_max_score_buffer_ = cl::Buffer(worker_.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, u_sz*sizeof(double));
	output_max_score_disp_buffer_ = cl::Buffer(worker_.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, u_sz*sizeof(double));
	output_avg_score_buffer_ = cl::Buffer(worker_.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, u_sz*sizeof(double));
}

	
epi_line_disparity_result disparity_estimator_cl::estimate_epi_line_disparity
(std::ptrdiff_t v, std::ptrdiff_t s, const ndarray_view<1, uchar>& mask) {
	worker_.queue.enqueueWriteBuffer(mask_buffer_, true, 0, u_sz, mask.start());

	std::size_t steps = disparity_steps;
	cl::NDRange global_range(u_sz, steps);
	cl::NDRange local_range(1, steps);
	
	std::ptrdiff_t i = 0;
	worker_.kernel.setArg(i++, (cl_double)depth_score_color_max_threshold);
	worker_.kernel.setArg(i++, (cl_int)(v - v_min_));
	worker_.kernel.setArg(i++, (cl_int)s);
	worker_.kernel.setArg(i++, epi_image_array_);
	worker_.kernel.setArg(i++, min_disp_buffer_);
	worker_.kernel.setArg(i++, max_disp_buffer_);
	worker_.kernel.setArg(i++, mask_buffer_);
	worker_.kernel.setArg(i++, steps*sizeof(cl_double), nullptr);
	worker_.kernel.setArg(i++, output_max_score_buffer_);
	worker_.kernel.setArg(i++, output_max_score_disp_buffer_);
	worker_.kernel.setArg(i++, output_avg_score_buffer_);
	worker_.queue.enqueueNDRangeKernel(worker_.kernel, cl::NullRange, global_range, local_range);
	
	epi_line_disparity_result result;
	
	ndarray<1, real> max_score(make_ndsize(u_sz)), avg_score(make_ndsize(u_sz));
	worker_.queue.enqueueReadBuffer(output_max_score_buffer_, true, 0, u_sz*sizeof(real), max_score.start());
	worker_.queue.enqueueReadBuffer(output_max_score_disp_buffer_, true, 0, u_sz*sizeof(real), result.disparity.start());
	worker_.queue.enqueueReadBuffer(output_avg_score_buffer_, true, 0, u_sz*sizeof(real), avg_score.start());
	
	for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
		if(mask[u]) result.confidence[u] = edge_conf_[v - v_min_][u][s] * std::abs(max_score[u] - avg_score[u]);
		else { result.disparity[u] = NAN; result.confidence[u] = 0.0; }
	}
	
	return result;
}

}
