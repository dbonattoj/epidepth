#ifndef MF_OPENCL_H_
#define MF_OPENCL_H_

namespace mf {

cl::Context get_cl_context() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	assert(all_platforms.size() > 0);
	cl::Platform default_platform = all_platforms.back();

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &all_devices);
	assert(all_devices.size() > 0);
	cl::Device default_device = all_devices.front();

	return cl::Context({default_device});
}

}

#endif
