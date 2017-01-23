#include "opencl.h"
#include "common.h"
#include <vector>

namespace mf {

cl::Context get_cl_context() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	Assert(all_platforms.size() > 0);
	cl::Platform default_platform = all_platforms.back();

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &all_devices);
	Assert(all_devices.size() > 0);
	cl::Device default_device = all_devices.front();

	return cl::Context({default_device});
}

}
