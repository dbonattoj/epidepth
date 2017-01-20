#ifndef MF_OPENCL_H_
#define MF_OPENCL_H_

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

namespace mf {

cl::Context get_cl_context();

}

#endif