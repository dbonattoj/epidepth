/*
Author : Tim Lenertz
Date : May 2016

Copyright (c) 2016, Université libre de Bruxelles

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files to deal in the Software without restriction, including the rights to use, copy, modify, merge,
publish the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MF_EIGEN_H_
#define MF_EIGEN_H_

#if defined(MF_COMPILER_CLANG) || defined(MF_COMPILER_GCC)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wdeprecated-register"
	// Eigen uses C++14 deprecated 'register' keyword
#endif

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#if defined(MF_COMPILER_CLANG) || defined(MF_COMPILER_GCC)
	#pragma GCC diagnostic pop
#endif

#include "common.h"
#include "nd.h"

namespace mf {

/// Floating point type used for Eigen objects.
using Eigen_scalar = real;

template<std::size_t Rows, std::size_t Cols>
using Eigen_mat = Eigen::Matrix<Eigen_scalar, Rows, Cols>;
	
using Eigen_vec2 = Eigen_mat<2, 1>;
using Eigen_vec3 = Eigen_mat<3, 1>;
using Eigen_vec4 = Eigen_mat<4, 1>;
	
using Eigen_mat2 = Eigen_mat<2, 2>;
using Eigen_mat3 = Eigen_mat<3, 3>;
using Eigen_mat4 = Eigen_mat<4, 4>;

using Eigen_translation2 = Eigen::Translation<Eigen_scalar, 2>;
using Eigen_translation3 = Eigen::Translation<Eigen_scalar, 3>;
using Eigen_angleaxis = Eigen::AngleAxis<Eigen_scalar>;

using Eigen_affine2 = Eigen::Transform<Eigen_scalar, 2, Eigen::Affine>;
using Eigen_affine3 = Eigen::Transform<Eigen_scalar, 3, Eigen::Affine>;

using Eigen_projective2 = Eigen::Transform<Eigen_scalar, 2, Eigen::Projective>;
using Eigen_projective3 = Eigen::Transform<Eigen_scalar, 3, Eigen::Projective>;

using Eigen_hyperplane2 = Eigen::Hyperplane<Eigen_scalar, 2>;
using Eigen_hyperplane3 = Eigen::Hyperplane<Eigen_scalar, 3>;

using Eigen_line2 = Eigen::ParametrizedLine<Eigen_scalar, 2>;
using Eigen_line3 = Eigen::ParametrizedLine<Eigen_scalar, 3>;

using Eigen_quaternion = Eigen::Quaternion<Eigen_scalar>;

}

#include "nd.h"

namespace tff {

/// Elem traits specialization for Eigen matrix type.
/** 2 dimensional matrix becomes 1 dimensional vector elem. (No support for multiple dimensions here) */
template<typename Scalar, std::size_t Rows, std::size_t Columns>
struct elem_traits<Eigen::Matrix<Scalar, Rows, Columns>> :
	elem_traits_base<
		Eigen::Matrix<Scalar, Rows, Columns>,
		Scalar,
		Rows * Columns,
		false
	> { };


}

#endif
