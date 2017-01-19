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

#include "projection_view_frustum.h"
#include "depth_projection_parameters.h"
#include <array>
#include <cmath>
#include <stdexcept>

#include "math_constants.h"
#include "../eigen.h"

namespace mf {
	
	
depth_projection_parameters projection_view_frustum::depth_projection_(real z_near, real z_far) {
	depth_projection_parameters dparam;
	dparam.d_near = 0.0;
	dparam.d_far = 1.0;
	if(z_near > 0.0) {
		dparam.z_near = z_near;
		dparam.z_far = z_far;
		dparam.flip_z = false;
	} else {
		dparam.z_near = -z_near;
		dparam.z_far = -z_far;
		dparam.flip_z = true;
	}
	return dparam;
}


projection_view_frustum projection_view_frustum::symmetric_perspective
(real near_width, real near_height, real z_near, real z_far) {
	depth_projection_parameters dparam = depth_projection_(z_near, z_far);
	real two_nz = 2.0 * dparam.z_near;

	Eigen_mat4 mat; mat <<
		two_nz / near_width, 0.0, 0.0, 0.0,
		0.0, two_nz / near_height, 0.0, 0.0,
		0.0, 0.0, dparam.offset(), dparam.factor(),
		0.0, 0.0, (dparam.flip_z ? -1.0 : 1.0), 0.0;
				
	return projection_view_frustum(mat);
}


projection_view_frustum projection_view_frustum::symmetric_perspective_fov
(angle near_width, angle near_height, real z_near, real z_far) {
	depth_projection_parameters dparam = depth_projection_(z_near, z_far);

	Eigen_mat4 mat; mat <<
		std::tan(half_pi - near_width), 0.0, 0.0, 0.0,
		0.0, std::tan(half_pi - near_height), 0.0, 0.0,
		0.0, 0.0, dparam.offset(), dparam.factor(),
		0.0, 0.0, (dparam.flip_z ? -1.0 : 1.0), 0.0;
				
	return projection_view_frustum(mat);
}


projection_view_frustum projection_view_frustum::symmetric_perspective_fov_x
(angle near_width, real aspect_ratio, real z_near, real z_far) {
	depth_projection_parameters dparam = depth_projection_(z_near, z_far);
	real w = std::tan(half_pi - near_width);
	real h = w / aspect_ratio;

	Eigen_mat4 mat; mat <<
		w, 0.0, 0.0, 0.0,
		0.0, h, 0.0, 0.0,
		0.0, 0.0, dparam.offset(), dparam.factor(),
		0.0, 0.0, (dparam.flip_z ? -1.0 : 1.0), 0.0;
				
	return projection_view_frustum(mat);
}



projection_view_frustum projection_view_frustum::symmetric_perspective_fov_y
(angle near_height, real aspect_ratio, real z_near, real z_far) {
	depth_projection_parameters dparam = depth_projection_(z_near, z_far);
	real h = std::tan(half_pi - near_height);
	real w = h * aspect_ratio;

	Eigen_mat4 mat; mat <<
		w, 0.0, 0.0, 0.0,
		0.0, h, 0.0, 0.0,
		0.0, 0.0, dparam.offset(), dparam.factor(),
		0.0, 0.0, (dparam.flip_z ? -1.0 : 1.0), 0.0;
				
	return projection_view_frustum(mat);
}



projection_view_frustum projection_view_frustum::asymmetric_perspective
(distance_pair near_x, distance_pair near_y, real z_near, real z_far) {
	depth_projection_parameters dparam = depth_projection_(z_near, z_far);
	real two_nz = 2.0 * dparam.z_near;
	real x_diff = near_x.second - near_x.first;
	real y_diff = near_y.second - near_y.first;

	Eigen_mat4 mat; mat <<
		two_nz / x_diff, 0.0, (near_x.first + near_x.second)/x_diff, 0.0,
		0.0, two_nz / y_diff, (near_y.first + near_y.second)/y_diff, 0.0,
		0.0, 0.0, dparam.offset(), dparam.factor(),
		0.0, 0.0, (dparam.flip_z ? -1.0 : 1.0), 0.0;
				
	return projection_view_frustum(mat);
}


projection_view_frustum projection_view_frustum::asymmetric_perspective_fov
(angle_pair near_x, angle_pair near_y, real z_near, real z_far) {
	angle_pair near_x_proj { std::tan(near_x.first) * z_near, std::tan(near_x.second) * z_near };
	angle_pair near_y_proj { std::tan(near_y.first) * z_near, std::tan(near_y.second) * z_near };
	return asymmetric_perspective(near_x_proj, near_y_proj, z_near, z_far);
}



real projection_view_frustum::aspect_ratio() const {
	return view_projection_matrix()(1, 1) / view_projection_matrix()(0, 0);
}


bool projection_view_frustum::is_symmetric_x() const {
	return (view_projection_matrix()(0, 2) == 0.0);
}


bool projection_view_frustum::is_symmetric_y() const {
	return (view_projection_matrix()(1, 2) == 0.0);
}


bool projection_view_frustum::is_symmetric() const {
	return is_symmetric_x() && is_symmetric_y();
}


void projection_view_frustum::adjust_fov_x_to_aspect_ratio(real aspect_ratio) {
	if(! is_symmetric()) throw std::logic_error("cannot adjust frustum to image size when it is not symmetric");
	view_projection_matrix()(0, 0) = view_projection_matrix()(1, 1) / aspect_ratio;
}

void projection_view_frustum::adjust_fov_y_to_aspect_ratio(real aspect_ratio) {
	if(! is_symmetric()) throw std::logic_error("cannot adjust frustum to image size when it is not symmetric");
	view_projection_matrix()(1, 1) = view_projection_matrix()(0, 0) * aspect_ratio;
}



}
