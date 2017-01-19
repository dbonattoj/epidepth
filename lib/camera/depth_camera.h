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

#ifndef MF_DEPTH_CAMERA_H_
#define MF_DEPTH_CAMERA_H_

#include "camera.h"

namespace mf {

/// Camera which maps 3D coordinates of point to depth value in addition to the 2D image coordinates, base class.
class depth_camera : public camera {
public:
	using depth_type = real;
	using depth_image_coordinates_type = Eigen_vec3;

	using camera::camera;

	virtual Eigen_vec3 point(const depth_image_coordinates_type&) const = 0;

	virtual orthogonal_distance_type orthogonal_distance(depth_type) const = 0;

	virtual depth_type depth(orthogonal_distance_type) const = 0;

	virtual depth_type depth(const Eigen_vec3& p) const {
		return this->depth(camera::orthogonal_distance(p));
	}
	
	virtual depth_type depth(const spherical_coordinates& sp) const {
		return this->depth(camera::point(sp));
	}

	virtual depth_image_coordinates_type depth_project(const Eigen_vec3& p) const = 0;
};
	
}

#endif
