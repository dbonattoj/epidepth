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

#ifndef MF_PROJECTION_VIEW_FRUSTUM_H_
#define MF_PROJECTION_VIEW_FRUSTUM_H_

#include "view_frustum.h"
#include "angle.h"
#include "../eigen.h"
#include <utility>
#include <array>

namespace mf {
	
struct depth_projection_parameters;

/// View frustum of a perspective camera, without pose.
/** Frustum apex is at camera center, axis-aligned, and pointing in +Z or -Z direction.
 ** Can be created for perspective projection. */
class projection_view_frustum : public view_frustum {
private:
	projection_view_frustum(const Eigen_mat4& mat) : view_frustum(pose(), mat) { }

	static depth_projection_parameters depth_projection_(real z_near, real z_far);

public:
	using distance_pair = std::pair<real, real>;
	using angle_pair = std::pair<angle, angle>;
	
	/// Create with symmetric perspective, given width and height of near clipping plane.
	static projection_view_frustum symmetric_perspective
		(real near_width, real near_height, real z_near, real z_far);
		
	/// Create with symmetric perspective, given horizontal and vertical field of view angles.
	static projection_view_frustum symmetric_perspective_fov
		(angle near_width, angle near_height, real z_near, real z_far);
	
	/// Create with symmetric perspective, given horizontal field of view and image aspect ratio.
	static projection_view_frustum symmetric_perspective_fov_x
		(angle near_width, real aspect_ratio, real z_near, real z_far);

	/// Create with symmetric perspective, given vertical field of view and image aspect ratio.		
	static projection_view_frustum symmetric_perspective_fov_y
		(angle near_height, real aspect_ratio, real z_near, real z_far);
	
	/// Create with asymmetric perspective, given horizontal and vertical ranges of near clipping plane.		
	static projection_view_frustum asymmetric_perspective
		(distance_pair near_x, distance_pair near_y, real z_near, real z_far);

	/// Create with asymmetric perspective, given horizontal and vertical limit angles of field of view.		
	static projection_view_frustum asymmetric_perspective_fov
		(angle_pair near_x, angle_pair near_y, real z_near, real z_far);	
	
	const Eigen_mat4& projection_matrix() const {
		return view_frustum::view_projection_matrix();
	}
	
	real aspect_ratio() const;
	bool is_symmetric_x() const;
	bool is_symmetric_y() const;
	bool is_symmetric() const;
	
	void adjust_fov_x_to_aspect_ratio(real aspect_ratio);
	void adjust_fov_y_to_aspect_ratio(real aspect_ratio);
};

}

#endif
