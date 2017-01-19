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

#ifndef MF_VIEW_FRUSTUM_H_
#define MF_VIEW_FRUSTUM_H_

#include <utility>
#include <array>
#include "angle.h"
#include "../eigen.h"
#include "depth_projection_parameters.h"
#include "bounding_box.h"
#include "pose.h"
// TODO bounding_box -> Eigen_alignedbox

namespace mf {

/// View frustum of a perspective camera.
class view_frustum {
public:	
	enum intersection {
		outside_frustum,
		inside_frustum,
		partially_inside_frustum
	};
	
	using edge = std::pair<Eigen_vec3, Eigen_vec3>;
	using corners_array = std::array<Eigen_vec3, 8>;
	using edges_array = std::array<edge, 12>;
	using planes_array = std::array<Eigen_hyperplane3, 6>;

	explicit view_frustum(const Eigen_mat4& mat);

protected:
	view_frustum(const pose& ps, const Eigen_mat4& proj_mat);
	view_frustum() = delete;

private:
	Eigen_mat4 matrix_;

public:
	const Eigen_mat4& view_projection_matrix() const { return matrix_; }	
	Eigen_mat4& view_projection_matrix() { return matrix_; }	
	
	Eigen_hyperplane3 near_plane() const;
	Eigen_hyperplane3 far_plane() const;
	Eigen_hyperplane3 left_plane() const;
	Eigen_hyperplane3 right_plane() const;
	Eigen_hyperplane3 bottom_plane() const;
	Eigen_hyperplane3 top_plane() const;
	
	planes_array planes() const;
	corners_array corners() const;
	edges_array edges() const;

	bool contains(const Eigen_vec3&, bool consider_z_planes = true) const;
	intersection contains(const bounding_box&) const;
	static intersection contains(const planes_array&, const bounding_box&);
	
	view_frustum transform(const Eigen_affine3&) const;
};

}

#endif
