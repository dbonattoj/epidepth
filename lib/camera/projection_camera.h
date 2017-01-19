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

#ifndef MF_PROJECTION_CAMERA_H_
#define MF_PROJECTION_CAMERA_H_

#include "depth_camera.h"
#include "../geometry/depth_projection_parameters.h"
#include "../nd.h"
#include <utility>

namespace mf {

/// Pin-hole camera with projection to a planar image space.
class projection_camera : public depth_camera {
public:
	using intrinsic_matrix_type = Eigen_mat3;

private:
	depth_projection_parameters depth_parameters_;
	Eigen_projective3 view_to_image_;
	Eigen_projective3 world_to_image_;
	Eigen_projective3 image_to_world_;
	
	static Eigen_mat4 projection_matrix_(const intrinsic_matrix_type&, const depth_projection_parameters&);
	void update_world_matrices_();

protected:
	void do_update_pose() override;

public:
	projection_camera(const pose&, const intrinsic_matrix_type&, const depth_projection_parameters&);
	
	projection_camera(const projection_camera&) = default;
	projection_camera& operator=(const projection_camera&) = default;
	
	const depth_projection_parameters& depth_parameters() const { return depth_parameters_; }
	const Eigen_projective3& view_to_image_transformation() const { return view_to_image_; }
	const Eigen_projective3& world_to_image_transformation() const { return world_to_image_; }
	const Eigen_projective3& image_to_world_transformation() const { return image_to_world_; }
	Eigen_mat3 intrinsic_matrix() const;

	virtual void scale(real factor);

	depth_type depth(orthogonal_distance_type z) const override {
		return depth_parameters_.depth(z);
	}
	
	image_coordinates_type project(const Eigen_vec3& p) const override {
		return (world_to_image_ * p.homogeneous()).eval().hnormalized().head(2);
	}
	
	depth_image_coordinates_type depth_project(const Eigen_vec3& p) const override {
		return (world_to_image_ * p.homogeneous()).eval().hnormalized().head(3);
	}
	
	Eigen_vec3 ray_direction(const image_coordinates_type& c) const override {
		real depth = depth_parameters_.d_near;
		Eigen_vec3 p(c[0], c[1], depth);
		return (image_to_world_ * p.homogeneous()).eval().hnormalized().normalized();
	}

	orthogonal_distance_type orthogonal_distance(depth_type d) const override {
		return depth_parameters_.orthogonal_distance(d);
	}
	
	Eigen_vec3 point(const depth_image_coordinates_type& cd) const override {
		return (image_to_world_ * cd.homogeneous()).eval().hnormalized();
	}
};


Eigen_projective3 homography_transformation(const projection_camera& from, const projection_camera& to);

Eigen_mat3 fundamental_matrix(const projection_camera& from, const projection_camera& to);

}


#endif
