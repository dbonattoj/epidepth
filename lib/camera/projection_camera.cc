#include "projection_camera.h"
#include "../geometry/projection_view_frustum.h"

#include <iostream>

namespace mf {

void projection_camera::update_world_matrices_() {
	world_to_image_ = view_to_image_ * absolute_pose().transformation_from_world();
	image_to_world_ = world_to_image_.inverse();
}


void projection_camera::do_update_pose() {
	update_world_matrices_();
}


void projection_camera::scale(real factor) {
	view_to_image_(0, 0) *= factor;
	view_to_image_(1, 1) *= factor;
	view_to_image_(0, 2) *= factor;
	view_to_image_(1, 2) *= factor;

	update_world_matrices_();
}


Eigen_mat4 projection_camera::projection_matrix_
(const intrinsic_matrix_type& intr, const depth_projection_parameters& dparam) {
	Assert(intr(2, 0) == 0.0);
	Assert(intr(2, 1) == 0.0);
	Assert(intr(2, 2) == 1.0);

	Eigen_mat4 proj; proj <<
		intr(0, 0), intr(0, 1), intr(0, 2), 0.0,
		intr(1, 0), intr(1, 1), intr(1, 2), 0.0,
		0.0, 0.0, dparam.offset(), dparam.factor(),
		0.0, 0.0, 1.0, 0.0;
		
	if(dparam.flip_z) {
		proj(2, 2) = -proj(2, 2);
		proj(3, 2) = -1.0;
	}
	
	return proj;
}


projection_camera::projection_camera
(const pose& ps, const intrinsic_matrix_type& intr, const depth_projection_parameters& dparam) :
	depth_camera(ps),
	depth_parameters_(dparam),
	view_to_image_(projection_matrix_(intr, dparam))
{
	update_world_matrices_();
}
	

Eigen_mat3 projection_camera::intrinsic_matrix() const {
	const Eigen_mat4& view_to_image = view_to_image_.matrix();
	Eigen_mat3 intrinsic; intrinsic <<
		view_to_image(0, 0), view_to_image(0, 1), view_to_image(0, 2),
		view_to_image(1, 0), view_to_image(1, 1), view_to_image(1, 2),
		view_to_image(3, 0), view_to_image(3, 1), view_to_image(3, 2);
	return intrinsic;
}


Eigen_projective3 homography_transformation(const projection_camera& from, const projection_camera& to) {
	return to.world_to_image_transformation() * from.image_to_world_transformation();
}


Eigen_mat3 fundamental_matrix(const projection_camera& from, const projection_camera& to) {
	return to.intrinsic_matrix().inverse().transpose() * essential_matrix(from, to) * from.intrinsic_matrix().inverse();
}

}
