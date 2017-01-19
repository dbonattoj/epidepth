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

#include "pose.h"
#include "angle.h"
#include "math_constants.h"
#include "../utility/string.h"
#include "spherical_coordinates.h"
#include <stdexcept>

#include <iostream>

namespace mf {

pose::pose() :
	position(Eigen_vec3::Zero()),
	orientation(Eigen_quaternion::Identity()) { }



pose::pose(const Eigen_affine3& t) :
	position(t.translation()),
	orientation(t.rotation())
{
	orientation.normalize();
}


Eigen_affine3 pose::transformation_from_world() const {
	return orientation.conjugate() * Eigen_translation3(-position);
}

Eigen_affine3 pose::transformation_to_world() const {
	return Eigen_translation3(position) * orientation;
}

Eigen_vec3 pose::euler_angles(std::ptrdiff_t a0, std::ptrdiff_t a1, std::ptrdiff_t a2) const {
	return orientation.toRotationMatrix().eulerAngles(a0, a1, a2);
}


std::ostream& operator<<(std::ostream& str, const pose& ps) {
	Eigen_vec3 euler = ps.euler_angles();
	str << "position (" << ps.position[0] << ", " << ps.position[1] << ", " << ps.position[2] << "); "
	<< " orientation (" << angle(euler[0]) << ", " << angle(euler[1]) << ", " << angle(euler[2]) << ")";
	return str;
}

std::string pose::to_string() const {
	return implode_to_string<Eigen_scalar>(',', {
		position[0],
		position[1],
		position[2],
		orientation.w(),
		orientation.x(),
		orientation.y(),
		orientation.z()
	});
}

pose pose::from_string(const std::string& str) {
	std::vector<Eigen_scalar> p = explode_from_string<Eigen_scalar>(',', str);
	if(p.size() != 7)
		throw std::invalid_argument("invalid string to convert to pose");
	
	Eigen_vec3 position(p[0], p[1], p[2]);
	Eigen_quaternion orientation(p[3], p[4], p[5], p[6]);
	return pose(position, orientation);
}


void pose::look_at(const Eigen_vec3& target) {
	Eigen_vec3 at_target = target - position;
	Eigen_vec3 at_depth(0, 0, 1);
	orientation.setFromTwoVectors(at_depth, at_target);
	orientation.normalize();
}

void pose::flip(const Eigen_vec3& axis) {
	orientation = Eigen_angleaxis(pi, axis) * orientation;
}


void pose::invert_orientation() {
	orientation = orientation.inverse();
}


pose interpolate(const pose& a, const pose& b, real t) {
	Eigen_vec3 new_position = (1.0-t)*a.position + t*b.position;
	Eigen_quaternion new_orientation = a.orientation.slerp(t, b.orientation);
	return pose(new_position, new_orientation);
}


}
