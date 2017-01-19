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

#include "plane.h"
#include <vector>
#include "../utility/string.h"

namespace mf {

plane::plane(Eigen_scalar a, Eigen_scalar b, Eigen_scalar c, Eigen_scalar d) :
normal(a, b, c), distance(d) {
	normalize();
}


plane::plane(const Eigen_vec3& p, const Eigen_vec3& n) :
normal(n), distance(n.dot(p)) {
	normalize();
}


plane::plane(const Eigen_vec3& p1, const Eigen_vec3& p2, const Eigen_vec3& p3) :
plane(p1, (p2 - p1).cross(p3 - p1)) { }


plane::plane(const pose& ps) :
plane(ps.position, ps.orientation * Eigen_vec3::UnitY()) {}

void plane::normalize() {
	Eigen_scalar norm = normal.norm();
	normal /= norm;
	distance /= norm;
}


Eigen_vec3 plane::project(const Eigen_vec3& p) const {
	Eigen_scalar a = normal[0], b = normal[1], c = normal[2];
	Eigen_scalar k = a*p[0] + b*p[1] + c*p[2] - distance;
	return Eigen_vec3(p[0] - a*k, p[1] - b*k, p[2] - c*k);
}


Eigen_vec3 plane::origin() const {
	return Eigen_vec3(normal[0]*distance, normal[1]*distance, normal[2]*distance);
}


Eigen_scalar signed_distance(const Eigen_vec3& pt, const plane& pl) {
	return pl.normal.dot(pt) + pl.distance;
}


Eigen_scalar distance(const Eigen_vec3& pt, const plane& pl) {
	Eigen_scalar d = signed_distance(pt, pl);
	return (d > 0.0 ? d : -d);
}

pose plane::to_pose() const {
	Eigen_vec3 translation = project(Eigen_vec3::Zero());
	Eigen_quaternion orientation = Eigen_quaternion::FromTwoVectors(Eigen_vec3::UnitY(), normal);
	return pose(translation, orientation);
}


plane::operator Eigen_hyperplane3() const {
	return Eigen_hyperplane3(origin(), normal);
}


void plane::apply_transformation(const Eigen_affine3& t) {
	normal = (t * Eigen_vec4(normal[0], normal[1], normal[2], 0.0)).head(3);
	distance += normal.dot(t.translation());
}

std::string plane::to_string() const {
	return implode_to_string<Eigen_scalar>(',', {
		normal[0],
		normal[1],
		normal[2],
		distance
	});
}


plane plane::from_string(const std::string& str) {
	std::vector<Eigen_scalar> p = explode_from_string<Eigen_scalar>(',', str);
	if(p.size() != 4)
		throw std::invalid_argument("invalid string to convert to plane");
	
	return plane(p[0], p[1], p[2], p[3]);
}




}
