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

#ifndef MF_PLANE_H_
#define MF_PLANE_H_

#include "../eigen.h"
#include <string>
#include "pose.h"

namespace mf {

/// Oriented plane in 3D space.
/** Represented using normal vector and distance value. That is the values (a, b, c, d) in the plane equation
 ** ax + by + cz = d. When normalized, representations are unique and distance is from origin to plane. */
struct plane {
	Eigen_vec3 normal = Eigen_vec3::Zero();
	Eigen_scalar distance = 0;
	
	plane() = default;
	plane(const plane&) = default;
	explicit plane(const pose&);
	
	/// Create from coefficients.
	/// Creates plane for equation ax + by + cz = d. Resulting plane is normalized.
	plane(Eigen_scalar a, Eigen_scalar b, Eigen_scalar c, Eigen_scalar d);
	
	/// Create from any point on plane and normal vector.
	/// Normal vector does not need to be normalized. Resulting plane is normalized.
	plane(const Eigen_vec3& p, const Eigen_vec3& n);
	
	/// Create any three points on the plane.
	/// Resulting plane is normalized. Points must not be aligned or coincide.
	plane(const Eigen_vec3& p1, const Eigen_vec3& p2, const Eigen_vec3& p3);
	
	/// Normalize the plane representation.
	void normalize();
	
	/// Project point on plane.
	/// Result is the point on plane that is closest to the given point.
	Eigen_vec3 project(const Eigen_vec3&) const;
	
	Eigen_vec3 origin() const;
	
	pose to_pose() const;
	
	operator Eigen_hyperplane3() const;
	
	void apply_transformation(const Eigen_affine3&);

	std::string to_string() const;
	static plane from_string(const std::string&);
};


Eigen_scalar signed_distance(const Eigen_vec3&, const plane&);
Eigen_scalar distance(const Eigen_vec3&, const plane&);


}

#endif
