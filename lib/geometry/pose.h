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

#ifndef MF_POSE_H_
#define MF_POSE_H_

#include "../eigen.h"
#include <ostream>
#include <string>
#include "angle.h"

namespace mf {

/// Position and orientation in space.
/** Represented in terms of a position, and orientation in world space. The orientation is in the translated space,
 ** a pose of an object means that the object was first translated to pose.position, then rotated by pose.orientation.
 ** A pose also defines an orthonormal coordinate system, aka the world as seen from a camera at that pose. */
class pose {
public:
	Eigen_vec3 position; ///< Position vector.
	Eigen_quaternion orientation; ///< Orientation quaternion. Must be kept normalized.
	// TODO make private, enforce normalization, change to matrix?

public:
	/// Create identity pose.
	pose();
		
	/// Copy-construct pose.
	pose(const pose&) = default;
	
	/// Construct pose from affine transformation.
	/** Transformation is transformation to world.
	 ** Assumes that the transformation consists only of translation and rotation. */
	pose(const Eigen_affine3&);
	
	/// Construct pose with given translation and rotation.
	template<typename Translation, typename Rotation>
	pose(const Translation& t, const Rotation& r) :
		position(t),
		orientation(r) { orientation.normalize(); }	
	
	/// Affine transformation from world space to pose coordinate system.
	Eigen_affine3 transformation_from_world() const;

	/// Affine transformation from pose coordinate system to world space.
	Eigen_affine3 transformation_to_world() const;
	
	Eigen_affine3 transformation_from(const pose& ps) const {
		return transformation_from_world() * ps.transformation_to_world();
	}
	
	Eigen_affine3 transformation_to(const pose& ps) const {
		return ps.transformation_from_world() * transformation_to_world();
	}
		
	Eigen_vec3 euler_angles(std::ptrdiff_t a0 = 0, std::ptrdiff_t a1 = 1, std::ptrdiff_t a2 = 2) const;
	
	std::string to_string() const;
	static pose from_string(const std::string&);
		
	Eigen_vec3 transform_from_world(const Eigen_vec3& p) const {
		return transformation_from_world() * p;
	}
	
	Eigen_vec3 transform_to_world(const Eigen_vec3& p) const {
		return transformation_to_world() * p;
	}
	
	void look_at(const Eigen_vec3&);
	
	void flip(const Eigen_vec3& axis = Eigen_vec3::UnitY());
	void invert_orientation();
};

/// Interpolate between two poses for `t = 0..1`
/** Linearily interpolated positions such that `pos = (1-t)*a.pos + t*b.pos`. Slerp interpolation of orientation. */
pose interpolate(const pose& a, const pose& b, real t);


std::ostream& operator<<(std::ostream&, const pose&);

}

#endif
