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

#ifndef MF_BOUNDING_BOX_H_
#define MF_BOUNDING_BOX_H_

#include <iosfwd>
#include <array>
#include <cmath>
#include <utility>
#include <string>
#include "../eigen.h"

namespace mf {

/// Axis-aligned 3D cuboid bounding box.
/** Represented using origin and extremity coordinates.
 ** Point is inside iff for every coordinate `i`, `origin[i] <= p[i] < extremity[i]`. */
struct bounding_box {
	using edge = std::pair<Eigen_vec3, Eigen_vec3>;
	using face = std::array<Eigen_vec3, 4>;

	using corners_array = std::array<Eigen_vec3, 8>;
	using edges_array = std::array<edge, 12>;
	using faces_array = std::array<face, 6>;

	Eigen_vec3 origin;
	Eigen_vec3 extremity;
	
	bounding_box() = default;
	bounding_box(const bounding_box&) = default;
	bounding_box(const Eigen_vec3& o, const Eigen_vec3& e);
	
	Eigen_vec3 center() const;
	Eigen_vec3 side_lengths() const;
	Eigen_scalar side_length(std::ptrdiff_t i = 0) const;
	Eigen_scalar volume() const;
	
	void expand_extremity(Eigen_scalar ep = 0.1);
	
	bool contains(const Eigen_vec3&) const;
	bool contains(const bounding_box&) const;
	
	corners_array corners() const;
	edges_array edges() const;
	faces_array faces() const;
	
	std::string to_string() const;
	static bounding_box from_string(const std::string&);
		
	friend std::ostream& operator<<(std::ostream&, const bounding_box&);
};



Eigen_scalar minimal_distance_sq(const bounding_box& a, const bounding_box& b);
Eigen_scalar maximal_distance_sq(const bounding_box& a, const bounding_box& b);

Eigen_scalar minimal_distance_sq(const Eigen_vec3& p, const bounding_box& b);
Eigen_scalar maximal_distance_sq(const Eigen_vec3& p, const bounding_box& b);

inline Eigen_scalar maximal_distance(const bounding_box& a, const bounding_box& b) {
	return std::sqrt( maximal_distance_sq(a, b) );
}

inline Eigen_scalar minimal_distance(const bounding_box& a, const bounding_box& b) {
	return std::sqrt( minimal_distance_sq(a, b) );
}


inline Eigen_scalar minimal_distance(const Eigen_vec3& p, const bounding_box& b) {
	return std::sqrt( minimal_distance_sq(p, b) );
}

inline Eigen_scalar maximal_distance(const Eigen_vec3& p, const bounding_box& b) {
	return std::sqrt( maximal_distance_sq(p, b) );
}


}

#endif
