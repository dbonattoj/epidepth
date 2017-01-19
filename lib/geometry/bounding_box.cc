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

#include <cmath>
#include <cassert>
#include <ostream>

#include "bounding_box.h"
#include "../utility/string.h"

namespace mf {

bounding_box::bounding_box(const Eigen_vec3& o, const Eigen_vec3& e) :
origin(o), extremity(e) {
	for(std::ptrdiff_t i = 0; i < 3; ++i) assert(origin[i] <= extremity[i]);
}


Eigen_vec3 bounding_box::center() const {
	return (origin + extremity) / 2;
}

Eigen_vec3 bounding_box::side_lengths() const {
	return (extremity - origin);
}

Eigen_scalar bounding_box::side_length(std::ptrdiff_t i) const {
	return (extremity[i] - origin[i]);
}


Eigen_scalar bounding_box::volume() const {
	Eigen_vec3 slen = side_lengths();
	return slen[0] * slen[1] * slen[2];
}

void bounding_box::expand_extremity(Eigen_scalar ep) {
	extremity += Eigen_vec3(ep, ep, ep);
}

bool bounding_box::contains(const Eigen_vec3& p) const {
	return
		(origin[0] <= p[0]) && (p[0] < extremity[0]) &&
		(origin[1] <= p[1]) && (p[1] < extremity[1]) &&
		(origin[2] <= p[2]) && (p[2] < extremity[2]);		
}


bool bounding_box::contains(const bounding_box& cub) const {
	for(std::ptrdiff_t i = 0; i < 3; ++i)
		if(cub.extremity[i] < origin[i] || cub.origin[i] > extremity[i]) return false;
	return true;
}


bounding_box::corners_array bounding_box::corners() const {
	return {
		// Left square
		Eigen_vec3(origin[0], origin[1], origin[2]),
		Eigen_vec3(origin[0], origin[1], extremity[2]),
		Eigen_vec3(origin[0], extremity[1], origin[2]),
		Eigen_vec3(origin[0], extremity[1], extremity[2]),
		
		// Right square
		Eigen_vec3(extremity[0], origin[1], origin[2]),
		Eigen_vec3(extremity[0], origin[1], extremity[2]),
		Eigen_vec3(extremity[0], extremity[1], origin[2]),
		Eigen_vec3(extremity[0], extremity[1], extremity[2])
	};
}


bounding_box::edges_array bounding_box::edges() const {
	auto corn = corners();
	return {
		// Left square
		edge(corn[0], corn[1]),
		edge(corn[1], corn[3]),
		edge(corn[3], corn[2]),
		edge(corn[2], corn[0]),
		
		// Right square
		edge(corn[0+4], corn[1+4]),
		edge(corn[1+4], corn[3+4]),
		edge(corn[3+4], corn[2+4]),
		edge(corn[2+4], corn[0+4]),
		
		// Joining edges
		edge(corn[0], corn[0+4]),
		edge(corn[1], corn[1+4]),
		edge(corn[2], corn[2+4]),
		edge(corn[3], corn[3+4])
	};
}


bounding_box::faces_array bounding_box::faces() const {
	auto corn = corners();
	return {
		face { corn[0], corn[1], corn[3], corn[2] }, // left
		face { corn[0+4], corn[1+4], corn[3+4], corn[2+4] }, // right
		face { corn[2], corn[2+4], corn[3+4], corn[3] }, // top
		face { corn[0], corn[0+4], corn[1+4], corn[1] }, // bottom
		face { corn[0], corn[0+4], corn[2+4], corn[2] }, // back
		face { corn[1], corn[1+4], corn[3+4], corn[3] }, // back
	};
}


Eigen_scalar minimal_distance_sq(const bounding_box& a, const bounding_box& b) {
	Eigen_scalar dist = 0;
	for(std::ptrdiff_t i = 0; i < 3; ++i) {
		if(a.extremity[i] < b.origin[i]) {
			Eigen_scalar d = b.origin[i] - a.extremity[i];
			dist += d * d;
		} else if(b.extremity[i] < a.origin[i]) {
			Eigen_scalar d = a.origin[i] - b.extremity[i];
			dist += d * d;
		}
	}
	return dist;
}


Eigen_scalar maximal_distance_sq(const bounding_box& a, const bounding_box& b) {
	Eigen_vec3 ap, bp;
	for(std::ptrdiff_t i = 0; i < 3; ++i) {
		Eigen_scalar d1 = std::abs(a.extremity[i] - b.origin[i]);
		Eigen_scalar d2 = std::abs(b.extremity[i] - a.origin[i]);
		if(d1 > d2) {
			ap[i] = a.extremity[i];
			bp[i] = b.origin[i];
		} else {
			ap[i] = a.origin[i];
			bp[i] = b.extremity[i];		
		}
	}
	return (ap - bp).squaredNorm();
}


Eigen_scalar minimal_distance_sq(const Eigen_vec3& p, const bounding_box& b) {
	Eigen_scalar dist = 0;
	for(std::ptrdiff_t i = 0; i < 3; ++i) {
		if(p[i] < b.origin[i]) {
			Eigen_scalar d = b.origin[i] - p[i];
			dist += d * d;
		} else if(p[i] > b.extremity[i]) {
			Eigen_scalar d = p[i] - b.extremity[i];
			dist += d * d;
		}
	}
	return dist;
}


Eigen_scalar maximal_distance_sq(const Eigen_vec3& p, const bounding_box& b) {
	Eigen_scalar max_dist = 0;
	for(const Eigen_vec3& c : b.corners()) {
		Eigen_scalar d = (p - c).squaredNorm();
		if(d > max_dist) max_dist = d;
	}
	return max_dist;
}


std::string bounding_box::to_string() const {
	return implode_to_string<Eigen_scalar>(',', {
		origin[0],
		origin[1],
		origin[2],
		extremity[0],
		extremity[1],
		extremity[2]
	});
}


bounding_box bounding_box::from_string(const std::string& str) {
	std::vector<Eigen_scalar> p = explode_from_string<Eigen_scalar>(',', str);
	if(p.size() != 6)
		throw std::invalid_argument("invalid string to convert to bounding box");
	Eigen_vec3 o(p[0], p[1], p[2]);	
	Eigen_vec3 e(p[3], p[4], p[5]);
	return bounding_box(o, e);	
}



std::ostream& operator<<(std::ostream& str, const bounding_box& c) {
	str << "[ (" << c.origin[0] << ", " << c.origin[1] << ", " << c.origin[2] << ");"
		   " (" << c.extremity[0] << ", " << c.extremity[1] << ", " << c.extremity[2] << ") [";
	return str;
}


}
