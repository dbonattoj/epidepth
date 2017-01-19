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

#include "spherical_coordinates.h"
#include "math_constants.h"
#include <ostream>
#include <cmath>

namespace mf {

spherical_coordinates spherical_coordinates::from_cartesian(const Eigen_vec3& c) {
	Eigen_scalar r = c.norm();
	return spherical_coordinates(
		r,
		std::atan2(c[0], -c[2]),
		half_pi - std::acos(c[1] / r)
	);
}


Eigen_vec3 spherical_coordinates::to_cartesian() const {
	return Eigen_vec3(
		std::cos(azimuth) * std::sin(elevation),
		std::cos(elevation),
		-std::sin(azimuth) * std::sin(elevation)
	) * radius;
}


bool spherical_coordinates::operator==(const spherical_coordinates& s) const {
	if(valid() && s.valid())
		return (radius == s.radius) && (azimuth == s.azimuth) && (elevation == s.elevation);
	else
		return (valid() == s.valid());
}


bool spherical_coordinates::operator!=(const spherical_coordinates& s) const {
	if(valid() && s.valid())
		return (radius != s.radius) || (azimuth != s.azimuth) || (elevation != s.elevation);
	else
		return (valid() != s.valid());
}


std::ostream& operator<<(std::ostream& str, const spherical_coordinates& c) {
	str << '(' << c.radius << ", " << c.azimuth << ", " << c.elevation << ')';
	return str;
}


}
