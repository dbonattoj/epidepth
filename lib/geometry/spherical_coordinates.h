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

#ifndef MF_SPHERICAL_COORDINATES_H_
#define MF_SPHERICAL_COORDINATES_H_

#include "angle.h"
#include <iosfwd>
#include "../eigen.h"

namespace mf {

/// Spherical coordinates.
/** Let O be origin, P be point. Radius is distance OP.
 ** Elevation is (polar) angle between OP and reference plane Y=0.
 ** Or equivalently angle between OP and zenith axis +Y, plus pi.
 ** Azimuth is angle between -Z and the projection of OP on Y=0.
 ** Both azimuth and elevation are always in [-pi, pi]. */
struct spherical_coordinates {
	Eigen_scalar radius = 0.0;
	angle azimuth = 0.0;
	angle elevation = 0.0;

	spherical_coordinates() = default;
	spherical_coordinates(const spherical_coordinates&) = default;
	spherical_coordinates(Eigen_scalar r, angle az, angle el) :
		radius(r), azimuth(az), elevation(el) { }
	
	static spherical_coordinates from_cartesian(const Eigen_vec3&);
	Eigen_vec3 to_cartesian() const;
	
	bool operator==(const spherical_coordinates&) const;
	bool operator!=(const spherical_coordinates&) const;	
	
	void invalidate() { radius = 0; }
	bool valid() const { return (radius != 0); }
	explicit operator bool () const { return valid(); }
};


std::ostream& operator<<(std::ostream&, const spherical_coordinates&);

}

#endif
