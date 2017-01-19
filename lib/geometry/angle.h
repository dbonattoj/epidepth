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

#ifndef MF_ANGLE_H_
#define MF_ANGLE_H_

#include <iosfwd>
#include <cmath>
#include "../common.h"
#include "math_constants.h"

namespace mf {

/// Angle, represented in radiants.
/** Defines I/O in degrees, and conversion. Provides unambiguity in what angles mean. */
struct angle {
private:
	real radiants_;

public:
	static angle degrees(real d) { return d * radiant_per_degree; }
	static angle radiants(real r) { return r; }

	angle() = default;
	angle(real r) : radiants_(r) { }
	angle(const angle&) = default;
	
	const real& get_radiants() const { return radiants_; }
	real get_degrees() const { return radiants_ * degree_per_radiant; }
	void set_radiants(real r) { radiants_ = r; }
	void set_degrees(real d) { radiants_ = d * radiant_per_degree; }
	
	operator real& () { return radiants_; }
	operator const real& () const { return radiants_; }
	
	angle& operator=(real r) { radiants_ = r; return *this; }
	angle& operator=(const angle&) = default;
	
	bool operator==(angle a) const { return radiants_ == a.radiants_; }
	bool operator!=(angle a) const { return radiants_ != a.radiants_; }
	
	static angle smallest_between(angle a, angle b);
};

inline angle operator"" _deg(long double deg) {
	return angle::degrees(static_cast<real>(deg));
}

inline angle operator"" _rad(long double rad) {
	return angle::radiants(static_cast<real>(rad));
}

std::ostream& operator<<(std::ostream&, angle);
std::istream& operator>>(std::istream&, angle&);

}

#endif
