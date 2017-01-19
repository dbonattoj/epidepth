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

#include "depth_projection_parameters.h"

namespace mf {

bool depth_projection_parameters::valid() const {
	if(z_near < 0.0 || z_far < 0.0) return false;
	if(z_far <= z_near) return false;
	return true;
}


real depth_projection_parameters::offset() const {	
	real dividend = (d_far * z_far) - (d_near * z_near);
	return dividend / (z_far - z_near);
}


real depth_projection_parameters::factor() const {
	real dividend = (d_near - d_far) * z_near * z_far;
	return dividend / (z_far - z_near);
}


real depth_projection_parameters::depth(orthogonal_distance_type z) const {
	if(flip_z) z = -z;
	return offset() + factor()/z;
}


real depth_projection_parameters::orthogonal_distance(depth_type d) const {
	real z = factor() / (d - offset());
	return (flip_z ? -z : z);
}


///////////////

depth_projection_parameters depth_projection_parameters::unsigned_normalized_disparity
(orthogonal_distance_type z_near, orthogonal_distance_type z_far) {
	depth_projection_parameters dparam;
	dparam.d_near = 1.0;
	dparam.d_far = 0.0;
	if(z_near > 0.0) {
		dparam.z_near = z_near;
		dparam.z_far = z_far;
		dparam.flip_z = false;
	} else {
		dparam.z_near = -z_near;
		dparam.z_far = -z_far;
		dparam.flip_z = true;
	}
	Assert(dparam.valid());
	
	return dparam;
}


}
