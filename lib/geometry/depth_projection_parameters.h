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

#ifndef MF_DEPTH_PROJECTION_PARAMETERS_H_
#define MF_DEPTH_PROJECTION_PARAMETERS_H_

#include "../common.h"

namespace mf {

/// Parameters of depth projection for projection camera.
/** Defines bijective mapping between orthogonal distance `z` of point (e.g. Z coordinate in camera view space), and
 ** projected depth `d`, of the form: `d = offset + factor/z`, or `d = -(offset - factor/z)` if `flip_z` is set.
 ** `offset` and `factor` are constants, and set such that `depth(z_near) == d_near` and `depth(z_far) == d_far`. */
struct depth_projection_parameters {
	using depth_type = real;
	using orthogonal_distance_type = real;
	
	depth_type d_near = 0.0; ///< Depth value to which z_near should be mapped.
	depth_type d_far = 1.0; ///< Depth value to which z_far should be mapped.

	orthogonal_distance_type z_near = 0.0; ///< Unsigned orthogonal distance of near clipping plane to camera.
	orthogonal_distance_type z_far = 0.0;	///< Unsigned orthogonal distance of far clipping plane to camera.

	bool flip_z = true; ///< Whether Z axis needs to be reversed, i.e. -Z direction maps to positive depth.
	
	/////
	
	bool valid() const;
		
	real offset() const;
	real factor() const;
	
	real depth(orthogonal_distance_type) const;
	real orthogonal_distance(depth_type) const;
	
	/////
	
	static depth_projection_parameters unsigned_normalized_disparity
		(orthogonal_distance_type z_near, orthogonal_distance_type z_far);
};

}

#endif
