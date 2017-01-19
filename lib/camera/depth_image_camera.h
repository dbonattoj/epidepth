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

#ifndef MF_DEPTH_IMAGE_CAMERA_H_
#define MF_DEPTH_IMAGE_CAMERA_H_

#include "image_camera.h"
#include "depth_image_mapping.h"

namespace mf {

/// Depth camera which handles mapping to integer depth values, base class.
template<typename Depth_mapping>
class depth_image_camera : public image_camera {
public:
	using depth_mapping_type = Depth_mapping;
	using pixel_depth_type = typename depth_mapping_type::pixel_depth_type;
	using depth_type = real;

private:
	depth_mapping_type mapping_;

public:
	depth_image_camera() = default;
	depth_image_camera(const ndsize<2>& image_size, const depth_mapping_type& mapping) :
		image_camera(image_size), mapping_(mapping) { }
	
	/// Verifies whether \a d is in bounds of valid depth values.
	bool in_depth_bounds(depth_type d) const {
		return mapping_.in_depth_bounds(d);
	}
	
	/// Map depth value to pixel depth value.
	/** \a d must be in bounds, i.e. `in_depth_bounds(d)` must be true. */
	pixel_depth_type to_pixel_depth(depth_type d) const {
		return mapping_.to_pixel_depth(d);
	}
	
	/// Map depth value to pixel depth value, and clamp to bounds.
	/** If \a d is not in bounds, result is clamped between minimum and maximum pixel depth. */
	pixel_depth_type to_pixel_depth_clamp(depth_type d) const {
		return mapping_.to_pixel_depth_clamp(d);
	}
	
	/// Map pixel depth value to depth value in range defined by depth projection parameter.
	/** Values outside pixel depth range are mapped to values outside real depth range. */
	depth_type to_depth(pixel_depth_type pix_d) const {
		return mapping_.to_depth(pix_d);
	}
};


}

#include "depth_image_camera.tcc"

#endif
