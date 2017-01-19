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

#ifndef MF_PROJECTION_IMAGE_CAMERA_H_
#define MF_PROJECTION_IMAGE_CAMERA_H_

#include "projection_camera.h"
#include "depth_image_camera.h"
#include "../nd.h"

namespace mf {

/// Pin-hole camera with mapping onto image pixel coordinates.
template<typename Depth_mapping>
class projection_image_camera : public projection_camera, public depth_image_camera<Depth_mapping> {
	using depth_image_camera_base = depth_image_camera<Depth_mapping>;
	using depth_mapping_type = typename depth_image_camera_base::depth_mapping_type;


public:
	projection_image_camera(const projection_camera&, const ndsize<2>&, const depth_mapping_type& = depth_mapping_type());
	projection_image_camera(const projection_image_camera&) = default;
	projection_image_camera& operator=(const projection_image_camera&) = default;
	
	void scale(real factor) override;
	void set_image_width(std::size_t) override;
	void set_image_height(std::size_t) override;
};

}

#include "projection_image_camera.tcc"

#endif
