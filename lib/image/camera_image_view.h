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

#ifndef MF_CAMERA_IMAGE_VIEW_H_
#define MF_CAMERA_IMAGE_VIEW_H_

#include "image_view.h"
#include "../camera/projection_image_camera.h"
#include <utility>

namespace mf {

template<typename Pixel>
class camera_image_view : public masked_image_view<Pixel> {
	using base = masked_image_view<Pixel>;

private:
	projection_image_camera camera_;

public:
	camera_image_view(const base&, const projection_camera&);
	
	const projection_image_camera& this_camera() const { return camera_; }
	projection_image_camera& this_camera() { return camera_; }
};

}

#include "camera_image_view.tcc"

#endif
