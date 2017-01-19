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

namespace mf {

template<typename Depth_mapping>
projection_image_camera<Depth_mapping>::projection_image_camera
(const projection_camera& cam, const ndsize<2>& image_size, const depth_mapping_type& mapping) :
	projection_camera(cam),
	depth_image_camera_base(image_size, mapping) { }


template<typename Depth_mapping>
void projection_image_camera<Depth_mapping>::scale(real factor) {
	const ndsize<2>& image_size = depth_image_camera_base::image_size();
	ndsize<2> scaled_image_size = make_ndsize(factor * image_size[0], factor * image_size[1]);

	projection_camera::scale(factor);
	depth_image_camera_base::set_image_size_(scaled_image_size);
}


template<typename Depth_mapping>
void projection_image_camera<Depth_mapping>::set_image_width(std::size_t imw) {
	const ndsize<2>& image_size = depth_image_camera_base::image_size();
	real factor = static_cast<real>(imw) / image_size[0];
	ndsize<2> scaled_image_size = image_camera::scaled_image_size(image_size, factor);


	projection_camera::scale(factor);
	depth_image_camera_base::set_image_size_(scaled_image_size);
}


template<typename Depth_mapping>
void projection_image_camera<Depth_mapping>::set_image_height(std::size_t imh) {
	const ndsize<2>& image_size = depth_image_camera_base::image_size();
	real factor = static_cast<real>(imh) / image_size[1];
	ndsize<2> scaled_image_size = image_camera::scaled_image_size(image_size, factor);
	
	projection_camera::scale(factor);
	depth_image_camera_base::set_image_size_(scaled_image_size);
}


}
