#include "../utility/misc.h"

namespace mf {

template<typename Pixel_depth>
inline bool depth_image_direct_mapping<Pixel_depth>::in_depth_bounds(depth_type d) const {
	if(is_bounded()) {
		static const depth_type pix_d_min = static_cast<depth_type>(pixel_depth_minimum());
		static const depth_type pix_d_max = static_cast<depth_type>(pixel_depth_maximum());
		depth_type pix_d = d;
		return (pix_d >= pix_d_min) && (pix_d <= pix_d_max);
	} else {
		return true;
	}
}


template<typename Pixel_depth>
inline auto depth_image_direct_mapping<Pixel_depth>::to_pixel_depth(depth_type d) const -> pixel_depth_type {
	return static_cast<pixel_depth_type>(d);
}


template<typename Pixel_depth>
inline auto depth_image_direct_mapping<Pixel_depth>::to_pixel_depth_clamp(depth_type d) const -> pixel_depth_type {
	if(is_bounded()) {
		static const depth_type pix_d_min = static_cast<depth_type>(pixel_depth_minimum());
		static const depth_type pix_d_max = static_cast<depth_type>(pixel_depth_maximum());
		depth_type pix_d = d;
		depth_type pix_d_clamped = clamp(pix_d, pix_d_min, pix_d_max);
		return to_pixel_depth(pix_d_clamped);
	} else {
		return to_pixel_depth(d);
	}
}


template<typename Pixel_depth>
inline auto depth_image_direct_mapping<Pixel_depth>::to_depth(pixel_depth_type pix_d) const -> depth_type {
	return static_cast<depth_type>(pix_d);
}


///////////////


template<typename Pixel_depth>
depth_image_linear_mapping<Pixel_depth>::depth_image_linear_mapping
(depth_type origin, depth_type range, pixel_depth_type pix_min, pixel_depth_type pix_max) :
	depth_origin_(origin),
	depth_range_(range),
	pixel_depth_minimum_(static_cast<depth_type>(pix_min)),
	pixel_depth_maximum_(static_cast<depth_type>(pix_max)),
	pixel_depth_range_(pixel_depth_maximum_ - pixel_depth_minimum_) { }


template<typename Pixel_depth>
depth_image_linear_mapping<Pixel_depth>::depth_image_linear_mapping(depth_type origin, depth_type range) :
	depth_image_linear_mapping(
		origin,
		range,
		std::numeric_limits<Pixel_depth>::min(),
		std::numeric_limits<Pixel_depth>::max()
	) { }
	

template<typename Pixel_depth>
depth_image_linear_mapping<Pixel_depth>::depth_image_linear_mapping() :
	depth_image_linear_mapping(0, 1) { }	


	

template<typename Pixel_depth>
bool depth_image_linear_mapping<Pixel_depth>::in_depth_bounds(depth_type d) const {
	depth_type pix_d = (d - depth_origin_) * pixel_depth_range_ / depth_range_;
	return (pix_d >= pixel_depth_minimum_) && (pix_d <= pixel_depth_maximum_);
}


template<typename Pixel_depth>
auto depth_image_linear_mapping<Pixel_depth>::to_pixel_depth(depth_type d) const -> pixel_depth_type {
	depth_type pix_d = (d - depth_origin_) * pixel_depth_range_ / depth_range_;
	return static_cast<pixel_depth_type>(pix_d);
}


template<typename Pixel_depth>
auto depth_image_linear_mapping<Pixel_depth>::to_pixel_depth_clamp(depth_type d) const -> pixel_depth_type {
	depth_type pix_d = (d - depth_origin_) * pixel_depth_range_ / depth_range_;
	depth_type pix_d_clamped = clamp(pix_d, pixel_depth_minimum_, pixel_depth_maximum_);
	return static_cast<pixel_depth_type>(pix_d);
}


template<typename Pixel_depth>
auto depth_image_linear_mapping<Pixel_depth>::to_depth(pixel_depth_type pix_d_) const -> depth_type {
	depth_type pix_d = static_cast<depth_type>(pix_d_);
	return (pix_d * depth_range_ / pixel_depth_range_) + depth_origin_;
}


}
