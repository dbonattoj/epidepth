#ifndef MF_DEPTH_IMAGE_MAPPING_H_
#define MF_DEPTH_IMAGE_MAPPING_H_

#include "../common.h"
#include <limits>
#include <type_traits>

namespace mf {

template<typename Pixel_depth>
class depth_image_direct_mapping {
public:
	using pixel_depth_type = Pixel_depth;
	using depth_type = real;

public:
	bool is_bounded() const { return ! std::numeric_limits<Pixel_depth>::has_infinity(); }
	pixel_depth_type pixel_depth_minimum() const { return std::numeric_limits<Pixel_depth>::min(); }
	pixel_depth_type pixel_depth_maximum() const { return std::numeric_limits<Pixel_depth>::max(); }
	
	bool in_depth_bounds(depth_type) const;
	pixel_depth_type to_pixel_depth(depth_type) const;
	pixel_depth_type to_pixel_depth_clamp(depth_type) const;
	depth_type to_depth(pixel_depth_type) const;
};


template<typename Pixel_depth>
class depth_image_linear_mapping {
public:
	using pixel_depth_type = Pixel_depth;
	using depth_type = real;

private:
	depth_type depth_origin_;
	depth_type depth_range_;
	depth_type pixel_depth_minimum_;
	depth_type pixel_depth_maximum_;
	depth_type pixel_depth_range_;

public:
	depth_image_linear_mapping();
	depth_image_linear_mapping(depth_type origin, depth_type range, pixel_depth_type pix_min, pixel_depth_type pix_max);
	depth_image_linear_mapping(depth_type origin, depth_type range);	
	
	bool is_bounded() const { return true; }
	pixel_depth_type pixel_depth_minimum() const { return static_cast<pixel_depth_type>(pixel_depth_minimum_); }
	pixel_depth_type pixel_depth_maximum() const { return static_cast<pixel_depth_type>(pixel_depth_maximum_); }
	
	bool in_depth_bounds(depth_type) const;
	pixel_depth_type to_pixel_depth(depth_type) const;
	pixel_depth_type to_pixel_depth_clamp(depth_type) const;
	depth_type to_depth(pixel_depth_type) const;
};

}

#include "depth_image_mapping.tcc"

#endif
