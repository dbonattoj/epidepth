#include "utility.h"
#include "io/image_export.h"

namespace mf {

void reals_export(const ndarray_view<2, real>& vw, const std::string& filename, real max_value) {
	ndarray<2, real> arr(vw);
	for(real& x : arr) if(std::isnan(x)) x = 0.0;
	image_export(make_image_view(arr.view()), filename, 0.0, max_value);
}

void reals_export(const ndarray_view<2, real>& vw, const std::string& filename, bool zero) {
	ndarray<2, real> arr(vw);
	real min_x = -1.0;
	if(! zero) for(real x : arr) if(! std::isnan(x)) if(min_x == -1.0 || x < min_x) min_x = x;
	else min_x = 0.0;
	for(real& x : arr) if(std::isnan(x)) x = min_x;
	image_export(make_image_view(arr.view()), filename);
}


void reals_export(const ndarray_view<1, real>& vw, const std::string& filename, bool zero) {
	const std::ptrdiff_t h = 100;
	ndarray<2, real> arr(make_ndsize(vw.shape()[0], h));
	for(std::ptrdiff_t y = 0; y < h; ++y) arr.slice(y, 1) = vw;
	reals_export(arr.view(), filename);
}

void reals_export(const ndarray_view<1, real>& vw, const std::string& filename, real max_value) {
	const std::ptrdiff_t h = 100;
	ndarray<2, real> arr(make_ndsize(vw.shape()[0], h));
	for(std::ptrdiff_t y = 0; y < h; ++y) arr.slice(y, 1) = vw;
	reals_export(arr.view(), filename, max_value);
}

}
