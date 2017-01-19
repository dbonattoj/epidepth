#include "utility.h"

namespace mf {

void reals_export(const ndarray_view<2, real>& vw, const std::string& filename) {
	ndarray<2, real> arr(vw);
	real min_x = -1.0;
	for(real x : arr) if(! std::isnan(x)) if(min_x == -1.0 || x < min_x) min_x = x;
	//for(real x : arr) std::cout << x << std::endl;
	for(real& x : arr) if(std::isnan(x)) x = min_x;
	image_export(make_image_view(arr.view()), filename);
}


void reals_export(const ndarray_view<1, real>& vw, const std::string& filename) {
	const std::ptrdiff_t h = 100;
	ndarray<2, real> arr(make_ndsize(vw.shape()[0], h));
	for(std::ptrdiff_t y = 0; y < h; ++y) arr.slice(y, 1) = vw;
	reals_export(arr.view(), filename);
}

}
