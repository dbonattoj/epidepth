#ifndef EPI_UTILITY_H_
#define EPI_UTILITY_H_

#include "common.h"
#include "nd.h"
#include "color.h"
#include <string>

namespace mf {

void reals_export(const ndarray_view<2, real>& vw, const std::string& filename);
void reals_export(const ndarray_view<1, real>& vw, const std::string& filename);

inline real color_diff(const rgb_color& a, const rgb_color& b) {
	// difference measure such that color_difference(a,b) == color_difference(b,a)
	real diff = 0.0;
	diff += (a.r <= b.r ? b.r - a.r : a.r - b.r);
	diff += (a.g <= b.g ? b.g - a.g : a.g - b.g);
	diff += (a.b <= b.b ? b.b - a.b : a.b - b.b);
	return diff;
}

}

#endif