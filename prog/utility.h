#ifndef EPI_UTILITY_H_
#define EPI_UTILITY_H_

#include "common.h"
#include "nd.h"
#include "color.h"

#include <iostream>
#include <string>


namespace mf {

void reals_export(const ndarray_view<2, real>& vw, const std::string& filename, bool zero = false);
void reals_export(const ndarray_view<1, real>& vw, const std::string& filename, bool zero = false);

inline real color_diff(const rgb_color& a, const rgb_color& b) {
	// difference measure such that color_difference(a,b) == color_difference(b,a)
	int sum = 0;
	sum += (a.r <= b.r ? b.r - a.r : a.r - b.r);
	sum += (a.g <= b.g ? b.g - a.g : a.g - b.g);
	sum += (a.b <= b.b ? b.b - a.b : a.b - b.b);
	real diff = real(sum) / (3 * 256);
	Assert_crit(diff >= 0.0 && diff <= 1.0);
	return diff;
}

}

#endif