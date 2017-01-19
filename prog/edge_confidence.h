#ifndef EPI_EDGE_CONFIDENCE_H_
#define EPI_EDGE_CONFIDENCE_H_

#include "opencv.h"
#include "nd.h"
#include "color.h"
#include <cstddef>

namespace mf {

ndarray<2, real> edge_confidence(const ndarray_view<2, rgb_color>& epi, std::ptrdiff_t radius);
ndarray<2, uchar> edge_confidence_mask(const ndarray_view<2, real>& conf, real threshold);

}

#endif