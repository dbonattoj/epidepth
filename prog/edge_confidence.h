#ifndef EPI_EDGE_CONFIDENCE_H_
#define EPI_EDGE_CONFIDENCE_H_

#include "nd.h"
#include "color.h"
#include <cstddef>
#include <cstdint>

namespace mf {

ndarray<2, real> edge_confidence(const ndarray_view<2, rgb_color>& epi);
ndarray<2, std::uint8_t> edge_confidence_mask(const ndarray_view<2, real>& conf);

}

#endif