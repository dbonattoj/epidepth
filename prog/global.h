#ifndef EPI_GLOBAL_H_
#define EPI_GLOBAL_H_

#include "common.h"
#include "sparse_epi.h"

#include "nd.h"
#include "color.h"

#include <cstddef>
#include <vector>

namespace mf {

// configuration
extern const char* image_path_format;

constexpr std::ptrdiff_t edge_confidence_rad = 4;
constexpr real edge_confidence_mask_min_threshold = 0.02;
constexpr std::ptrdiff_t edge_confidence_mask_width = 2;

constexpr real minimal_disparity = 0.0;
constexpr real maximal_disparity = 4.0;
constexpr real disparity_steps = 50;

constexpr real depth_score_color_threshold = 0.2;

constexpr std::ptrdiff_t bilateral_window_rad = 8;
constexpr real bilateral_window_color_threshold = 1.0;
constexpr real bileratal_window_confidence_threshold = 0.0;

constexpr real confidence_threshold = 0.005;

constexpr real propagation_color_threshold = 1.0;


// dimensions
extern const std::ptrdiff_t final_u_sz;
extern const std::ptrdiff_t final_v_sz;
extern std::ptrdiff_t u_sz;
extern std::ptrdiff_t v_sz;
extern const std::ptrdiff_t s_sz;
extern ndsize<2> epi_shp;
extern ndsize<2> image_shp;

extern const std::ptrdiff_t final_u_sz;
extern const std::ptrdiff_t final_v_sz;

// input, working data
extern std::size_t scale_down_factor;
extern ndarray<3, rgb_color> epis; // [v][u][s]

// output
extern std::vector<sparse_epi> final_sparse_epis; // [v] (epi shape: [u][s]), original scale
extern ndarray<3, real> final_epi_disparities; // [v][u][s], original scale


}

#endif
