#include "global.h"

namespace mf {


/*
const char* image_path_format = "../data/2016-07-26-rectified-kinect-parallel/Output%d.jpg";
const std::size_t image_count = 40;
const ndsize<2> image_shape = make_ndsize(864, 486);
*/

/*
const char* image_path_format = "../data/couch_image-raw_scaled/couch_image-raw_%04d.jpg";
const std::size_t image_count = 50;
const ndsize<2> image_shape = make_ndsize(816, 544);
*/

const char* image_path_format = "../data/mansion/mansion_image_%04d.jpg";
const std::size_t image_count = 100;
const ndsize<2> image_shape = make_ndsize(769, 483);


const ndsize<3> imc_shp = ndcoord_cat(image_count, image_shape);
const std::ptrdiff_t final_u_sz = imc_shp[1];
const std::ptrdiff_t final_v_sz = imc_shp[2];
std::ptrdiff_t u_sz = final_u_sz;
std::ptrdiff_t v_sz = final_v_sz;
const std::ptrdiff_t s_sz = imc_shp[0];
ndsize<2> epi_shp = make_ndsize(u_sz, s_sz);
ndsize<2> image_shp = make_ndsize(u_sz, v_sz);

std::size_t scale_down_factor = 1;
ndarray<3, rgb_color> epis(make_ndsize(v_sz, u_sz, s_sz));
std::vector<sparse_epi> final_sparse_epis(epis.shape()[0], sparse_epi(epi_shp));
ndarray<3, real> final_epi_disparities(make_ndsize(v_sz, u_sz, s_sz));


}