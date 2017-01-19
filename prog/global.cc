#include "global.h"

namespace mf {

namespace {
	const std::size_t image_count = 39;
	const ndsize<2> image_shape = make_ndsize(864, 486);
	const ndsize<3> imc_shp = ndcoord_cat(image_count, image_shape);
}


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