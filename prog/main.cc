#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <shared_mutex>

#include "nd.h"
#include "image/image.h"
#include "io/image_import.h"
#include "io/image_export.h"
#include "opencv.h"
#include "utility/misc.h"

#include "utility.h"
#include "edge_confidence.h"
#include "sparse_epi.h"
#include "disparity_estimation.h"

using namespace mf;


const std::size_t image_count = 39;
const auto image_shape = make_ndsize(864, 486);
const ndsize<3> imc_shp = ndcoord_cat(image_count, image_shape); // [s][u][v]

const std::ptrdiff_t s_sz = imc_shp[0];
const std::ptrdiff_t u_sz = imc_shp[1];
const std::ptrdiff_t v_sz = imc_shp[2];
const ndsize<2> epi_shp = make_ndsize(u_sz, s_sz);
const ndsize<2> image_shp = make_ndsize(u_sz, v_sz);

const std::ptrdiff_t bilateral_window_rad = 5;
const real segment_confidence_threshold = 0.0;


ndarray<3, rgb_color> epis(make_ndsize(v_sz, u_sz, s_sz)); // [v][u][s]
std::vector<sparse_epi> sparse_epis(epis.shape()[0], sparse_epi(epi_shp)); // [v] (epi shape: [u][s])



real epi_disparity_median(
	ndarray_view<2, rgb_color> colors, // [v][u]
	ndarray_view<2, real> disparities, // [v][u]
	ndarray_view<2, real> confidences, // [v][u]
	std::ptrdiff_t u, std::ptrdiff_t v,
	std::ptrdiff_t rad,
	real color_diff_threshold,
	real confidence_threshold
) {
	Assert_crit(! std::isnan(disparities[v][u]));
	
	std::vector<real> median_disparities;
	median_disparities.reserve(sq(2*rad + 1));
	
	const ndsize<2>& shp = colors.shape();
	std::ptrdiff_t u2_begin = u > rad ? u - rad : 0;
	std::ptrdiff_t v2_begin = v > rad ? v - rad : 0;
	std::ptrdiff_t u2_end = u < shp[1]-rad ? u + rad : shp[1];
	std::ptrdiff_t v2_end = v < shp[0]-rad ? v + rad : shp[0];
	for(std::ptrdiff_t u2 = u2_begin; u2 < u2_end; ++u2) {
		for(std::ptrdiff_t v2 = v2_begin; v2 < v2_end; ++v2) {
			real d = disparities[v2][u2];
			if(confidences[v2][u2] <= confidence_threshold) continue;
			Assert_crit(! std::isnan(d));
			if(color_diff(colors[v][u], colors[v2][u2]) > color_diff_threshold) continue;
			
			median_disparities.push_back(d);
		}
	}
	
	if(median_disparities.size() == 0) {
		return disparities[v][u];
	} else {
		std::ptrdiff_t pos = median_disparities.size() / 2;
		std::nth_element(median_disparities.begin(), median_disparities.begin() + pos, median_disparities.end());
		real d = median_disparities[pos];
		Assert_crit(! std::isnan(d));
		return d;
	}
}


image<rgb_color> load_image(std::ptrdiff_t s) {
	const std::string images_dir = "../data/2016-07-26-rectified-kinect-parallel/";

	std::ostringstream str;
	str << images_dir << "Output" << s << ".jpg";
	//str << images_dir << "Kinect_color_" << std::setw(6) << std::setfill('0') << i << "_000.png";
	std::string filename = str.str();

	image<rgb_color> orig_img = mf::image_import(filename);
	image<rgb_color> img(flip(image_shape));
	cv::resize(orig_img.cv_mat(), img.cv_mat(), cv::Size(image_shape[0], image_shape[1]), 0.0, 0.0, cv::INTER_LINEAR);

	return img;
}



void process_scale() {
	// compute edge confidence for all epis
	std::cout << "computing edge confidence for all epi..." << std::endl;
	ndarray<3, real> edge_confidences(epis.shape()); // [v][u][s]
	ndarray<3, uchar> edge_confidence_masks(epis.shape()); // [v][u][s]
	
	#pragma omp parallel for
	for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
		std::cout << '.' << std::flush;
		edge_confidences[v] = edge_confidence(epis[v], 8);
		edge_confidence_masks[v] = edge_confidence_mask(edge_confidences[v], 7e3);
	}
	std::cout << std::endl;
	
	
	// all estimated disparity+confidence values
	// computed lazily
	ndarray<3, real> disparities(epis.shape()); // [v][u][s]
	std::fill(disparities.begin(), disparities.end(), NAN);
	ndarray<3, real> confidences(epis.shape()); // [v][u][s]
	std::fill(confidences.begin(), confidences.end(), NAN);
	ndarray<2, uchar> epis_loaded_lines(make_ndsize(v_sz, s_sz)); // [v][s]
	std::fill(epis_loaded_lines.begin(), epis_loaded_lines.end(), 0);

	std::shared_timed_mutex epi_loaded_lines_mutex;
	
	
	auto process_epi_line = [&](std::ptrdiff_t v, std::ptrdiff_t s) {
		if(epis_loaded_lines[v][s]) return;
		//std::cout << "estimating disparity for epi v=" << v << " line s=" << s << std::endl;
		epi_line_disparity_result res = estimate_epi_line_disparity(
			s,
			epis[v],
			edge_confidences[v].slice(s, 1),
			edge_confidence_masks[v].slice(s, 1)
		);
		std::unique_lock<std::shared_timed_mutex> lock(epi_loaded_lines_mutex);
		disparities[v].slice(s, 1) = res.disparity;
		confidences[v].slice(s, 1) = res.confidence;
		epis_loaded_lines[v][s] = true;
	};
	
	std::vector<sparse_epi> sparse_epis(epis.shape()[0], sparse_epi(epi_shp)); // [v] (epi shape: [u][s])
	
	
	std::cout << "processing epis..." << std::endl;
	
	#pragma omp parallel for
	for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
		std::cout << "epi v=" << v << std::endl;
		ndarray_view<2, rgb_color> epi = epis[v];
		
		ndarray<2, real> new_disparities(epi_shp); // [u][s]
		std::fill(new_disparities.begin(), new_disparities.end(), NAN);
		
		const std::ptrdiff_t mid_s = s_sz / 2;
		std::ptrdiff_t s = mid_s;

		std::vector<std::ptrdiff_t> line_remaining_holes(s_sz);
		
		for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
			line_remaining_holes[s] = u_sz;
			for(std::ptrdiff_t u = 0; u < u_sz; ++u)
				if(confidences[v][u][s] <= segment_confidence_threshold) line_remaining_holes[s]--;
		}
		// holes = pixels that still need to be filled
		// masked pixels (low confidence) don't count as holes
		// line_remaining_holes[s] will become < 0 when masked pixels on other lines get filled by depth propagation
		
		
		bool finished = false;
		while(! finished) {
			std::cout << "   line s=" << s << std::endl;
			
			for(std::ptrdiff_t v2 = std::max<std::ptrdiff_t>(v - bilateral_window_rad, 0); v2 < std::min<std::ptrdiff_t>(v + bilateral_window_rad, v_sz); ++v2)
				process_epi_line(v2, s);
			
			for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
				real d;
				{
					std::shared_lock<std::shared_timed_mutex> lock(epi_loaded_lines_mutex);
					if(confidences[v][u][s] <= segment_confidence_threshold) continue;
					Assert_crit(! std::isnan(disparities[v][u][s])); // if NAN only if confidences == 0.0
					d = epi_disparity_median(
						epis.slice(s, 2), disparities.slice(s, 2), confidences.slice(s, 2),
						u, v,
						bilateral_window_rad, 100.0, 100.0
					);
				}
				
				Assert_crit(! std::isnan(d));
				new_disparities[u][s] = d;
				
				rgb_color avg_color = epis[v][u][s];
				
				auto propagate = [&](std::ptrdiff_t s2) {
					std::ptrdiff_t u2 = u  + (s - s2)*d;
					if(u2 < 0 || u2 >= u_sz) return false;
					if(1||color_diff(epi[u2][s2], avg_color) < 100.0) {
						if(std::isnan(new_disparities[u2][s2])) {
							line_remaining_holes[s2]--;
							new_disparities[u2][s2] = d;
						} else if(new_disparities[u2][s2] < d) {
							new_disparities[u2][s2] = d;
						}
						return true;
					} else {
						return false;
					}
				};
				
				for(std::ptrdiff_t s2 = s + 1; s2 < s_sz; ++s2) if(! propagate(s2)) break;
				for(std::ptrdiff_t s2 = s - 1; s2 >= 0; --s2) if(! propagate(s2)) break;
				
				sparse_epi_segment seg { d, u, s, avg_color };
				sparse_epis[v].add_segment(seg);
			}
			
			line_remaining_holes[s] = 0;
			
			std::ptrdiff_t lower_s, higher_s;
			for(lower_s = mid_s - 1; lower_s >= 0; lower_s--) if(line_remaining_holes[lower_s] > 0) break;
			for(higher_s = mid_s + 1; higher_s < s_sz; higher_s++) if(line_remaining_holes[higher_s] > 0) break;
			if(lower_s == -1 && higher_s == s_sz) finished = true;
			else if(lower_s == -1) s = higher_s;
			else if(higher_s == s_sz) s = lower_s;
			else if(higher_s - s > s - lower_s) s = lower_s;
			else s = higher_s;
			
			Assert_crit(finished || line_remaining_holes[s] > 0);
		}
		
		/*
		reals_export(new_disparities.view(), "../output/new_disparities_" + std::to_string(v) + ".png");
		*/
		
		ndarray<2, rgb_color> re_epi = sparse_epis[v].reconstruct();
		image_export(make_image_view(re_epi.view()), "../output/re_epi" + std::to_string(v) + ".png");

		/*
		real min_d = -1.0;
		for(real d : new_disparities) if(d != NAN) if(min_d == -1.0 || d < min_d) min_d = d;
		for(real& d : new_disparities) if(d == NAN) d = min_d;
		std::string filename = "../output/new_disparities_" + std::to_string(v) + ".png";
		image_export(make_image_view(new_disparities.view()), filename);
		*/

	}

}


int main() {
	// load all images/epis in memory
	std::cout << "loading epis..." << std::endl;
	for(int s = 0; s < s_sz; ++s) epis.slice(s, 2) = load_image(s).array_view();

	
	

}
