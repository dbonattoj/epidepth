#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <atomic>

#include "nd.h"
#include "image/image.h"
#include "io/image_import.h"
#include "io/image_export.h"
#include "opencv.h"
#include "utility/misc.h"

#include "global.h"
#include "utility.h"
#include "edge_confidence.h"
#include "sparse_epi.h"
#include "disparity_estimation.h"

using namespace mf;


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
	image<rgb_color> img(flip(image_shp));
	cv::resize(orig_img.cv_mat(), img.cv_mat(), cv::Size(image_shp[0], image_shp[1]), 0.0, 0.0, cv::INTER_LINEAR);

	return img;
}



void scale_down(
	const ndarray_view<3, real>& previous_epi_disparities, // [v][u][s] with previous size
	const ndarray_view<3, real>& output_scaled_epi_disparities // [v][u][s] with new (smaller) size
) {
	// scale down in u and v directions by factor 2 (downsample images)
	
	std::ptrdiff_t scaled_u_sz = u_sz / 2;
	std::ptrdiff_t scaled_v_sz = v_sz / 2;
	
	ndarray<3, rgb_color> scaled_epis(make_ndsize(scaled_v_sz, scaled_u_sz, s_sz));
	
	// downsample images
	#pragma omp parallel for
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray<2, rgb_color> img(epis.slice(s, 2));
		
		cv::Mat_<rgb_color> img_cv = to_opencv(img.view());
		cv::GaussianBlur(img_cv, img_cv, cv::Size(15, 15), 0.0, 0.0, cv::BORDER_DEFAULT);
		
		cv::Mat_<rgb_color> scaled_img_cv;
		cv::resize(img_cv, scaled_img_cv, cv::Size(scaled_u_sz, scaled_v_sz), 0.0, 0.0, cv::INTER_LINEAR);
		
		ndarray_view<2, rgb_color> scaled_img = to_ndarray_view<2, rgb_color>(scaled_img_cv);
		scaled_epis.slice(s, 2) = scaled_img;
		
		image_export(make_image_view(scaled_img), "../output/im_"+std::to_string(s)+"b.png");
	}
	
	// update global values, and move in new epis cube
	scale_down_factor *= 2;
	u_sz = scaled_u_sz;
	v_sz = scaled_v_sz;
	epis = std::move(scaled_epis);
	epi_shp = make_ndsize(u_sz, s_sz);
	image_shp = make_ndsize(u_sz, v_sz);
}


void scale_down_epi_disparities(
	const ndarray_view<3, real>& previous_epi_disparities, // [v][u][s] with previous size
	const ndarray_view<3, real>& output_scaled_epi_disparities // [v][u][s] with new (smaller) size
) {
	// downsample the estimated disparities
	#pragma omp parallel for
	for(std::ptrdiff_t v = 0; v < output_scaled_epi_disparities.shape()[0]; ++v)
		output_scaled_epi_disparities[v] = step(previous_epi_disparities[2*v], 0, 2);
}


void compute_min_max_disparities(
	const ndarray_view<2, real>& existing_disparities,
	const ndarray_view<2, real>& output_min_disparities,
	const ndarray_view<2, real>& output_max_disparities
) {
	std::fill(output_min_disparities.begin(), output_min_disparities.end(), NAN);
	std::fill(output_max_disparities.begin(), output_max_disparities.end(), NAN);
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray_view<1, real> existing = existing_disparities.slice(s, 1);

		ndarray<1, real> closest_left(make_ndsize(u_sz)), closest_right(make_ndsize(u_sz));
		
		real left_d = NAN;
		for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
			if(! std::isnan(existing[u])) left_d = existing[u];
			else closest_left[u] = left_d;
		}
		
		real right_d = NAN;
		for(std::ptrdiff_t u = u_sz-1; u >= 0; --u) {
			if(! std::isnan(existing[u])) right_d = existing[u];
			else closest_right[u] = right_d;
		}
		
		for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
			if(! std::isnan(existing[u])) continue;
			real l = closest_left[u], r = closest_right[u];
			real out_min, out_max;
			if(std::isnan(l) && std::isnan(r)) {
				out_min = minimal_disparity;
				out_max = maximal_disparity;
			} else if(! std::isnan(l)) {
				out_min = minimal_disparity;
				out_max = l;
			} else if(! std::isnan(r)) {
				out_min = minimal_disparity;
				out_max = r;
			} else {
				out_min = std::min(l, r);
				out_max = std::max(l, r);
			}
			output_min_disparities[u][s] = out_min;
			output_max_disparities[u][s] = out_max;
		}
	}
}


void process_scale(
	const ndarray_view<3, real>& output_epi_disparities // [v][u][s]
) {
	// compute edge confidence, and min+max disparities for all epis
	std::cout << "computing edge confidence, min+max disparities for all epi..." << std::endl;
	ndarray<3, real> epi_edge_confidences(epis.shape()); // [v][u][s]
	ndarray<3, uchar> epi_edge_confidence_masks(epis.shape()); // [v][u][s]
	
	ndarray<3, real> epi_min_disparities(epis.shape());
	ndarray<3, real> epi_max_disparities(epis.shape());
	
	#pragma omp parallel for
	for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
		std::cout << '.' << std::flush;
		epi_edge_confidences[v] = edge_confidence(epis[v], 8);
		epi_edge_confidence_masks[v] = edge_confidence_mask(epi_edge_confidences[v], 7e3);
		
		compute_min_max_disparities(output_epi_disparities[v], epi_min_disparities[v], epi_max_disparities[v]);
	}
	std::cout << std::endl;
	
	
	// estimated disparities+confidences values for all epis
	// computed on demand, for epi lines
	// for pixels rejected by edge confidence mask: epi_confidences = 0, and epi_disparities = NAN
	// so: even if segment_confidence_threshold is set to 0, pixels with epi_confidences==0 still need to be rejected
	//     because no estimated disparity was calculated for them
	ndarray<3, real> epi_disparities(epis.shape()); // [v][u][s]
	ndarray<3, real> epi_confidences(epis.shape()); // [v][u][s]
	ndarray<2, std::atomic<bool>> epi_loaded_lines(make_ndsize(v_sz, s_sz)); // [v][s]
	std::shared_timed_mutex epi_loaded_lines_mutex;
	std::fill(epi_disparities.begin(), epi_disparities.end(), NAN);
	std::fill(epi_confidences.begin(), epi_confidences.end(), NAN);
	std::fill(epi_loaded_lines.begin(), epi_loaded_lines.end(), false);
	
	auto process_epi_line = [&](std::ptrdiff_t v, std::ptrdiff_t s) {
		// estimate disparities+confidences for epi line (v,s)
		if(epi_loaded_lines[v][s].load()) return;
		
		
		epi_line_disparity_result res = estimate_epi_line_disparity(
			s,
			epis[v],
			epi_edge_confidences[v].slice(s, 1),
			epi_edge_confidence_masks[v].slice(s, 1),
			epi_min_disparities[v].slice(s, 1),
			epi_max_disparities[v].slice(s, 1)
		);
		
		std::unique_lock<std::shared_timed_mutex> lock(epi_loaded_lines_mutex);
		epi_disparities[v].slice(s, 1) = res.disparity;
		epi_confidences[v].slice(s, 1) = res.confidence;
		epi_loaded_lines[v][s] = true;
	};
	
	
	// refine and propagate estimated disparities for all epis
	// write into output_disparities (function argument)
	std::cout << "processing epis..." << std::endl;
	#pragma omp parallel for
	for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
		// for epi line (v,s) refine estimated disparities using bilateral median over v and u
		// then propagate each disparity pixel over the epi by drawing line segment in output_disparities
		// repeat for different s, until output_disparities is filled (except for low epi_confidences[v][u][s] pixels)
		std::cout << "epi v=" << v << std::endl;
		
		ndarray_view<2, real> output_disparities = output_epi_disparities[v];
		ndarray_view<2, rgb_color> epi = epis[v];
		ndarray_view<2, real> disparities = epi_disparities[v];
		ndarray_view<2, real> confidences = epi_confidences[v];
		
		// holes = output_disparities pixels that still need to be filled
		// masked pixels (= low confidence) don't count as holes
		// line_remaining_holes[s] will become < 0 when masked pixels on other lines get filled by depth propagation
		std::vector<std::ptrdiff_t> line_remaining_holes(s_sz, u_sz);
		for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
			line_remaining_holes[s] = u_sz;
			for(std::ptrdiff_t u = 0; u < u_sz; ++u)
				if(! std::isnan(output_disparities[u][s])) line_remaining_holes[s]--;
		}
		// decrement for pixels that are already filled, from previous process_scale() call
		// confidences[u][s] not loaded yet. decrement for masked pixels during "s-iteration" for that s.
		
		
		// start with s in mid-height ("s-iteration")
		const std::ptrdiff_t mid_s = s_sz / 2;
		std::ptrdiff_t s = mid_s;
		
		bool finished = false;
		while(! finished) {
			std::cout << "   line s=" << s << std::endl;
			
			// need disparities+confidences values for all line s, on window of neighboring epis
			for(std::ptrdiff_t v2 = std::max<std::ptrdiff_t>(v - bilateral_window_rad, 0);
				v2 < std::min<std::ptrdiff_t>(v + bilateral_window_rad, v_sz);
				++v2)
				process_epi_line(v2, s);
			
			// moving over line (v,s)
			for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
				// skip pixel if already filled
				// (from depth propagation in previous s-iteration, or from previous process_scale() call)
				if(! std::isnan(output_disparities[u][s])) continue;
				// don't decrement line_remaining_holes here:
				// - if previous process_scale() call: got decremented before s-iteration loop
				// - if from previous s-iteration: got decremented with depth propagation
				
				// refine disparities[u][s], using median filter over neighboring u and v (which were loaded before)
				real d;
				{
					std::shared_lock<std::shared_timed_mutex> lock(epi_loaded_lines_mutex);
					
					// skip low confidence pixels
					if(confidences[u][s] <= segment_confidence_threshold) {
						line_remaining_holes[s]--; // skipped low confidence pixels don't count as holes
						continue;
					}
					Assert_crit(! std::isnan(disparities[u][s])); // if NAN only if confidences == 0.0
					
					// refine disparity d = [u][s]
					d = epi_disparity_median(
						epis.slice(s, 2), epi_disparities.slice(s, 2), epi_confidences.slice(s, 2),
						u, v,
						bilateral_window_rad, 100.0, 100.0
					);
					Assert_crit(! std::isnan(d));
				}
				
				// fill this pixel
				output_disparities[u][s] = d;
				line_remaining_holes[s]--;
				
				// TODO average color for this pixel (over segment)
				rgb_color avg_color = epis[v][u][s];
				
				// propagate d: draw segment upwards and downwards starting from pixel (u,s),
				// until color in epi becomes too different
				auto propagate = [&](std::ptrdiff_t s2) {
					Assert_crit(s2 != s);
					std::ptrdiff_t u2 = u  + (s - s2)*d;
					if(u2 < 0 || u2 >= u_sz) return false;
					if(1||color_diff(epi[u2][s2], avg_color) < 100.0) {
						if(std::isnan(output_disparities[u2][s2])) {
							line_remaining_holes[s2]--;
							output_disparities[u2][s2] = d;
						} else if(output_disparities[u2][s2] < d) {
							output_disparities[u2][s2] = d;
						}
						return true;
					} else {
						return false;
					}
				};
				for(std::ptrdiff_t s2 = s + 1; s2 < s_sz; ++s2) if(! propagate(s2)) break;
				for(std::ptrdiff_t s2 = s - 1; s2 >= 0; --s2) if(! propagate(s2)) break;

				// TODO create sparse epi
			}
			
			Assert_crit(line_remaining_holes[s] <= 0);
			
			// repeat with s set to the closest (lower or higher) for which line_remaining_holes[s] > 0
			// stop if none such exists
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
		
		
		//reals_export(output_disparities, "../output/new_disparities_" + std::to_string(v) + ".png");
		
		/*
		ndarray<2, rgb_color> re_epi = sparse_epis[v].reconstruct();
		image_export(make_image_view(re_epi.view()), "../output/re_epi" + std::to_string(v) + ".png");
		*/
		
	}

}


void add_to_final_epi_disparities(const ndarray<3, real>& scaled_epi_disparities) {
	#pragma omp parallel for
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray<2, real> scaled_disparity_image = scaled_epi_disparities.slice(s, 2);
		
	}
}


int main() {
	// load all images/epis in memory
	std::cout << "loading epis..." << std::endl;
	for(int s = 0; s < s_sz; ++s) epis.slice(s, 2) = load_image(s).array_view();

	ndarray<3, real> epi_disparities(make_ndsize(v_sz, u_sz, s_sz));
	std::fill(epi_disparities.begin(), epi_disparities.end(), NAN);
	
	std::cout << "***** first run: scale 1" << std::endl;
	process_scale(epi_disparities.view());
	
	final_epi_disparities.assign(epi_disparities);
	
	while(u_sz > 10 || v_sz > 10) {
		scale_down();
		std::cout << "***** next run: scale " << scale_down_factor << std::endl;
		
		ndarray<3, real> scaled_epi_disparities(make_ndsize(v_sz, u_sz, s_sz));
		scale_down_epi_disparities(epi_disparities, scaled_epi_disparities);
		
		process_scale(scaled_epi_disparities);
		
		epi_disparities = std::move(scaled_epi_disparities);
		add_to_final_epi_disparities(epi_disparities);
	}
	
	
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray_view<2, real> disparity_image = final_epi_disparities.slice(s, 2);
		reals_export(disparity_image, "../output/disp_"+std::to_string(s)+".png");
	}
}
