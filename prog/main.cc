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
#include "disparity_estimator_native.h"
#include "disparity_estimator_cl.h"

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
#include <stack>
#include <cstdio>
#include <omp.h>

#define EPI_USE_OPENCL

using namespace mf;

real epi_disparity_median(
	ndarray_view<2, rgba_color> colors, // [v][u]
	ndarray_view<2, real> disparities, // [v][u]
	ndarray_view<2, real> confidences, // [v][u]
	std::ptrdiff_t u, std::ptrdiff_t v
) {
	//Assert_crit(! std::isnan(disparities[v][u]));
	
	const std::ptrdiff_t rad = bilateral_window_rad;
	
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
			if(confidences[v2][u2] <= bileratal_window_confidence_min_threshold) continue;
			//Assert_crit(! std::isnan(d));
			if(color_diff(colors[v][u], colors[v2][u2]) > bilateral_window_color_diff_max_threshold) continue;
			
			median_disparities.push_back(d);
		}
	}
	
	if(median_disparities.size() == 0) {
		return disparities[v][u];
	} else {
		std::ptrdiff_t pos = median_disparities.size() / 2;
		std::nth_element(median_disparities.begin(), median_disparities.begin() + pos, median_disparities.end());
		real d = median_disparities[pos];
		//Assert_crit(! std::isnan(d));
		return d;
	}
}


image<rgba_color> load_image(std::ptrdiff_t s) {
	constexpr std::size_t image_path_buffer_size = 256;
	char image_path_buffer[image_path_buffer_size];
	
	std::snprintf(image_path_buffer, image_path_buffer_size, image_path_format, (int)s);

	image<rgba_color> orig_img = mf::image_import(image_path_buffer);
	image<rgba_color> img(flip(image_shp));
	cv::resize(orig_img.cv_mat(), img.cv_mat(), cv::Size(image_shp[0], image_shp[1]), 0.0, 0.0, cv::INTER_LINEAR);

	return img;
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
	const ndarray_view<3, real>& output_epi_disparities, // [v][u][s]
	bool last
) {
	// compute edge confidence, and min+max disparities for all epis
	std::cout << "computing edge confidence, min+max disparities for all epi..." << std::endl;
	ndarray<3, real> epi_edge_confidences(epis.shape()); // [v][u][s]
	ndarray<3, uchar> epi_edge_confidence_masks(epis.shape()); // [v][u][s]
	
	ndarray<3, real> epi_min_disparities(epis.shape());
	ndarray<3, real> epi_max_disparities(epis.shape());
	
	if(last) {
		std::fill(epi_edge_confidences.begin(), epi_edge_confidences.end(), INFINITY);
		std::fill(epi_edge_confidence_masks.begin(), epi_edge_confidence_masks.end(), 0xff);
	}
	
	#pragma omp parallel for
	for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
		std::cout << '.' << std::flush;
		if(! last) {
			epi_edge_confidences[v] = edge_confidence(epis[v]);
			epi_edge_confidence_masks[v] = edge_confidence_mask(epi_edge_confidences[v]);
		}
			
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
	
	std::mutex sparse_epis_mutex;
	
	auto process_epi_line = [&](std::ptrdiff_t v, std::ptrdiff_t s, disparity_estimator& estimator) {
		// estimate disparities+confidences for epi line (v,s)
		if(epi_loaded_lines[v][s].load()) return;
		
		ndarray<1, uchar> mask(make_ndsize(u_sz));
		for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
			bool b = epi_edge_confidence_masks[v][u][s] && std::isnan(output_epi_disparities[v][u][s]);
			mask[u] = b ? 255 : 0;
		}
		
		epi_line_disparity_result res = estimator.estimate_epi_line_disparity(v, s, mask.view());
						
		std::unique_lock<std::shared_timed_mutex> lock(epi_loaded_lines_mutex);
		epi_disparities[v].slice(s, 1) = res.disparity;
		epi_confidences[v].slice(s, 1) = res.confidence;
		epi_loaded_lines[v][s] = true;
	};
	
	
	// refine and propagate estimated disparities for all epis
	// write into output_disparities (function argument)
	std::cout << "processing epis..." << std::endl;

	#ifdef EPI_USE_OPENCL
	#pragma omp parallel for schedule(guided)
	#else
	#pragma omp parallel for schedule(dynamic)
	#endif
	for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
	//std::ptrdiff_t v = 100; if(true) {
		std::ptrdiff_t v_min = std::max<std::ptrdiff_t>(0, v - bilateral_window_rad);
		std::ptrdiff_t v_max = std::min<std::ptrdiff_t>(v_sz, v + bilateral_window_rad);
		
		#ifdef EPI_USE_OPENCL
		disparity_estimator_cl estimator(
			omp_get_thread_num() % disparity_estimator_cl::workers_count(),
			v_min,
			v_max,
			epis(v_min, v_max),
			epi_edge_confidences(v_min, v_max),
			epi_min_disparities(v_min, v_max),
			epi_max_disparities(v_min, v_max)
		);
		#else
		disparity_estimator_native estimator(
			v_min,
			v_max,
			epis(v_min, v_max),
			epi_edge_confidences(v_min, v_max),
			epi_min_disparities(v_min, v_max),
			epi_max_disparities(v_min, v_max)
		);
		#endif
		
		sparse_epi sparse(epi_shp);
		
		// for epi line (v,s) refine estimated disparities using bilateral median over v and u
		// then propagate each disparity pixel over the epi by drawing line segment in output_disparities
		// repeat for different s, until output_disparities is filled (except for low epi_confidences[v][u][s] pixels)
		std::cout << '.' << std::flush;
		//std::cout << char('A' + v*('Z'-'A')/v_sz) << std::flush;
		
		ndarray_view<2, real> output_disparities = output_epi_disparities[v]; // [u][s]
		ndarray_view<2, rgba_color> epi = epis[v]; // [u][s]
		ndarray_view<2, real> disparities = epi_disparities[v]; // [u][s] (NAN when low confidence, or when already in output_disparities from previous call)
		ndarray_view<2, real> confidences = epi_confidences[v]; // [u][s]
		
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
		//	std::cout << "  line " << s << std::endl;
			
			// need disparities+confidences values for all line s, on window of neighboring epis
			for(std::ptrdiff_t v2 = v_min; v2 < v_max; ++v2)
				process_epi_line(v2, s, estimator);
			
			// moving over line (v,s)
			for(std::ptrdiff_t u = 0; u < u_sz; ++u) {
				// skip pixel if already filled
				// (from depth propagation in previous s-iteration, or from previous process_scale() call)
				if(! std::isnan(output_disparities[u][s])) continue;
				// don't decrement line_remaining_holes here:
				// - if previous process_scale() call: got decremented before s-iteration loop
				// - if from previous s-iteration: got decremented with depth propagation
				
				// refine disparities[u][s], using median filter over neighboring u and v (which were loaded before)
				real d = disparities[u][s];
				{
					std::shared_lock<std::shared_timed_mutex> lock(epi_loaded_lines_mutex);
					
					// skip low confidence pixels (except on last scale)
					if(confidences[u][s] <= confidence_min_threshold && !last) {
						line_remaining_holes[s]--; // skipped low confidence pixels don't count as holes
						continue;
					}
					Assert_crit(! std::isnan(disparities[u][s])); // if NAN only if confidences == 0.0
					
					// refine disparity d = [u][s]
					d = epi_disparity_median(
						epis.slice(s, 2), epi_disparities.slice(s, 2), epi_confidences.slice(s, 2),
						u, v
					);
					Assert_crit(! std::isnan(d));
				}
				
				// fill this pixel
				output_disparities[u][s] = d;
				line_remaining_holes[s]--;
				
				// TODO average color for this pixel (over segment)
				rgba_color avg_color = epis[v][u][s];
				
				// propagate d: draw segment upwards and downwards starting from pixel (u,s),
				// until color in epi becomes too different
				auto propagate = [&](std::ptrdiff_t s2) {
					Assert_crit(s2 != s);
					std::ptrdiff_t u2 = u  + (s - s2)*d/scale_down_factor;
					if(u2 < 0 || u2 >= u_sz) return false;
					if(color_diff(epi[u2][s2], avg_color) <= propagation_color_threshold) {
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
				
				
				sparse_epi_segment seg{ d, u, s, avg_color };
				sparse.add_segment(seg);
				{
					std::lock_guard<std::mutex> lock(sparse_epis_mutex);
					final_sparse_epis[v] = sparse;
				}
			}
			
			Assert_crit(line_remaining_holes[s] <= 0);
			
			// repeat with s set to the closest (lower or higher) for which line_remaining_holes[s] > 0
			// stop if none such exists
			std::ptrdiff_t lower_s, higher_s;
			for(lower_s = mid_s - 1;  lower_s >= 0; lower_s--) if(line_remaining_holes[lower_s] > 0) break;
			for(higher_s = mid_s + 1; higher_s < s_sz; higher_s++) if(line_remaining_holes[higher_s] > 0) break;
			if(lower_s == -1 && higher_s == s_sz) finished = true;
			else if(lower_s == -1) s = higher_s;
			else if(higher_s == s_sz) s = lower_s;
			else if(higher_s - s > s - lower_s) s = lower_s;
			else s = higher_s;
			
			Assert_crit(finished || line_remaining_holes[s] > 0);
		}
		
		
		//reals_export(epi_edge_confidences[v], "../output/ec"+std::to_string(v)+".png");
		//image_export(make_image_view(epi_edge_confidence_masks[v]), "../output/ecm"+std::to_string(v)+".png");
		//reals_export(epi_confidences[v], "../output/c"+std::to_string(v)+".png");
		//reals_export(epi_disparities[v], "../output/d"+std::to_string(v)+".png", true);
		//reals_export(output_disparities, "../output/dr"+std::to_string(v)+".png", true);
		//image_export(make_image_view(epi), "../output/e"+std::to_string(v)+".png");
		//image_export(make_image_view(final_sparse_epis[v].reconstruct().view()), "../output/es"+std::to_string(v)+".png");
		
		
		//std::terminate();
	}
	std::cout << std::endl;
	
	reals_export(epi_disparities.slice(s_sz/2,2), "../output/disp.png", maximal_disparity);
	reals_export(output_epi_disparities.slice(s_sz/2,2), "../output/rdisp.png", maximal_disparity);
	reals_export(epi_confidences.slice(s_sz/2,2), "../output/conf.png");

	/*
	std::ptrdiff_t v = 100;
	reals_export(epi_edge_confidences[v], "../output/ec"+std::to_string(v)+".png");
	image_export(make_image_view(epi_edge_confidence_masks[v]), "../output/ecm"+std::to_string(v)+".png");
	reals_export(epi_confidences[v], "../output/c"+std::to_string(v)+".png");
	reals_export(epi_disparities[v], "../output/d"+std::to_string(v)+".png", maximal_disparity);
	reals_export(output_epi_disparities[v], "../output/dr"+std::to_string(v)+".png", maximal_disparity);
	image_export(make_image_view(epis[v]), "../output/e"+std::to_string(v)+".png");
	image_export(make_image_view(final_sparse_epis[v].reconstruct().view()), "../output/es"+std::to_string(v)+".png");
	 */

}


void scale_down() {
	// scale down in u and v directions by factor 2 (downsample images)
	
	std::ptrdiff_t scaled_u_sz = u_sz / 2;
	std::ptrdiff_t scaled_v_sz = v_sz / 2;
	
	ndarray<3, rgba_color> scaled_epis(make_ndsize(scaled_v_sz, scaled_u_sz, s_sz));
	
	// downsample images
	#pragma omp parallel for
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		ndarray<2, rgba_color> img(epis.slice(s, 2));
		
		cv::Mat_<rgba_color> img_cv = to_opencv(img.view());
		//cv::GaussianBlur(img_cv, img_cv, cv::Size(15, 15), 0.0, 0.0, cv::BORDER_DEFAULT);
		
		cv::Mat_<rgba_color> scaled_img_cv;
		cv::resize(img_cv, scaled_img_cv, cv::Size(scaled_u_sz, scaled_v_sz), 0.0, 0.0, cv::INTER_LINEAR);
		
		ndarray_view<2, rgba_color> scaled_img = to_ndarray_view<2, rgba_color>(scaled_img_cv);
		scaled_epis.slice(s, 2) = scaled_img;
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
	const ndarray_view<3, real>& output_scaled_epi_disparities // [v][u][s] with new (half) size
) {
	// out.u_sz = prev.u_sz / 2  -->  prev.u_sz = 2*out.u_sz  OR  2*out.u+1
	// out.v_sz = prev.v_sz / 2
	
	bool odd_v = (output_scaled_epi_disparities.shape()[0] > 2*previous_epi_disparities.shape()[0]);
	bool odd_u = (output_scaled_epi_disparities.shape()[1] > 2*previous_epi_disparities.shape()[1]);
	
	std::ptrdiff_t scaled_v_sz = output_scaled_epi_disparities.shape()[0];
	std::ptrdiff_t scaled_u_sz = output_scaled_epi_disparities.shape()[1];
	
	// downsample the estimated disparities
	#pragma omp parallel for
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		for(std::ptrdiff_t v = 0; v < scaled_v_sz; ++v) {
			bool no_v = odd_v && (v == scaled_v_sz-1);
			for(std::ptrdiff_t u = 0; u < scaled_u_sz; ++u) {
				bool no_u = odd_u && (u == scaled_u_sz-1);
				real d1 = previous_epi_disparities[2*v][2*u][s];
				real d2 = no_u ? NAN : previous_epi_disparities[2*v][2*u+1][s];
				real d3 = no_v ? NAN : previous_epi_disparities[2*v+1][2*u][s];
				real d4 = (no_u || no_v) ? NAN : previous_epi_disparities[2*v+1][2*u+1][s];
				
				real min_d = d1;
				if(d2 < min_d) min_d = d2; // comparison false if d2 and/or min_d is NAN
				if(d3 < min_d) min_d = d3;
				if(d4 < min_d) min_d = d4;

				output_scaled_epi_disparities[v][u][s] = min_d;
			}
		}
	}
}


void scale_up_and_merge_epi_disparities(
	const ndarray_view<3, real>& smaller_epi_disparities, // [v][u][s]
	const ndarray_view<3, real>& larger_epi_disparities // [v][u][s]
) {
	#pragma omp parallel for
	for(std::ptrdiff_t s = 0; s < s_sz; ++s) {
		cv::Mat_<real> smaller_cv, larger_cv;
		copy_to_opencv(smaller_epi_disparities.slice(s, 2), smaller_cv);
		copy_to_opencv(larger_epi_disparities.slice(s, 2), larger_cv);
		
		cv::Mat_<uchar> mask_cv = (larger_cv != larger_cv); // 1 for NAN pixels
		cv::resize(smaller_cv, smaller_cv, larger_cv.size(), 0.0, 0.0, cv::INTER_NEAREST);
		smaller_cv.copyTo(larger_cv, mask_cv);
		
		copy_to_ndarray_view(larger_cv, larger_epi_disparities.slice(s, 2));
	}
}


void export_epi_disparities(const ndarray_view<3, real>& vw, const std::string& filename) {
	Assert(vw.has_default_strides_without_padding());
	std::ofstream stream(filename, std::ios::binary);
	stream.write(reinterpret_cast<const std::ofstream::char_type*>(vw.start()), vw.size()*sizeof(real));
}

tff::ndarray<3, real> import_epi_disparities(const ndsize<3>& sz, const std::string& filename, bool remove = false) {
	tff::ndarray<3, real> arr(sz);
	{
		std::ifstream stream(filename, std::ios::binary);
		stream.read(reinterpret_cast<std::ofstream::char_type*>(arr.start()), arr.size()*sizeof(real));
	}
	if(remove) std::remove(filename.c_str());
	return arr;
}


int main() {
	disparity_estimator_cl::setup_cl();
	
	// load all images/epis in memory
	std::cout << "## loading epis..." << std::endl;
	for(int s = 0; s < s_sz; ++s) epis.slice(s, 2) = load_image(s).array_view();
	
	ndarray<3, real> epi_disparities(make_ndsize(v_sz, u_sz, s_sz));
	std::fill(epi_disparities.begin(), epi_disparities.end(), NAN);

	final_sparse_epis = std::vector<sparse_epi>(v_sz, sparse_epi(epi_shp));
	
	std::stack<ndsize<3>> pyramid_sizes;
	
	std::cout << "## down" << std::endl;
	while(u_sz > 10 && v_sz > 10) {
		std::cout << "***** scale " << scale_down_factor << ", shape " << epi_shp << std::endl;
		bool last = !((u_sz > 20) && (v_sz > 20));
		if(last) std::cout << "(last)" << std::endl;
		pyramid_sizes.push(make_ndsize(v_sz, u_sz, s_sz));
		
		std::cout << "** processing epi disparities" << std::endl;
		process_scale(epi_disparities.view(), last);
		
		break;
		
		std::cout << "** scaling down" << std::endl;
		scale_down();
		
		std::cout << "** preparing scaled epi disparities" << std::endl;
		ndarray<3, real> scaled_epi_disparities(make_ndsize(v_sz, u_sz, s_sz));
		scale_down_epi_disparities(epi_disparities.view(), scaled_epi_disparities.view());
		
		std::cout << "** exporting previous epi disparities" << std::endl;
		//export_epi_disparities(epi_disparities.view(), "../output/epi_disparities_"+std::to_string(scale_down_factor/2)+".dat");
		
		epi_disparities = std::move(scaled_epi_disparities);
	}
	
	int i = 0;
	for(real s = 0; s < s_sz; s += 0.1, ++i) {
		ndarray<2, rgba_color> recons(image_shp);
		for(std::ptrdiff_t v = 0; v < v_sz; ++v) {
			recons.slice(v, 1) = final_sparse_epis[v].reconstruct_line(s);
		}
		image_export(image<rgba_color>(flip(recons.view())).view(), "../output/recons" + std::to_string(i) + ".png");
	}
	
	
	return 0;
	
	std::size_t scale_back_up_factor = scale_down_factor;
	
	ndarray<3, real> smaller_epi_disparities = std::move(epi_disparities);

	std::cout << "## up" << std::endl;
	while(scale_back_up_factor != 1) {
		scale_back_up_factor /= 2;
		
		ndsize<3> larger_shp = pyramid_sizes.top();

		std::cout << "smaller: " << smaller_epi_disparities.shape() << ", larger: " << larger_shp << std::endl;
		pyramid_sizes.pop();
		
		std::cout << "** importing larger" << std::endl;
		tff::ndarray<3, real> larger_epi_disparities =
			import_epi_disparities(larger_shp, "../output/epi_disparities_"+std::to_string(scale_back_up_factor)+".dat");
		
		std::cout << "** upscale and merge" << std::endl;
		scale_up_and_merge_epi_disparities(smaller_epi_disparities.view(), larger_epi_disparities.view());
	
		reals_export(larger_epi_disparities.slice(s_sz/2,2), "../output/scale"+std::to_string(scale_back_up_factor)+"_dispMg.png");
		
		smaller_epi_disparities = std::move(larger_epi_disparities);
	}
	
	
	
	//for(std::ptrdiff_t s = 0; s < s_sz; s += s_sz/20)
	//	reals_export(final_epi_disparities.slice(s,2), "../output/final/final_disp_"+std::to_string(s)+".png");
}
