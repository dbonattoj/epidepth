#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef S_SZ
#error S_SZ constant not defined at kernel compile time
#endif


constant sampler_t epi_sampler =
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_FILTER_LINEAR |
	CLK_ADDRESS_CLAMP;


inline double color_diff(float4 a, float4 b) {
	b.w = 1.0;
	return fast_length(a - b);
}


inline float ep_kernel(float4 a, float4 b, float threshold) {
	b.w = 1.0;
	float q = fast_length(a - b) / threshold;
	q = 1.0f - min(q, 1.0f);
	return q*q;
}


kernel void estimate_epi_line_disparity(
	double depth_score_color_threshold,
	int v,
	int s,
	read_only image2d_array_t epi_array, // x: s, y: u
	global const double* min_disp_3dim,
	global const double* max_disp_3dim,
	constant const uchar* mask,
	local double* scores,
	global double* output_max_score,
	global double* output_max_score_disp,
	global double* output_avg_score
) {
	const size_t u_sz = get_global_size(0);
	const ptrdiff_t u = get_global_id(0);
	const size_t d_sz = get_local_size(1);
	const ptrdiff_t d_idx = get_local_id(1);

	const global double* min_disp = min_disp_3dim + u_sz*S_SZ*v;
	const global double* max_disp = max_disp_3dim + u_sz*S_SZ*v;
	
	uchar m = mask[u];
	if(m == 0) return;

	double max_d = max_disp[S_SZ*u + s];
	double min_d = min_disp[S_SZ*u + s];

	double d = min_d + (max_d-min_d)*d_idx/d_sz;

	float4 cols[S_SZ];
	for(ptrdiff_t s2 = 0; s2 < S_SZ; ++s2) {
		float u2 = u + (s - s2)*d;
		float4 col2 = read_imagef(epi_array, epi_sampler, (float4)(s2, u2, v, 0));
		// if u2 out of bounds, col2 is (0.0f, 0.0f, 0.0f, 0.0f)
		cols[s2] = col2;
	}

	float4 col = read_imagef(epi_array, epi_sampler, (float4)(s, u, v, 0));
	
	for(int i = 0; i < 5; ++i) {
		float4 num = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		float den = 0.0f;
		for(ptrdiff_t s2 = 0; s2 < S_SZ; ++s2) {
			float4 col2 = cols[s2];
			float w = ep_kernel(col, col2, depth_score_color_threshold);
			den += w;
			num += w * col2;
		}
		col = num / den;
	}


	float score = 0.0;
	for(ptrdiff_t s2 = 0; s2 < S_SZ; ++s2) {
		float4 col2 = cols[s2];
		score += ep_kernel(col, col2, depth_score_color_threshold);
	}
	score /= S_SZ;


	scores[d_idx] = score;

	barrier(CLK_LOCAL_MEM_FENCE);
	if(d_idx != 0) return;

	double max_score = -1.0;
	double max_score_disp;
	double sum_score = 0.0;
	for(ptrdiff_t d_idx = 0; d_idx < d_sz; ++d_idx) {
		double score = scores[d_idx];
		if(score > max_score) {
			max_score = score;
			max_score_disp = min_d + (max_d-min_d)*d_idx/d_sz;
		}
		sum_score += score;
	}
	output_max_score[u] = max_score;
	output_max_score_disp[u] = max_score_disp;
	output_avg_score[u] = sum_score / d_sz;
}
