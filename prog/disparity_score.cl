#ifndef S_SZ
#error S_SZ constant not defined at kernel compile time
#endif

constant sampler_t epi_sampler =
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_FILTER_NEAREST |
	CLK_ADDRESS_CLAMP;


kernel void disparity_score(
	ptrdiff_t s,
	float d_min,
	float d_incr,
	read_only image2d_t epi,
	global float* max_scores,
	global float* max_score_disparities,
	global float* avg_scores,
	local float* u_scores
) {
	const size_t u = get_global_id(0);
	const size_t d_step = get_local_id(1);
	const real d = d_min + d_step*d_incr;
	
	const size_t u_sz = get_image_width(epi);

	const float uf = float(u);
	float4 colors[S_SZ];
	for(ptrdiff_t s2 = 0; s2 < S_SZ; ++s2) {
		float u2 = uf + d*(s - s2);
		colors[s2] = read_imagef(epi, epi_sampler, float2(u2, s2));
		// if u2 out of bounds, returns (0.0f, 0.0f, 0.0f, 1.0f)
	}
	
	float score = 0.0f;
	float score_w = 0.0f;
	float4 mean_col = read_imagef(epi, epi_sampler, float2(u, s));
	for(ptrdiff_t i = 0; i < S_SZ; ++i) {
		float4 col = colors[i];
		if(col.w == 1.0) continue;
		
		float d = fast_length(col - mean_col) / 0.02f;
		score += max(1.0f - d, 0.0f);
		score_w += 1.0f;
	}
	score /= score_w;
	
	u_scores[d_step] = score;

	////
	barrier(CLK_LOCAL_MEM_FENCE);
	if(get_local_id(1) != 0) return;
	
	float max_score = -1.0f;
	float max_score_disparity;
	float avg_score = 0.0f;
	const size_t d_steps = get_local_size(1);
	for(ptrdiff_t d_step = 0; d_step < d_steps; ++d_step) {
		float score = u_scores[d_step];
		avg_scores += score;
		if(score > max_scores) {
			max_score = score;
			max_score_disparity = d_min + d_step*d_incr;
		}
	}
	avg_scores /= d_steps;
		
	max_scores[u] = max_score;
	max_score_disparities[u] = max_score_disparity;
	avg_scores[u] = avg_score;
}
