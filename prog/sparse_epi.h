#ifndef EPI_SPARSE_EPI_H_
#define EPI_SPARSE_EPI_H_

#include <cstddef>
#include "common.h"
#include "nd.h"
#include "color.h"

namespace mf {

struct sparse_epi_segment {
	real d;
	std::ptrdiff_t u;
	std::ptrdiff_t s;
	rgba_color avg_color;
};

class sparse_epi {
private:
	ndsize<2> shape_;
	mutable std::vector<sparse_epi_segment> segments_;
	
public:
	explicit sparse_epi(const ndsize<2>& shp) : shape_(shp) { }
	
	void add_segment(const sparse_epi_segment& seg);
	ndarray<2, rgba_color> reconstruct() const;
	ndarray<1, rgba_color> reconstruct_line(real s) const;
};

}

#endif
