#ifndef FGLT_CUH
#define FGLT_CUH

#include "sparse/csc/csc.hpp"

#define NGRAPHLET 5

namespace cuFGLT {

	int compute(
			sparse::CSC<double> const * const adj,
			double * const f,
			double * const fn);
}

#endif
