#ifndef CUFGLT_CUH
#define CUFGLT_CUH

#include "sparse/csc/csc.hpp"

#ifndef NGRAPHLET
#define NGRAPHLET 5
#endif

#ifndef NUMBLOCKS
#define NUMBLOCKS 512
#endif

#ifndef NUMTHREADS
#define NUMTHREADS 32
#endif

namespace cuFGLT {

	int compute(
			sparse::CSC<double> const * const adj,
			double * const f,
			double * const fn);
}

#endif
