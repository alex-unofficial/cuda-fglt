/* header file for FGLT with CUDA
 * Copyright (C) 2023  Alexandros Athanasiadis
 *
 * This file is part of cuda-fglt
 *                                                                        
 * cuda-fglt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *                                                                        
 * cuda-fglt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *                                                                        
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>. 
 */

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
