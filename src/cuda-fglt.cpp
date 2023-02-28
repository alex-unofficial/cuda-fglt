/* cuda-fglt main function
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

#include <iostream>
#include <cstdlib>

#include "sparse/csc/csc.hpp"
#include "cuFGLT/cufglt.cuh"
#include "FGLT/fglt.hpp"

using namespace std;

int main(int argc, char **argv) {

	char *mtx_fname;
	if(argc > 1) {
		mtx_fname = argv[1];
	} else {
		cout << "not enough arguments" << endl;
		exit(1);
	}

	cout << "=== loading file " << mtx_fname << endl;
	sparse::CSC<double> const * const matrix = new sparse::CSC<double> (mtx_fname);
	cout << "num_cols = " << matrix->n_cols << endl;
	cout << "num_nz = " << matrix->n_nz << endl;
	cout << "=== loading done" << endl << endl;

	double *cu_fbase = (double *) malloc(NGRAPHLET * matrix->n_cols * sizeof(double));
	double *cu_fn_base = (double *) malloc(NGRAPHLET * matrix->n_cols * sizeof(double));

	double *cu_f[NGRAPHLET];
	double *cu_fn[NGRAPHLET];

	double **f = (double **) malloc(NGRAPHLET * sizeof(double *));
	double **fn = (double **) malloc(NGRAPHLET * sizeof(double *));

	for(int k = 0 ; k < NGRAPHLET ; k++) {
		cu_f[k] = (double *)(cu_fbase + matrix->n_cols * k);
		cu_fn[k] = (double *)(cu_fn_base + matrix->n_cols * k);

		f[k] = (double *) malloc(matrix->n_cols * sizeof(double));
		fn[k] = (double *) malloc(matrix->n_cols * sizeof(double));
	}

	cout << "=== starting CUDA compute operation" << endl;

	cout << "NUMBLOCKS = " << NUMBLOCKS << endl;
	cout << "NUMTHREADS = " << NUMTHREADS << endl;

	int cu_total_time = cuFGLT::compute(matrix, cu_fbase, cu_fn_base);

	cout << "CUDA Total time: " << cu_total_time << " msec" << endl;

	cout << "=== CUDA compute operation done" << endl << endl;


	cout << "=== starting CPU compute operation" << endl;

	cout << "cilkThreads = " << FGLT::getWorkers() << endl;

	int cpu_total_time = FGLT::compute(
			f, fn, matrix->row_idx, matrix->col_ptr,
			matrix->n_cols, matrix->n_nz, FGLT::getWorkers()
			);

	cout << "CPU Total time: " << cpu_total_time << " msec" << endl;
	cout << "=== CPU compute operation done" << endl << endl;

	cout << "=== checking for errors" << endl;
	int errors = 0;
	for(int i = 0 ; i < matrix->n_cols ; i++) {
		for(int k = 0 ; k < NGRAPHLET ; k++) {
			if(f[k][i] != cu_f[k][i]) {
				errors += 1;
				cout << errors << ": f[" << k << "][" << i << "] = " << f[k][i] << " =/= " 
				     << "cu_f[" << k << "][" << i << "] = " << cu_f[k][i] << endl;
			}

			if(fn[k][i] != cu_fn[k][i]) {
				errors += 1;
				cout << errors << ": fn[" << k << "][" << i << "] = " << fn[k][i] << " =/= " 
				     << "cu_fn[" << k << "][" << i << "] = " << cu_fn[k][i] << endl;
			}
		}
	}
	cout << "found " << errors << " errors." << endl;
	cout << "=== error checking done" << endl << endl;

	free(cu_fbase);
	free(cu_fn_base);

	for(int k = 0 ; k < NGRAPHLET ; k++) {
		free(f[k]);
		free(fn[k]);
	}

	free(f);
	free(fn);

	delete matrix;

	return 0;
}
