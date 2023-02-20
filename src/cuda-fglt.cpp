#include <iostream>
#include <cstdlib>

#include "sparse/csc/csc.hpp"
#include "cuFGLT/fglt.cuh"

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
	cout << "=== loading done" << endl << endl;

	double *f_base = (double *) malloc(NGRAPHLET * matrix->n_cols * sizeof(double));
	double *fn_base = (double *) malloc(NGRAPHLET * matrix->n_cols * sizeof(double));

	double *f[NGRAPHLET];
	double *fn[NGRAPHLET];
	for(int k = 0 ; k < NGRAPHLET ; k++) {
		f[k] = (double *)(f_base + matrix->n_cols * k);
		fn[k] = (double *)(fn_base + matrix->n_cols * k);
	}

	cout << "=== starting compute operation" << endl;
	cuFGLT::compute(matrix, f_base, fn_base);
	cout << "=== compute operation done" << endl << endl;

	cout << "RESULTS:" << endl;
	for(int i = 0 ; i < matrix->n_cols ; i++) {
		cout << f[0][i] << " " << f[1][i] << " " << f[2][i] << " " << f[3][i] << " " << f[4][i] << endl;
	}

	free(f_base);
	free(fn_base);

	delete matrix;

	return 0;
}
