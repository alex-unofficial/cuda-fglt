#include <iostream>
#include <cstdlib>

#include "sparse/csc/csc.hpp"
#include "fglt/fglt.cuh"

using namespace std;

int main(int argc, char **argv) {

	char *mtx_fname;
	if(argc > 1) {
		mtx_fname = argv[1];
	} else {
		cout << "not enough arguments" << endl;
		exit(1);
	}

	sparse::CSC<double> const * const matrix = new sparse::CSC<double> (mtx_fname);

	double *f_base = (double *) malloc(NGRAPHLET * matrix->n_cols * sizeof(double *));
	double *fn_base = (double *) malloc(NGRAPHLET * matrix->n_cols * sizeof(double *));

	double *f[NGRAPHLET];
	double *fn[NGRAPHLET];
	for(int k = 0 ; k < NGRAPHLET ; k++) {
		f[k] = (double *)(f_base + matrix->n_cols * k);
		fn[k] = (double *)(fn_base + matrix->n_cols * k);
	}

	cuFGLT::compute(matrix, f_base, fn_base);

	for(int i = 0 ; i < matrix->n_cols ; i++) {
		cout << f[0][i] << " " << f[1][i] << " " << f[2][i] << " " << f[3][i] << " " << f[4][i] << endl;
	}

	delete matrix;

	return 0;
}
