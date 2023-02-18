#include <iostream>
#include <cstdlib>

#include "sparse/csc/csc.hpp"

using namespace std;

int main(int argc, char **argv) {

	char *mtx_fname;
	if(argc > 1) {
		mtx_fname = argv[1];
	} else {
		cout << "not enough arguments" << endl;
		exit(1);
	}

	sparse::CSC<int> matrix = sparse::CSC<int> (mtx_fname);

	cout << matrix.n_cols << " " << matrix.n_nz << endl;

	for(int i = 0 ; i < matrix.n_nz ; i++) {
		cout << matrix.row_idx[i] << " ";
	}
	cout << endl;

	for(int j = 0 ; j < matrix.n_cols + 1 ; j++) {
		cout << matrix.col_ptr[j] << " ";
	}
	cout << endl;

	return 0;
}
