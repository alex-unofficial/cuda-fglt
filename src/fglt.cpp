#include <iostream>
#include <cstdlib>

extern "C" int import_matrix(char *mtx_fname, int *num_cols, int *num_nz, int **row_idx, int **col_ptr);

int main(int argc, char **argv) {

	char *mtx_fname;
	if(argc > 1) {
		mtx_fname = argv[1];
	} else {
		std::cout << "file does not exist" << std::endl;
		exit(1);
	}

	int n_cols;
	int n_nz;
	int *row_idx;
	int *col_ptr;

	import_matrix(mtx_fname, &n_cols, &n_nz, &row_idx, &col_ptr);

	std::cout << n_cols << " " << n_nz << std::endl;

	for(int i = 0 ; i < n_nz ; i++) {
		std::cout << row_idx[i] << " ";
	}
	std::cout << std::endl;

	for(int j = 0 ; j < n_cols + 1 ; j++) {
		std::cout << col_ptr[j] << " ";
	}
	std::cout << std::endl;

	return 0;
}
