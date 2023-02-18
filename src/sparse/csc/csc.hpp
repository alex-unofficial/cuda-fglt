#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <cstdlib>
#include <cstring>

namespace sparse {

	template <typename datatype> struct CSC {
		int n_cols;
		int n_nz;

		datatype *nz_values;
		int *row_idx;
		int *col_ptr;

		CSC();
		CSC(const CSC<datatype> &other);
		CSC(int n_cols, int n_nz);
		CSC(const char *mtx_fname);

		~CSC();
	};

	extern "C" int import_matrix(
			const char *mtx_fname, 
			int *num_cols, 
			int *num_nz, 
			int **row_idx, 
			int **col_ptr
			);

	template <typename datatype> CSC<datatype>::CSC() {
		this->n_cols = 0;
		this->n_nz = 0;

		this->nz_values = NULL;
		this->row_idx = NULL;
		this->col_ptr = NULL;
	}

	template <typename datatype> CSC<datatype>::CSC(const CSC<datatype> &other) {
		this->n_cols = other->n_cols;
		this->n_nz = other->n_nz;

		this->nz_values = other->nz_values;
		this->row_idx = other->row_idx;
		this->col_ptr = other->col_ptr;
	}

	template <typename datatype> CSC<datatype>::CSC(int n_cols, int n_nz) {
		this->n_cols = n_cols;
		this->n_nz = n_nz;

		this->row_idx = (int *) malloc(n_nz * sizeof(int));
		this->col_ptr = (int *) malloc((n_cols + 1) * sizeof(int));

		this->nz_values = (datatype *) malloc(n_nz * sizeof(datatype));
		memset(this->nz_values, (datatype) 1, n_nz * sizeof(datatype));
	}

	template <typename datatype> CSC<datatype>::CSC(const char *mtx_fname) {
		import_matrix(mtx_fname, &this->n_cols, &this->n_nz, &this->row_idx, &this->col_ptr);

		this->nz_values = (datatype *) malloc(this->n_nz * sizeof(datatype));
		memset(this->nz_values, 1, this->n_nz * sizeof(datatype));
	}

	template <typename datatype> CSC<datatype>::~CSC() {
		free(this->nz_values);
		free(this->col_ptr);
		free(this->row_idx);
	}
}

#endif
