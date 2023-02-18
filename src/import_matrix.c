/* import matrix function
 * Copyright (C) 2022-2023  Alexandros Athanasiadis
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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <errno.h>
#include <string.h>

#include "mmio.h"

/* Compare the second index given 2 pairs of integer points
 *
 * helper function to be used inside qsort.
 * takes as input a pair a,b of void* that point to pairs of numbers,
 * used for row and column indexes, and compares them based on their second element
 * effectively comparing between their column indices.
 *
 * returns -1 if col_a < col_b or if col_a = col_b and row_a < row_b
 * 			1 if col_a > col_b or if col_a = col_b and row_a > row_b
 * 			0 if col_a = col_b and row_a = row_b
 */
static int comp_col(const void *a, const void *b) {
	int row_a = ((int *)a)[0];
	int row_b = ((int *)b)[0];

	int col_a = ((int *)a)[1];
	int col_b = ((int *)b)[1];
	
	if(col_a < col_b) return -1;
	else if(col_a > col_b) return 1;
	else {
		if(row_a < row_b) return -1;
		else if(row_a > row_b) return 1;
		else return 0;
	}
}

/* Imports a graph's adjacency matrix from a MatrixMarket .mtx file and 
 * converts it to the CSC sparse matrix format.
 *
 * Takes as input the path to the .mtx file to be imported and pointers
 * to the variables that need to be set for the CSC format
 *
 * It initializes the arrays based on the size of the matrix and fills them with the correct values.
 *
 * The .mtx file must be in the format sparse (coordinate), and the values must
 * be either integer, pattern or real.
 *
 * we are concerned mostly with the shape of the graph the matrix represents so the
 * values are discarded, and only the location of the nonzero elements is saved.
 */
int import_matrix(char *mtx_fname, int *num_cols, int *num_nz, int **row_idx, int **col_ptr) {

	// Attempt to open the file mtx_fname and checking for errors.
	FILE *mtx_file = NULL;
	mtx_file = fopen(mtx_fname, "r"); if(mtx_file == NULL) { 
		fprintf(stderr, "Error opening file: %s\n%s\n", mtx_fname, strerror(errno));
		return 1;
	}

	// The typecode struct stores information about the type of matrix the .mtx file represents
	MM_typecode mtx_type;

	// Attempt to read the banner of the matrix, handle errors
	int mm_read_err_code = 0;
	mm_read_err_code = mm_read_banner(mtx_file, &mtx_type);
	if(mm_read_err_code) {
		fprintf(stderr, "Error reading MatrixMarket banner: %s\n", mtx_fname);

		switch(mm_read_err_code) {
			case MM_PREMATURE_EOF:
				fprintf(stderr, "Items missing from file header\n");
				break;
			case MM_NO_HEADER:
				fprintf(stderr, "File missing header\n");
				break;
			case MM_UNSUPPORTED_TYPE:
				fprintf(stderr, "Invalid header information\n");
				break;
			default:
				fprintf(stderr, "Unknown error code: %d\n", mm_read_err_code);
				break;
		}

		fclose(mtx_file);
		return 1;
	}

	// the dimensions of the matrix and the nonzero elements
	int n_rows = 0;
	int n_cols = 0;
	int n_nz = 0;

	// Attempt to read size information, only if matrix is of type coordinate
	if(mm_is_coordinate(mtx_type)) {
		mm_read_err_code = mm_read_mtx_crd_size(mtx_file, (int *) &n_rows, (int *) &n_cols, (int *) &n_nz);
	} else {
		char* type = mm_typecode_to_str(mtx_type);
		fprintf(stderr, "Invalid matrix type: %s\nmatrix must be of type coordinate\n", type);
		free(type);

		fclose(mtx_file);
		return 1;
	}

	// Handle errors related to reading the size information
	if(mm_read_err_code) {
		fprintf(stderr, "Error reading MatrixMarket matrix size: %s\n", mtx_fname);

		switch(mm_read_err_code) {
			case MM_PREMATURE_EOF:
				fprintf(stderr, "EOF encountered while reading matrix size\n");
				break;
			default:
				fprintf(stderr, "Unknown error code: %d\n", mm_read_err_code);
				break;
		}

		fclose(mtx_file);
		return 1;
	}

	// graph adjacent matrices are square. fail if n_rows != n_cols
	if(n_rows != n_cols) {
		fprintf(stderr, 
				"Error: incompatible size: %s\nmatrix must have equal number of rows and columns.\n", 
				mtx_fname);

		fclose(mtx_file);
		return 1;
	}

	// is true if the matrix is of type symmetric
	bool is_symmetric = (bool) mm_is_symmetric(mtx_type);

	// ind_num is the number of indices and it may be nnz if the matrix is general,
	// or up to 2*nnz if the matrix is symmetric.
	int ind_num = is_symmetric? 2*n_nz : n_nz;

	// indices will store the pairs of row, column indices of nonzero elements
	// in the .mtx file, later to be stored in the appropriate structs
	int indices[ind_num][2];

	// num_entries will hold the real number of entries in the indices list.
	int num_entries = 0;

	for(int i = 0 ; i < n_nz ; ++i) {
		int row, col;

		int fscanf_match_needed = 0;
		int fscanf_match_count = 0;

		// get the indices from the file, depending on the type of Matrix in the file,
		// discarding the value of the matrix if needed.
		if(mm_is_pattern(mtx_type)) {
			fscanf_match_needed = 2;
			fscanf_match_count = fscanf(mtx_file, "%u %u\n", &row, &col);
		} else if(mm_is_integer(mtx_type)) {
			int val;
			fscanf_match_needed = 3;
			fscanf_match_count = fscanf(mtx_file, "%u %u %d\n", &row, &col, &val);
		} else if(mm_is_real(mtx_type)) {
			double val;
			fscanf_match_needed = 3;
			fscanf_match_count = fscanf(mtx_file, "%u %u %lf\n", &row, &col, &val);
		} else {
			fprintf(stderr, "MatrixMarket file is of unsupported format: %s\n", mtx_fname);

			fclose(mtx_file);
			return 1;
		}

		if(fscanf_match_count == EOF) {
			if(ferror(mtx_file)) {
				fprintf(stderr, "Error: reading from %s\n", mtx_fname);
			} else {
				fprintf(stderr, "Error: fscanf matching failure for %s\n", mtx_fname);
			}
			
			fclose(mtx_file);
			return 1;
		} else if(fscanf_match_count < fscanf_match_needed) {
			fprintf(stderr, "Error reading from %s:\nfscanf early matching failure\n", mtx_fname);

			fclose(mtx_file);
			return 1;
		}

		if(row > n_rows || col > n_cols) {
			fprintf(stderr, "Invalid index in .mtx file %s\n", mtx_fname);

			fclose(mtx_file);
			return 1;
		}

		if(!is_symmetric || row == col) {
			// if the matrix is non-symmetric or the entry is on the matrix diagonal
			// simply add the entry to the indices list
			indices[num_entries][0] = row - 1;
			indices[num_entries][1] = col - 1;

			num_entries += 1;
		} else {
			// if the matrix is symmetric then add both the entry and it's inverse (j, i)
			// to the indices list
			indices[num_entries][0] = row - 1;
			indices[num_entries][1] = col - 1;

			indices[num_entries + 1][0] = col - 1;
			indices[num_entries + 1][1] = row - 1;

			num_entries += 2;
		}
	}

	// since we have read all the data from the .mtx file we can close it.
	fclose(mtx_file);


	// initialize the arrays to be returned
	*num_cols = n_cols;
	*num_nz = num_entries;

	*row_idx = (int *) malloc(*num_nz * sizeof(int));
	*col_ptr = (int *) malloc((*num_cols + 1) * sizeof(int));

	/* the indices array is effectively a COO representation of the matrix.
	 *
	 * the code for converting from COO to CSC/CSR was adapted from the link below:
	 * https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
	 */

	// sort the indices based on the column index in order to create the CSC format
	qsort(indices, *num_nz, 2 * sizeof(int), comp_col);

	// col_ptr[col + 1] will initialy hold the number of nz elements in col, 
	// then we perform a cumulative sum which will be in the form we need.
	
	// initialize col_ptr to 0. 
	for(int i = 0 ; i <= *num_cols ; ++i) {
		(*col_ptr)[i] = 0;
	}

	// then we loop over the COO array
	for(int i = 0 ; i < *num_nz ; ++i) {
		int row = indices[i][0];
		int col = indices[i][1];

		// store the row in row_idx as is
		(*row_idx)[i] = row;

		// increase col_ptr for index: col+1 since 1 more nz element is present in col
		(*col_ptr)[col + 1] += 1;
	}

	// then perform the cumulative sum
	for(int i = 0 ; i < *num_cols ; ++i) {
		(*col_ptr)[i + 1] += (*col_ptr)[i];
	}

	return 0;
}

