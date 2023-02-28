/* methods for computing FGLT with CUDA
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

#include "cufglt.cuh"

#include <iostream>
#include <chrono>

using namespace std;

__global__ static void fill_d0(
		double * const d_f0,
		int n_cols);


__global__ static void compute_d1(
		double * const d_f1, 
		int const * const d_col_ptr,
		int n_cols);


__global__ static void compute_d2(
		double * const d_f2,
		int const * const d_row_idx,
		int const * const d_col_ptr,
		double const * const d_p1,
		int n_cols);


__global__ static void compute_d3(
		double * const d_f3,
		double const * const d_p1,
		int n_cols);


__global__ static void compute_d4(
		double * const d_f4,
		int const * const d_row_idx,
		int const * const d_col_ptr,
		int n_cols);


static void raw2net(
		const double * const f0,
		const double * const f1,
		const double * const f2,
		const double * const f3,
		const double * const f4,
		double * const fn0,
		double * const fn1,
		double * const fn2,
		double * const fn3,
		double * const fn4,
		int n_cols);


__device__ static void sum_reduce(
		double * const t_results,
		int thread_num);


__device__ static int sparse_dot_prod(
		int const * const vec1_idx,
		int vec1_nnz,
		int const * const vec2_idx,
		int vec2_nnz);


int cuFGLT::compute(
		sparse::CSC<double> const * const adj,
		double * const f_base,
		double * const fn_base) {

	/* extract matrix information */

	const int n_cols = adj->n_cols;
	const int n_nz = adj->n_nz;

	int const * const row_idx = adj->row_idx;
	int const * const col_ptr = adj->col_ptr;


	/* Allocate and transfer data from host to device */

	double *d_f_base; 
	int d_pitch;

	// allocate memory for raw and net frequencies on device
	cudaMallocPitch((void **) &d_f_base, (size_t *) &d_pitch, n_cols * sizeof(double), NGRAPHLET);

	// create pointers to the frequency vectors
	double *d_f[NGRAPHLET];

	double *f[NGRAPHLET];
	double *fn[NGRAPHLET];

	for(int k = 0 ; k < NGRAPHLET ; k++) {
		d_f[k] = (double *)((char*)d_f_base + k * d_pitch);

		f[k] = (f_base + n_cols * k);
		fn[k] = (fn_base + n_cols * k);
	}

	int *d_row_idx;
	int *d_col_ptr;

	// allocate memory for the matrix arrays
	cudaMalloc((void **) &d_row_idx, n_nz * sizeof(int));
	cudaMalloc((void **) &d_col_ptr, (n_cols + 1) * sizeof(int));

	// Create streams and events
	cudaStream_t d0_stream, d13_stream, d42_stream;
	cudaStreamCreate(&d0_stream);
	cudaStreamCreate(&d13_stream);
	cudaStreamCreate(&d42_stream);

	cudaEvent_t col_ptr_cpy, d1_complete;
	cudaEventCreate(&col_ptr_cpy);
	cudaEventCreate(&d1_complete);

	/* Compute raw frequencies */

	auto start = chrono::high_resolution_clock::now();
	// transfer matrix data from host to device
	cudaMemcpyAsync(d_col_ptr, col_ptr, (n_cols + 1) * sizeof(int), cudaMemcpyHostToDevice, d13_stream);
	cudaEventRecord(col_ptr_cpy, d13_stream);

	cudaMemcpyAsync(d_row_idx, row_idx, n_nz * sizeof(int), cudaMemcpyHostToDevice, d42_stream);

	// fill d0
	fill_d0<<<NUMBLOCKS, NUMTHREADS, 0, d0_stream>>>(d_f[0], n_cols);

	// compute d1
	compute_d1<<<NUMBLOCKS, NUMTHREADS, 0, d13_stream>>>(d_f[1], d_col_ptr, n_cols);
	cudaEventRecord(d1_complete, d13_stream);
	
	// compute d3
	compute_d3<<<NUMBLOCKS, NUMTHREADS, 0, d13_stream>>>(d_f[3], d_f[1], n_cols);

	// compute d4
	cudaStreamWaitEvent(d42_stream, col_ptr_cpy);
	compute_d4<<<NUMBLOCKS, NUMTHREADS, 0, d42_stream>>>(d_f[4], d_row_idx, d_col_ptr, n_cols);

	// compute d2
	cudaStreamWaitEvent(d42_stream, d1_complete);
	compute_d2<<<NUMBLOCKS, NUMTHREADS>>>(d_f[2], d_row_idx, d_col_ptr, d_f[1], n_cols);

	/* Transfer data from device to host and free memory on device */

	cudaDeviceSynchronize();
	cudaMemcpy2D(
			f_base, n_cols * sizeof(double), 
			d_f_base, d_pitch, 
			n_cols * sizeof(double), NGRAPHLET, 
			cudaMemcpyDeviceToHost);

	/* Transform raw freq to net freq */

	raw2net(f[0],  f[1],  f[2],  f[3],  f[4], 
	        fn[0], fn[1], fn[2], fn[3], fn[4],
	        n_cols);

	auto end = chrono::high_resolution_clock::now();
	auto total_duration = chrono::duration_cast<chrono::milliseconds>(end - start);

	cudaFree(d_f_base);

	cudaFree(d_row_idx);
	cudaFree(d_col_ptr);

	return total_duration.count();
}


__global__ static void fill_d0(
		double * const d_f0,
		int n_cols) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < n_cols) {
		d_f0[idx] = 1.0;
		idx += blockDim.x * gridDim.x;
	}
}


__global__ static void compute_d1(
		double * const d_f1, 
		int const * const d_col_ptr,
		int n_cols) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < n_cols) {
		d_f1[idx] = (double)(d_col_ptr[idx + 1] - d_col_ptr[idx]);
		idx += blockDim.x * gridDim.x;
	}
}


__global__ static void compute_d2(
		double * const d_f2,
		int const * const d_row_idx,
		int const * const d_col_ptr,
		double const * const d_p1,
		int n_cols) {

	int i = blockIdx.x;

	const int thread_num = blockDim.x;
	__shared__ double t_results[NUMTHREADS];

	while(i < n_cols) {
		int tid = threadIdx.x;
		double t_sum = 0;

		int j = d_col_ptr[i] + tid;
		int final_idx = d_col_ptr[i + 1];

		while(j < final_idx) {
			t_sum += d_p1[d_row_idx[j]];
			j += thread_num;
		}

		t_results[tid] = t_sum;

		sum_reduce(t_results, thread_num);

		if(tid == 0) {
			d_f2[i] = (double)(t_results[0] - d_p1[i]);
		}

		i += gridDim.x;
	}
}


__global__ static void compute_d3(
		double * const d_f3,
		double const * const d_p1,
		int n_cols) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < n_cols) {
		double p1_i = d_p1[idx];
		d_f3[idx] = (double)(p1_i * (p1_i - 1)) / 2.0;
		idx += blockDim.x * gridDim.x;
	}
}


__global__ static void compute_d4(
		double * const d_f4,
		int const * const d_row_idx,
		int const * const d_col_ptr,
		int n_cols) {

	int i = blockIdx.x;

	const int thread_num = blockDim.x;
	__shared__ double t_results[NUMTHREADS];

	while(i < n_cols) {
		int tid = threadIdx.x;
		double t_sum = 0;

		int first_idx = d_col_ptr[i];
		int final_idx = d_col_ptr[i + 1];

		int const * const ai = d_row_idx + first_idx;
		int ai_nnz = final_idx - first_idx;

		int j_ptr = first_idx + tid;
		while(j_ptr < final_idx) {
			int j = d_row_idx[j_ptr];

			int const * const aj = d_row_idx + d_col_ptr[j];
			int aj_nnz = d_col_ptr[j + 1] - d_col_ptr[j];

			t_sum += sparse_dot_prod(ai, ai_nnz, aj, aj_nnz);

			j_ptr += thread_num;
		}

		t_results[tid] = t_sum;

		sum_reduce(t_results, thread_num);

		if(tid == 0) {
			d_f4[i] = (double)(t_results[0]) / 2.0;
		}

		i += gridDim.x;
	}
}


static void raw2net(
		const double * const f0,
		const double * const f1,
		const double * const f2,
		const double * const f3,
		const double * const f4,
		double * const fn0,
		double * const fn1,
		double * const fn2,
		double * const fn3,
		double * const fn4,
		int n_cols) {

	for(int idx = 0 ; idx < n_cols ; idx++) {
		fn0[idx] = f0[idx];
		fn1[idx] = f1[idx];
		fn2[idx] = f2[idx] - 2 * f4[idx];
		fn3[idx] = f3[idx] - f4[idx];
		fn4[idx] = f4[idx];
	}
}


__device__ static void sum_reduce(
		double * const t_results,
		int thread_num) {
	
	int tid = threadIdx.x;

	__syncthreads();

	for(int offset = 1; offset < thread_num ; offset <<= 1) {
		int mod = offset << 1;

		if(tid % mod == 0 && (tid + offset) < thread_num) {
			t_results[tid] += t_results[tid + offset];
		}

		__syncthreads();
	}
}


__device__ static int sparse_dot_prod(
		int const * const vec1_idx,
		int vec1_nnz,
		int const * const vec2_idx,
		int vec2_nnz) {

	double res = 0;

	int i_ptr = 0, j_ptr = 0;

	int i = vec1_idx[i_ptr];
	int j = vec2_idx[j_ptr];

	while(i_ptr < vec1_nnz && j_ptr < vec2_nnz) {

		if(i < j) {
			i = vec1_idx[++i_ptr];
		} else if(j < i) {
			j = vec2_idx[++j_ptr];
		} else {
			res += 1;

			i = vec1_idx[++i_ptr];
			j = vec2_idx[++j_ptr];
		}
	}

	return res;
}
