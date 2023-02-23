#include "fglt.cuh"

#include <iostream>
#include <chrono>

#define NUMBLOCKS 512
#define NUMTHREADS 32

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

	// transfer matrix data from host to device

	auto memcpy_in_start = chrono::high_resolution_clock::now();
	cudaMemcpy(d_row_idx, row_idx, n_nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_ptr, col_ptr, (n_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
	auto memcpy_in_end = chrono::high_resolution_clock::now();

	auto memcpy_in_duration = chrono::duration_cast<chrono::milliseconds>(memcpy_in_end - memcpy_in_start);
	cout << "memcpy in time: " << memcpy_in_duration.count() << " msec" << endl;

	/* Compute raw frequencies */

	
	// fill d0
	auto d0_start = chrono::high_resolution_clock::now();
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_f[0], n_cols);
	cudaDeviceSynchronize();
	auto d0_end = chrono::high_resolution_clock::now();

	auto d0_duration = chrono::duration_cast<chrono::milliseconds>(d0_end - d0_start);

	// compute d1
	auto d1_start = chrono::high_resolution_clock::now();
	compute_d1<<<NUMBLOCKS, NUMTHREADS>>>(d_f[1], d_col_ptr, n_cols);
	cudaDeviceSynchronize();
	auto d1_end = chrono::high_resolution_clock::now();

	auto d1_duration = chrono::duration_cast<chrono::milliseconds>(d1_end - d1_start);
	
	// compute d2
	auto d2_start = chrono::high_resolution_clock::now();
	compute_d2<<<NUMBLOCKS, NUMTHREADS>>>(d_f[2], d_row_idx, d_col_ptr, d_f[1], n_cols);
	cudaDeviceSynchronize();
	auto d2_end = chrono::high_resolution_clock::now();

	auto d2_duration = chrono::duration_cast<chrono::milliseconds>(d2_end - d2_start);

	// compute d3
	auto d3_start = chrono::high_resolution_clock::now();
	compute_d3<<<NUMBLOCKS, NUMTHREADS>>>(d_f[3], d_f[1], n_cols);
	cudaDeviceSynchronize();
	auto d3_end = chrono::high_resolution_clock::now();

	auto d3_duration = chrono::duration_cast<chrono::milliseconds>(d3_end - d3_start);

	// compute d4
	auto d4_start = chrono::high_resolution_clock::now();
	compute_d4<<<NUMBLOCKS, NUMTHREADS>>>(d_f[4], d_row_idx, d_col_ptr, n_cols);
	cudaDeviceSynchronize();
	auto d4_end = chrono::high_resolution_clock::now();

	auto d4_duration = chrono::duration_cast<chrono::milliseconds>(d4_end - d4_start);

	auto raw_freq_duration = d0_duration + d1_duration + d2_duration + d3_duration + d4_duration;
	cout << "total raw frequence time: " << raw_freq_duration.count() << " msec" << endl;
	cout << "--- d0 time: " << d0_duration.count() << " msec" << endl;
	cout << "--- d1 time: " << d1_duration.count() << " msec" << endl;
	cout << "--- d2 time: " << d2_duration.count() << " msec" << endl;
	cout << "--- d3 time: " << d3_duration.count() << " msec" << endl;
	cout << "--- d4 time: " << d4_duration.count() << " msec" << endl;

	/* Transfer data from device to host and free memory on device */

	auto memcpy_out_start = chrono::high_resolution_clock::now();
	cudaMemcpy2D(
			f_base, n_cols * sizeof(double), 
			d_f_base, d_pitch, 
			n_cols * sizeof(double), NGRAPHLET, 
			cudaMemcpyDeviceToHost);
	auto memcpy_out_end = chrono::high_resolution_clock::now();

	auto memcpy_out_duration = chrono::duration_cast<chrono::milliseconds>(memcpy_out_end - memcpy_out_start);
	cout << "memcpy out time: " << memcpy_out_duration.count() << " msec" << endl;

	/* Transform raw freq to net freq */

	auto raw2net_start = chrono::high_resolution_clock::now();
	raw2net(f[0],  f[1],  f[2],  f[3],  f[4], 
	        fn[0], fn[1], fn[2], fn[3], fn[4],
	        n_cols);
	auto raw2net_end = chrono::high_resolution_clock::now();

	auto raw2net_duration = chrono::duration_cast<chrono::milliseconds>(raw2net_end - raw2net_start);
	cout << "raw2net time: " << raw2net_duration.count() << " msec" << endl;


	auto total_duration = memcpy_in_duration + raw_freq_duration + raw2net_duration + memcpy_out_duration;
	cout << endl << "TOTAL TIME: " << total_duration.count() << " msec" << endl << endl;

	cudaFree(d_f_base);

	cudaFree(d_row_idx);
	cudaFree(d_col_ptr);

	return 0;
}


__global__ static void fill_d0(
		double * const d_f0,
		int n_cols) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < n_cols) {
		d_f0[idx] = 1.0;
		idx += blockDim.x;
	}
}


__global__ static void compute_d1(
		double * const d_f1, 
		int const * const d_col_ptr,
		int n_cols) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < n_cols) {
		d_f1[idx] = (double)(d_col_ptr[idx + 1] - d_col_ptr[idx]);
		idx += blockDim.x;
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
		while(j < d_col_ptr[i + 1]) {
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
		d_f3[idx] = (double)(d_p1[idx] * (d_p1[idx] - 1)) / 2.0;
		idx += blockDim.x;
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

		int const * const ai = d_row_idx + d_col_ptr[i];
		int ai_nnz = d_col_ptr[i + 1] - d_col_ptr[i];

		int j_ptr = d_col_ptr[i] + tid;
		while(j_ptr < d_col_ptr[i + 1]) {
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
	while(i_ptr < vec1_nnz && j_ptr < vec2_nnz) {

		int i = vec1_idx[i_ptr];
		int j = vec2_idx[j_ptr];

		if(i < j) {
			i_ptr += 1;
		} else if(j < i) {
			j_ptr += 1;
		} else {
			res += 1;

			i_ptr += 1;
			j_ptr += 1;
		}
	}

	return res;
}
