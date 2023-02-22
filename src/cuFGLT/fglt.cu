#include "fglt.cuh"

#define NUMBLOCKS 512
#define NUMTHREADS 32

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


__global__ static void raw2net(
		const double * const d_f0,
		const double * const d_f1,
		const double * const d_f2,
		const double * const d_f3,
		const double * const d_f4,
		double * const d_fn0,
		double * const d_fn1,
		double * const d_fn2,
		double * const d_fn3,
		double * const d_fn4,
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
		double * const f,
		double * const fn) {

	/* extract matrix information */

	const int n_cols = adj->n_cols;
	const int n_nz = adj->n_nz;

	int const * const row_idx = adj->row_idx;
	int const * const col_ptr = adj->col_ptr;


	/* Allocate and transfer data from host to device */

	double *d_f_base; 
	double *d_fn_base;
	int d_pitch;

	// allocate memory for raw and net frequencies on device
	cudaMallocPitch((void **) &d_f_base, (size_t *) &d_pitch, n_cols * sizeof(double), NGRAPHLET);
	cudaMallocPitch((void **) &d_fn_base, (size_t *) &d_pitch, n_cols * sizeof(double), NGRAPHLET);

	// create pointers to the frequency vectors
	double *d_f[NGRAPHLET];
	double *d_fn[NGRAPHLET];
	for(int k = 0 ; k < NGRAPHLET ; k++) {
		d_f[k] = (double *)((char*)d_f_base + k * d_pitch);
		d_fn[k] = (double *)((char*)d_fn_base + k * d_pitch);
	}

	int *d_row_idx;
	int *d_col_ptr;

	// allocate memory for the matrix arrays
	cudaMalloc((void **) &d_row_idx, n_nz * sizeof(int));
	cudaMalloc((void **) &d_col_ptr, (n_cols + 1) * sizeof(int));

	// transfer matrix data from host to device
	cudaMemcpy(d_row_idx, row_idx, n_nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_ptr, col_ptr, (n_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);


	/* Compute raw frequencies */

	// fill f0 with 1
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_f[0], n_cols);

	// compute f1
	compute_d1<<<NUMBLOCKS, NUMTHREADS>>>(d_f[1], d_col_ptr, n_cols);

	// compute f2
	compute_d2<<<NUMBLOCKS, NUMTHREADS>>>(d_f[2], d_row_idx, d_col_ptr, d_f[1], n_cols);

	// compute f3
	compute_d3<<<NUMBLOCKS, NUMTHREADS>>>(d_f[3], d_f[1], n_cols);

	// compute f4
	compute_d4<<<NUMBLOCKS, NUMTHREADS>>>(d_f[4], d_row_idx, d_col_ptr, n_cols);

	/* Transform raw freq to net freq */

	raw2net<<<NUMBLOCKS, NUMTHREADS>>>( 
	    d_f[0],  d_f[1],  d_f[2],  d_f[3],  d_f[4], 
	    d_fn[0], d_fn[1], d_fn[2], d_fn[3], d_fn[4],
	    n_cols);

	/* Transfer data from device to host and free memory on device */

	cudaMemcpy2D(
			f, n_cols * sizeof(double), 
			d_f_base, d_pitch, 
			n_cols * sizeof(double), NGRAPHLET, 
			cudaMemcpyDeviceToHost);

	cudaMemcpy2D(
			fn, n_cols * sizeof(double), 
			d_fn_base, d_pitch, 
			n_cols * sizeof(double), NGRAPHLET, 
			cudaMemcpyDeviceToHost);

	cudaFree(d_f_base);
	cudaFree(d_fn_base);

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


__global__ static void raw2net(
		const double * const d_f0,
		const double * const d_f1,
		const double * const d_f2,
		const double * const d_f3,
		const double * const d_f4,
		double * const d_fn0,
		double * const d_fn1,
		double * const d_fn2,
		double * const d_fn3,
		double * const d_fn4,
		int n_cols) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < n_cols) {
		d_fn0[idx] = d_f0[idx];
		d_fn1[idx] = d_f1[idx];
		d_fn2[idx] = d_f2[idx] - 2 * d_f4[idx];
		d_fn3[idx] = d_f3[idx] - d_f4[idx];
		d_fn4[idx] = d_f4[idx];

		idx += blockDim.x;
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
