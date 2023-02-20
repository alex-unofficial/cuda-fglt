#include "fglt/fglt.cuh"

#define NUMBLOCKS 16
#define NUMTHREADS 128

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
		double const * const p1,
		int n_cols);

__global__ static void compute_d3(
		double * const d_f3,
		double const * const p1,
		int n_cols);

int cuFGLT::compute(
		sparse::CSC<double> const * const adj,
		double * const f,
		double * const fn) {


	/* extract matrix information */

	const int n_cols = adj->n_cols;
	const int n_nz = adj->n_nz;

	double const * const nz_values = adj->nz_values;
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

	double *d_nz_values;
	int *d_row_idx;
	int *d_col_ptr;

	// allocate memory for the matrix arrays
	cudaMalloc((void **) &d_nz_values, n_nz * sizeof(int));
	cudaMalloc((void **) &d_row_idx, n_nz * sizeof(int));
	cudaMalloc((void **) &d_col_ptr, (n_cols + 1) * sizeof(int));

	// transfer matrix data from host to device
	cudaMemcpy(d_nz_values, nz_values, n_nz * sizeof(int), cudaMemcpyHostToDevice);
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

	// TODO f3, f4
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_f[4], n_cols);


	/* Transform raw freq to net freq */

	// TODO raw2net
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_fn[0], n_cols);
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_fn[1], n_cols);
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_fn[2], n_cols);
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_fn[3], n_cols);
	fill_d0<<<NUMBLOCKS, NUMTHREADS>>>(d_fn[4], n_cols);


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
		double const * const p1,
		int n_cols) {

	int i = blockIdx.x;

	const int thread_num = blockDim.x;
	__shared__ double t_results[NUMTHREADS];

	while(i < n_cols) {
		int tid = threadIdx.x;
		double t_sum = 0;

		int j = d_col_ptr[i] + tid;
		while(j < d_col_ptr[i + 1]) {
			t_sum += p1[d_row_idx[j]];
			j += thread_num;
		}

		t_results[tid] = t_sum;

		__syncthreads();

		for(int offset = 1; offset < thread_num ; offset <<= 1) {
			int mod = offset << 1;

			if(tid % mod == 0 && (tid + offset) < thread_num) {
				t_results[tid] += t_results[tid + offset];
			}

			__syncthreads();
		}

		if(tid == 0) {
			d_f2[i] = (double)(t_results[0] - p1[i]);
		}

		i += gridDim.x;
	}
}

__global__ static void compute_d3(
		double * const d_f3,
		double const * const p1,
		int n_cols) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < n_cols) {
		d_f3[idx] = p1[idx] * (p1[idx] - 1) / 2;
		idx += blockDim.x;
	}
}
