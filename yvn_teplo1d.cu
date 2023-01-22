#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <device_functions.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>  

#define BLOCK_SIZE_X 30
#define a_const 1

__device__ __host__ double Kurant_condition(double h) {
	double ret = 1 / 2.0f * powf(h, 2) / (2.0f * powf(a_const, 2));
	return ret;
}

__device__ __host__ void swap(double* &c, double* &b) {
	double *temp = c;
	c = b;
	b = temp;
}

double phi(double x) {
	double elem = cos(x);
	return elem;

}

__device__ __host__ double stop_condition(double* arr1, double* arr2, int Na, double eps) {
	double flag = 1.0f;
	double maxx = -1.0f;
	for (int i = 0; i < Na; i++) {
		if (fabs(arr1[i] - arr2[i]) > eps)
			flag = 0.0f;
		if (fabs(arr1[i] - arr2[i]) > maxx)
			maxx = fabs(arr1[i] - arr2[i]);
	}
	printf("max = %e \n", maxx);
	return flag;
}

__global__ void stop_condition_gpu(double* arr1, double* arr2, int Na, double* flag, double eps) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (fabs(arr1[i] - arr2[i]) > eps)
		flag[0] = 0.0f;
}

void cpu_var(double* prev, double* next, double r, double Nt, double Nx, double eps) {
	for (int t_i = 0; t_i < Nt; t_i++) {
		for (int x_i = 1; x_i < Nx - 1; x_i++) {
			next[x_i] = r * (prev[x_i - 1] - 2.0f*prev[x_i] + prev[x_i + 1]) + prev[x_i];
		}
		swap(next, prev);
	}
}

__global__ void gpu_var(double* dev_prev, double* dev_next, double r, double Nx) {
	int x_i = threadIdx.x + blockIdx.x*blockDim.x;
	if (x_i>0 && x_i< Nx-1)
		dev_next[x_i] = r * (dev_prev[x_i - 1] - 2.0f*dev_prev[x_i] + dev_prev[x_i + 1]) + dev_prev[x_i];
}

__global__ void gpu_var_shared(double* prev, double* next, double r, double Nx) {
	int x_i = threadIdx.x + blockIdx.x*blockDim.x;
	int s_x_i = threadIdx.x + 1;
	__shared__ double shared_block[BLOCK_SIZE_X + 2];
	shared_block[s_x_i] = prev[x_i];
	shared_block[0] = prev[x_i - 1];
	shared_block[BLOCK_SIZE_X + 1] = prev[x_i + 1];

	if (x_i > 0 && x_i < Nx - 1)
		next[x_i] = r * (shared_block[s_x_i - 1] - 2.0f*shared_block[s_x_i] + shared_block[s_x_i + 1]) + shared_block[s_x_i];
}


int main(int argc, char **argv) {

	double eps = 0.00000008;
	double xmin = -50;
	double xmax = 50;
	double t0 = 0;
	double T = 1;
	int Nx = 132*4;
	double h = (xmax-xmin) / ((double)Nx - 1);
	double tau = Kurant_condition(h);
	int Nt = (int)((1 + tau) / tau);
	double r = powf(a_const, 2)*tau / powf(h, 2);
	double* x = (double*)malloc(sizeof(double) * Nx);
	double* prev = (double*)malloc(sizeof(double) * Nx);
	double* next = (double*)malloc(sizeof(double) * Nx);
	double* prev1 = (double*)malloc(sizeof(double) * Nx);
	double* next1 = (double*)malloc(sizeof(double) * Nx);
	double* prev_cpu = (double*)malloc(sizeof(double) * Nx);
	double* next_cpu = (double*)malloc(sizeof(double) * Nx);
	
	for (int i = 0; i < Nx; i++) {
		x[i] = xmin + h * i;
	}
	for (int i = 0; i < Nx; i++) {
		prev[i] = phi(x[i]);
		next[i] = phi(x[i]);
		prev1[i] = phi(x[i]);
		next1[i] = phi(x[i]);
		prev_cpu[i] = phi(x[i]);
		next_cpu[i] = phi(x[i]);
	}

	double flag[1] = { 1.0f };
	double *dev_prev, *dev_next, *dev_prev1, *dev_next1, *dev_flag;
	cudaMalloc((void**)&dev_flag, sizeof(double));
	cudaMalloc((void**)&dev_prev, Nx * sizeof(double));
	cudaMalloc((void**)&dev_next, Nx * sizeof(double));
	cudaMalloc((void**)&dev_prev1, Nx * sizeof(double));
	cudaMalloc((void**)&dev_next1, Nx * sizeof(double));
	dim3 numBlocks(Nx / BLOCK_SIZE_X + 1);
	dim3 threadsPerBlock(BLOCK_SIZE_X);

	double time_spent = 0.0;
	clock_t begin1 = clock();
	cpu_var(prev_cpu, next_cpu, r, Nt, Nx, eps);
	clock_t end1 = clock();
	time_spent += (double)(end1 - begin1) / CLOCKS_PER_SEC;
	printf("The elapsed time is %f seconds \n", time_spent);


	begin1 = clock();
	cudaMemcpy(dev_prev, prev, Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_next, next, Nx * sizeof(double), cudaMemcpyHostToDevice);
	time_spent = 0.0;
	for (int t_i = 0; t_i < Nt; t_i++) {
		gpu_var << <numBlocks, threadsPerBlock >> > (dev_prev, dev_next, r, Nx);
		swap(dev_prev, dev_next);
	}
	cudaMemcpy(prev, dev_prev, Nx * sizeof(double), cudaMemcpyDeviceToHost);
	end1 = clock();
	time_spent += (double)(end1 - begin1) / CLOCKS_PER_SEC;
	printf("The elapsed time is %f seconds \n", time_spent);
	

	time_spent = 0.0;
	begin1 = clock();
	cudaMemcpy(dev_prev1, prev1, Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_next1, next1, Nx * sizeof(double), cudaMemcpyHostToDevice);
	for (int t_i = 0; t_i < Nt; t_i++) {
		gpu_var_shared <<<numBlocks, threadsPerBlock >> > (dev_prev1, dev_next1, r, Nx);
		swap(dev_prev, dev_next);
	}
	cudaMemcpy(prev1, dev_prev1, Nx * sizeof(double), cudaMemcpyDeviceToHost);
	end1 = clock();
	time_spent += (double)(end1 - begin1) / CLOCKS_PER_SEC;
	printf("The elapsed time is %f seconds \n", time_spent);
}