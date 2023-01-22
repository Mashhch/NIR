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

#define N (32*32)
#define BLOCK_SIZE 30

__constant__ double constants[8];

__device__ __host__ double f(double x, double t) {
	return 2 * x*t + (1 + tanh(x - t) - 2 * powf(tanh(x - t), 2)) / cosh(x - t);
}

__host__ __device__ double u0(double x, double t) {
	return 1 / cosh(x - t) + x * powf(t, 2);
}

__host__ __device__ double gamma1(double t) {
	return powf(t, 2) + (1 + tanh(t)) / cosh(t);
}

__host__ __device__ double gamma2(double t) {
	return powf(t, 2) + 1 / cosh(1 - t);
}

__host__ __device__ double fi(double x) {
	return 1 / cosh(x);
}


__global__ void method_progonki(double* a, double* b, double* c, double* d, double* y, int n) {
	double *A, *B;
	A = (double*)malloc(sizeof(double)*n);
	B = (double*)malloc(sizeof(double)*n);

	A[0] = -c[0] / b[0];
	B[0] = d[0] / b[0];

	for (int i = 1; i < n - 1; i++) {
		A[i] = -c[i] / (b[i] + a[i] * A[i - 1]);
		A[n - 1] = 0;
	}
	for (int i = 1; i < n; i++) {
		B[i] = (d[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1]);
	}
	y[n - 1] = B[n - 1];
	for (int i = n - 2; i >= 0; i--) {
		y[i] = B[i] + A[i] * y[i + 1];
	}

}

__global__ void next_2_ord(double *a, double *b, double *c, double *d, double* prev, double t, double* x, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int num = threadIdx.x;
	__shared__ double prev_shared[BLOCK_SIZE + 2];
	int start_block = blockIdx.x * blockDim.x;
	int end_block = (blockIdx.x + 1) * blockDim.x;
	prev_shared[num+1] = prev[i];
	prev_shared[0] = prev[start_block - 1];
	prev_shared[BLOCK_SIZE + 1] = prev[end_block];
	

	__syncthreads();
	if (i < n - 1 && i>0) {
		d[i] = -prev_shared[num+1] - constants[7] * f(x[i], (t - 0.5) * constants[7]) + (constants[6] - 1) * (constants[7] * powf(constants[4], 2) / powf(constants[5], 2)) * (prev_shared[num + 2] - 2 * prev_shared[num+1] + prev_shared[num]);
	}
	if (i == 0) {
		if (constants[0] == 0) {
			d[0] = gamma1(t*constants[7]);
		}
		else
		{
			d[0] = prev[0] + powf(constants[4], 2) * constants[7] / (2 * powf(constants[5], 2))*(-gamma1(t*constants[7]) * 2 * constants[5] / constants[0] + prev[1] - 2 * prev[0] + prev[1]
				- (gamma1((t - 1)*constants[7]) - constants[2] * prev[0]) * 2 * constants[5] / constants[0]) + constants[7] * f(x[0], (t - 0.5)*constants[7]);
		}
	}
	if (i == n - 1) {
		if (constants[1] == 0) {
			d[n - 1] = gamma2(t*constants[7]);
		}
		else {
			d[n - 1] = prev[0] + powf(constants[4], 2) * constants[7] / (2 * powf(constants[5], 2))*(gamma2(t*constants[7]) * 2 * constants[5] / constants[1] + prev[n - 2] - 2 * prev[n - 1] + prev[n - 2]
				+ (gamma2((t - 1)*constants[7]) - constants[3] * prev[0]) * 2 * constants[5] / constants[1]) + constants[7] * f(x[n - 1], (t - 0.5)*constants[7]);
		}

	}
}

__global__ void abc(double *a, double *b, double *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n - 1 && i>0) {
		a[i] = constants[7] * powf(constants[4], 2) * constants[6] / powf(constants[5], 2);
		b[i] = -1 - 2 * constants[7] * powf(constants[4], 2) * constants[6] / powf(constants[5], 2);
		c[i] = constants[7] * powf(constants[4], 2) * constants[6] / powf(constants[5], 2);
	}

	if (i == 0) {
		if (constants[0] == 0) {
			b[0] = constants[2];
		}
		else
		{
			b[0] = 1 - powf(constants[4], 2) * constants[7] / (powf(constants[5], 2) * 2) * (-2 + constants[2] * 2 * constants[5] / constants[0]);
			c[0] = -powf(constants[4], 2) * constants[7] / powf(constants[5], 2);
		}
	}
	if (i == n - 1) {
		if (constants[1] == 0) {
			b[n - 1] = constants[3];
		}
		else {
			b[n - 1] = 1 - powf(constants[4], 2) * constants[7] / (powf(constants[5], 2) * 2)*(-2 - constants[3] * 2 * constants[5] / constants[1]);
			a[n - 1] = -powf(constants[4], 2) * constants[7] / powf(constants[5], 2);
		}
	}
}


int main()
{
	double *a, *b, *c, *d, *prev, *dev_a, *dev_b, *dev_c, *dev_d, *dev_prev, *x, *dev_x, *t, dev_t;

	unsigned int mem_size = sizeof(double)*N;

	double alpha[2] = { 1, 0 };
	double beta[2] = { 1, 1 };
	double a_const = 1;

	a = (double*)malloc(mem_size);
	b = (double*)malloc(mem_size);
	c = (double*)malloc(mem_size);
	d = (double*)malloc(mem_size);
	prev = (double*)malloc(sizeof(double) * N);

	cudaMalloc((void**)&dev_a, mem_size);
	cudaMalloc((void**)&dev_b, mem_size);
	cudaMalloc((void**)&dev_c, mem_size);
	cudaMalloc((void**)&dev_d, mem_size);
	cudaMalloc((void**)&dev_prev, mem_size);



	double x_left = 0;
	double x_right = 1;
	double t0 = 0;
	double T = 1;
	double sigma = 0.5;
	int Nx = N;
	int Nt = 201;
	double h = (double)1 / (Nx - 1);
	double tau = (double)1 / (Nt - 1);
	int x_size = sizeof(double) * Nx;
	x = (double*)malloc(x_size);
	t = (double*)malloc(sizeof(double) * Nt);
	double* u0_ = (double*)malloc(sizeof(double) * Nx);

	cudaMalloc((void**)&dev_t, sizeof(double) * Nt);
	cudaMalloc((void**)&dev_x, x_size);

	double* next_2 = (double*)malloc(sizeof(double) * Nx);
	double* errors_ = (double*)malloc(sizeof(double) * Nx);

	double myconst[8];
	myconst[0] = alpha[0];
	myconst[1] = alpha[1];
	myconst[2] = beta[0];
	myconst[3] = beta[1];
	myconst[4] = a_const;
	myconst[5] = h;
	myconst[6] = sigma;
	myconst[7] = tau;
	cudaMemcpyToSymbol(constants, myconst, 8*sizeof(double));

	for (int i = 0; i < Nx; i++) {
		a[i] = 0;
		b[i] = 0;
		c[i] = 0;
		d[i] = 0;
	}



	for (int i = 0; i < Nx; i++) {
		x[i] = x_left + i * h;
	}
	for (int i = 0; i < Nt; i++) {
		t[i] = t0 + i * tau;
	}
	for (int i = 0; i < Nx; i++) {
		u0_[i] = u0(x[i], t[Nt - 1]);
		prev[i] = fi(x[i]);
		//printf("%f \n", x[i]);dev_prev
		//printf("%f \n", t[Nt - 1]);
		//printf("%d \n", u0_[i]);
	}

	double time_spent = 0.0;
	clock_t begin1 = clock();
	cudaMemcpy(dev_x, x, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, a, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d, d, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_prev, prev, mem_size, cudaMemcpyHostToDevice);

	dim3 numBlocks(Nx / BLOCK_SIZE + 1);
	dim3 threadsPerBlock(BLOCK_SIZE);
	abc << <numBlocks, threadsPerBlock >> > (dev_a, dev_b, dev_c, N);
	for (int i = 1; i < Nt; i++) {
		next_2_ord << <numBlocks, threadsPerBlock >> > (dev_a, dev_b, dev_c, dev_d, dev_prev, i, dev_x, N);
		method_progonki << <1, 1>> > (dev_a, dev_b, dev_c, dev_d, dev_prev, N);	
	}

	cudaMemcpy(prev, dev_prev, mem_size, cudaMemcpyDeviceToHost);


	clock_t end1 = clock();

	// рассчитать прошедшее время, найдя разницу (end - begin) и
	// деление разницы на CLOCKS_PER_SEC для перевода в секунды
	time_spent += (double)(end1 - begin1) / CLOCKS_PER_SEC;

	printf("The elapsed time is %f seconds \n", time_spent);
	return 0;
}