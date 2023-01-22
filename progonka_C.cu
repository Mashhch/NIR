#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>  


float f(float x, float t) {
	return 2 * x*t + (1 + tanh(x - t) - 2 * powf(tanh(x - t), 2)) / cosh(x - t);
}

float u0(float x, float t) {
	return 1 / cosh(x - t) + x * powf(t, 2);
}

float gamma1(float t) {
	return powf(t, 2) + (1 + tanh(t)) / cosh(t);
}

float gamma2(float t) {
	return powf(t, 2) + 1 / cosh(1 - t);
}

float fi(float x) {
	return 1.0f / cosh(x);
}

float * alpha_func() {
	float alpha_[2] = { 1.0f, 0.0f };
	return alpha_;
}

float* beta_func() {
	float beta[2] = { 1.0f, 1.0f };
	return  beta;
}


float* method_progonki(float* a, float* b, float* c, float* d, int n) {
	float *A, *B, *y;
	A = (float*)malloc(sizeof(float)*n);
	B = (float*)malloc(sizeof(float)*n);
	y = (float*)malloc(sizeof(float)*n);

	A[0] = -c[0] / b[0];
	B[0] = d[0] / b[0];

	for (int i = 1; i < n - 1; i++) {
		A[i] = -c[i] / (b[i] + a[i] * A[i - 1]);
	}
	A[n - 1] = 0;
	for (int i = 1; i < n; i++) {
		B[i] = (d[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1]);
	}

	y[n - 1] = B[n - 1];
	for (int i = n - 2; i >= 0; i--) {
		y[i] = B[i] + A[i] * y[i + 1];
	}

	return y;
}


float* next_2_ord(float* prev, float tau, float sigma, float a_const, float t, float* x, float h, int n) {
	float alpha[2] = { 1, 0 };
	float beta[2] = { 1, 1 };

	float *a, *b, *c, *d;
	a = (float*)malloc(sizeof(float) * n);
	b = (float*)malloc(sizeof(float) * n);
	c = (float*)malloc(sizeof(float) * n);
	d = (float*)malloc(sizeof(float) * n);
	a[0] = 0;
	c[n - 1] = 0;

	if (alpha[0] == 0) {
		b[0] = beta[0];
		d[0] = gamma1(t*tau);
	}
	else
	{
		b[0] = 1 - powf(a_const, 2) * tau / (powf(h, 2) * 2) * (-2 + beta[0] * 2 * h / alpha[0]);
		c[0] = -powf(a_const, 2) * tau / powf(h, 2);
		d[0] = prev[0] + powf(a_const, 2) * tau / (2 * powf(h, 2))*(-gamma1(t*tau) * 2 * h / alpha[0] + prev[1] - 2 * prev[0] + prev[1]
			- (gamma1((t - 1)*tau) - beta[0] * prev[0]) * 2 * h / alpha[0]) + tau * f(x[0], (t - 0.5)*tau);
	}
	if (alpha[1] == 0) {
		a[n - 1] = 0;
		b[n - 1] = beta[1];
		d[n - 1] = gamma2(t*tau);
	}
	else {
		d[n - 1] = prev[0] + powf(a_const, 2) * tau / (2 * powf(h, 2))*(gamma2(t*tau) * 2 * h / alpha[1] + prev[n - 2] - 2 * prev[n - 1] + prev[n - 2]
			+ (gamma2((t - 1)*tau) - beta[1] * prev[0]) * 2 * h / alpha[1]) + tau * f(x[n - 1], (t - 0.5)*tau);
		b[n - 1] = 1 - powf(a_const, 2) * tau / (powf(h, 2) * 2)*(-2 - beta[1] * 2 * h / alpha[1]);
		a[n - 1] = -powf(a_const, 2) * tau / powf(h, 2);
	}
	for (int i = 1; i < n - 1; i++) {
		a[i] = tau * powf(a_const, 2) * sigma / powf(h, 2);
		b[i] = -1 - 2 * tau * powf(a_const, 2) * sigma / powf(h, 2);
		c[i] = tau * powf(a_const, 2) * sigma / powf(h, 2);
		d[i] = -prev[i] - tau * f(x[i], (t - 0.5) * tau) + (sigma - 1) * (tau * powf(a_const, 2) / powf(h, 2)) * (prev[i + 1] - 2.0f * prev[i] + prev[i - 1]);
	}

	float* ret = method_progonki(a, b, c, d, n);
	return ret;

}

void swap(float* &c, float* &b) {
	float *temp = c;
	c = b;
	b = temp;
}

int main()
{
	float x_left = 0;
	float x_right = 1;
	float a_const = 1;
	float t0 = 0;
	float T = 1;
	float sigma = 0.5;
	float tau = 0.005;

	int Nx = 32*32;
	float h = 1.0f / (Nx - 1);
	int Nt = 201;
	int x_size = sizeof(float) * Nx;
	float* x = (float*)malloc(x_size);
	//printf("lol %d", sizeof(x)/sizeof(x[0]));
	float* t = (float*)malloc(sizeof(float) * Nt);
	float* u0_ = (float*)malloc(sizeof(float) * Nx);
	float* prev_2 = (float*)malloc(sizeof(float) * Nx);
	float* next_2 = (float*)malloc(sizeof(float) * Nx);
	float* errors_ = (float*)malloc(sizeof(float) * Nx);

	for (int i = 0; i < Nx; i++) {
		x[i] = x_left + i * h;
	}
	for (int i = 0; i < Nt; i++) {
		t[i] = t0 + i * tau;
	}
	for (int i = 0; i < Nx; i++) {
		u0_[i] = u0(x[i], t[Nt - 1]);
		//printf("%f \n", x[i]);
		//printf("%f \n", t[Nt - 1]);
	}
	for (int i = 0; i < Nx; i++) {
		prev_2[i] = fi(x[i]);
		//printf("lol %f \n", prev_2[i]);
	}
	double time_spent = 0.0;
	clock_t begin1 = clock();
	for (int i = 1; i < Nt; i++) {
		next_2 = next_2_ord(prev_2, tau, sigma, a_const, i, x, h, Nx);
		swap(next_2, prev_2);
		int r = 0;
	}

	clock_t end1 = clock();

	// рассчитать прошедшее время, найдя разницу (end - begin) и
	// деление разницы на CLOCKS_PER_SEC для перевода в секунды
	time_spent += (double)(end1 - begin1) / CLOCKS_PER_SEC;

	printf("The elapsed time is %f seconds \n", time_spent);

	return 0;
}