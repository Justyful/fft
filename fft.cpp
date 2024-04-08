#include <complex>
#include <iostream>

#define TWOPI 6.2831853071795864769252867

using std::cin;
using std::cout;
using std::endl;
using std::complex;

void DFT(complex<double>* x, complex<double>* ans, int size) {
	complex<double> exp;
	double a = -TWOPI / size;
	for (int i = 0; i < size; i++) {
		ans[i] = complex<double>(0, 0);
		for (int j = 0; j < size; j++) {
			exp = std::polar<double>(1, 1.0 * a * i * j);
			ans[i] += x[j] * exp;
		}
	}
}

void FFT(complex<double>* x, int size) {
	if (size == 1) {
		return;
	}

	int halfsize = size / 2;
	complex<double>* even = new complex<double>[halfsize];
	complex<double>* odd = new complex<double>[halfsize];

	for (int i = 0; i < halfsize; i++) {
		even[i] = x[2*i];
		odd[i] = x[2*i + 1];
	}

	FFT(even, halfsize);
	FFT(odd, halfsize);

	complex<double> w;
	double a = -TWOPI / size;
	for (int i = 0; i < halfsize; i++) {
		w = std::polar<double> (1, a * i);
		odd[i] *= w;
		x[i] = even[i] + odd[i];
		x[i + halfsize] = even[i] - odd[i];
	}
	delete[] even;
	delete[] odd;
}

void read_vec(complex<double>* x, int size) {
	float re, im;
	for (int i = 0; i < size; i++) {
		cin >> re >> im;
		x[i] = complex<double>(re, im);
	}
}

void print_vec(complex<double>* x, int size) {
	for (int i = 0; i < size; i++) {
		cout << x[i].real() << " + " << x[i].imag() << "i" << endl;
	}
	cout << endl;
}

int main() {
	int N;
	cin >> N;
	complex<double>* x = new complex<double>[N];
	complex<double>* y = new complex<double>[N];

	read_vec(x, N);
	print_vec(x, N);

	DFT(x, y, N);
	print_vec(y, N);

	FFT(x, N);
	print_vec(x, N);

	complex<double>* diff = new complex<double>[N];
	double error = 0.0;
	for (int i = 0; i < N; i++) {
		diff[i] = x[i] - y[i];
		error += std::norm(diff[i]);
	}
	cout << sqrt(error) << endl;

	delete[] x;
	delete[] y;
	delete[] diff;
	return 0;
}









