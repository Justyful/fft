#include <complex>
#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iomanip>

#define TWOPI 6.2831853071795864769252867

using std::cin;
using std::cout;
using std::endl;
using std::complex;

void DFT(complex<double>* const x, complex<double>* ans, int const size) {
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

void DFT(complex<double>*& x, int const size) {
	complex<double>* ans = new complex<double>[size];
	complex<double> exp;
	double a = -TWOPI / size;
	for (int i = 0; i < size; i++) {
		ans[i] = complex<double>(0, 0);
		for (int j = 0; j < size; j++) {
			exp = std::polar<double>(1, 1.0 * a * i * j);
			ans[i] += x[j] * exp;
		}
	}
	delete[] x;
	x = ans;
}

int stop_rec = 1;

void FFT(complex<double>*& x, int const size) {
	if (size == stop_rec) {
		complex<double>* help = new complex<double>[size];
		for (int i = 0; i < size; i++) {
			help[i] = x[i];
		}
		complex<double> exp;
		double a = -TWOPI / size;
		for (int i = 0; i < size; i++) {
			x[i] = complex<double>(0, 0);
			for (int j = 0; j < size; j++) {
				exp = std::polar<double>(1, 1.0 * a * i * j);
				x[i] += help[j] * exp;
			}
		}
		return;
	}

	int halfsize = size / 2;
	complex<double>* even = new complex<double>[halfsize];
	complex<double>* odd = new complex<double>[halfsize];

	for (int i = 0; i < halfsize; i++) {
		even[i] = x[2*i];
		odd[i] = x[2*i + 1];
	}
#pragma omp task 
	FFT(even, halfsize);
#pragma omp task 
	FFT(odd, halfsize);
#pragma omp taskwait
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

void fill_vec(complex<double>* x, int size) {
	std::srand(std::time(0));
	for (int i = 0; i < size; i++) {
		x[i] = std::complex<double>(std::rand() / 1000000.0, std::rand() / 1000000.0);
	}
}

int main() {
	// int N;
	int n_threads;
	// cout << "size = ";
	// cin >> N;
	// cout << "n_threads = ";
	// cin >> n_threads;
	// cout << "stop_iter = ";
	// cin >> stop_rec;

	std::ofstream file;
	file.open("data_fft_omp.txt");
	// 2^10 ... 2^27 = 134217728
	file << "SIZE       |DFT_TIME   |FFT_TIME   |L2_ERROR    |C_ERROR" << endl;
	for (int N = 1024; N < 134217729; N *= 2) {
		stop_rec = 32;
		n_threads = 2000;
		complex<double>* x = new complex<double>[N];
		complex<double>* y = new complex<double>[N];

		fill_vec(x, N);
		auto start = std::chrono::high_resolution_clock::now();
		if (N < 65537) {
			DFT(x, y, N);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration1 = end - start;

		start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(n_threads)
{
#pragma omp single
		FFT(x, N);
}
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration2 = end - start;

		complex<double> diff;
		double l2_error = 0.0;
		double c_error = 0.0;
		for (int i = 0; i < N; i++) {
			diff = x[i] - y[i];
			if (std::norm(diff) > c_error) {
				c_error = std::norm(diff);
			}
			l2_error += std::norm(diff);
		}
		if (N > 65536) {
			l2_error = 0;
			c_error = 0;
			duration1 = std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now();
		}
		file << std::setw(10) << std::left << N << " |";
		file << std::setw(10) << std::left << duration1.count() << " |";
		file << std::setw(10) << std::left << duration2.count()<< " |";
		file << std::setw(11) << std::left << sqrt(l2_error) << " |" ;
		file << std::setw(10) << std::left << c_error << endl;
		delete[] x;
		delete[] y;
	}
	file << endl << "OMP: n_threads = 2000, stop = 32" << endl;
	file.close();
	return 0;
}









