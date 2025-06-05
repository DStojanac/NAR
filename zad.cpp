#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional> // For std::ref

using namespace std;
using namespace chrono;

typedef vector<vector<double>> Matrix;

Matrix generate_random_matrix(int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    Matrix A(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = dis(gen);
    return A;
}

void lu_decomposition_single(Matrix A, Matrix& L, Matrix& U) {
    int n = A.size();
    L = Matrix(n, vector<double>(n, 0));
    U = Matrix(n, vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int k = i; k < n; ++k) {
            double sum = 0;
            for (int j = 0; j < i; ++j)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }

        for (int k = i; k < n; ++k) {
            if (i == k)
                L[i][i] = 1;
            else {
                double sum = 0;
                for (int j = 0; j < i; ++j)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

// Funkcija za paralelnu obradu redova
void compute_U_row(Matrix& A, Matrix& L, Matrix& U, int i, int start, int end) {
    for (int k = start; k < end; ++k) {
        double sum = 0;
        for (int j = 0; j < i; ++j)
            sum += L[i][j] * U[j][k];
        U[i][k] = A[i][k] - sum;
    }
}

void compute_L_column(Matrix& A, Matrix& L, Matrix& U, int i, int start, int end) {
    for (int k = start; k < end; ++k) {
        if (i == k) {
            L[i][i] = 1;
        } else {
            double sum = 0;
            for (int j = 0; j < i; ++j)
                sum += L[k][j] * U[j][i];
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

void lu_decomposition_parallel(Matrix A, Matrix& L, Matrix& U, int num_threads) {
    int n = A.size();
    L = Matrix(n, vector<double>(n, 0));
    U = Matrix(n, vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        // --- Paralelno za U ---
        vector<std::thread> threads_U;
        int chunk = (n - i + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = i + t * chunk;
            int end = min(n, start + chunk);
            if (start < end)
                threads_U.emplace_back(compute_U_row, std::ref(A), std::ref(L), std::ref(U), i, start, end);
        }
        for (auto& th : threads_U)
            th.join();

        // --- Paralelno za L ---
        vector<std::thread> threads_L;
        for (int t = 0; t < num_threads; ++t) {
            int start = i + t * chunk;
            int end = min(n, start + chunk);
            if (start < end)
                threads_L.emplace_back(compute_L_column, std::ref(A), std::ref(L), std::ref(U), i, start, end);
        }
        for (auto& th : threads_L)
            th.join();
    }
}

void benchmark(int size, int num_threads) {
    cout << "Matrica " << size << "x" << size << endl;

    Matrix A = generate_random_matrix(size);
    Matrix L, U;

    auto start = high_resolution_clock::now();
    lu_decomposition_single(A, L, U);
    auto end = high_resolution_clock::now();
    auto duration_single = duration_cast<milliseconds>(end - start).count();
    cout << "  Jedna nit:     " << duration_single << " ms" << endl;

    start = high_resolution_clock::now();
    lu_decomposition_parallel(A, L, U, num_threads);
    end = high_resolution_clock::now();
    auto duration_multi = duration_cast<milliseconds>(end - start).count();
    cout << "  Vise niti (" << num_threads << "): " << duration_multi << " ms" << endl;

    cout << "-------------------------------------------" << endl;
}

int main() {
    int num_threads = 10; // Tvoj CPU
    vector<int> sizes = {100, 500, 750, 1000};

    for (int size : sizes)
        benchmark(size, num_threads);

    return 0;
}

// Kompajliraj sa: g++ zad.cpp -o zad -std=c++11