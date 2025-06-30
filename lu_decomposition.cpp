#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h> 
#include <algorithm> 


using namespace std;

// --- Configuration ---
const int N = 1024;    // Matrix size 
const int BS = 64;     // Block size


vector<vector<double>> A(N, vector<double>(N));
vector<vector<double>> L(N, vector<double>(N));
vector<vector<double>> U(N, vector<double>(N));

/**
 * @brief Initializes the matrices for the LU decomposition.
 * 
 * Fills matrix A with values that ensure it is non-singular.
 * Resets L and U to zero.
 */
void initialize_matrices() {
    cout << "Initializing " << N << "x" << N << " matrices..." << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // A is filled with values to make it diagonally dominant, which guarantees
            // that the matrix is invertible and LU decomposition without pivoting is stable.
            if (i == j) {
                A[i][j] = N;
            } else {
                A[i][j] = 1.0;
            }
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }
}

/**
 * @brief Performs blocked LU decomposition.
 * 
 * This version is optimized for cache efficiency and parallelization.
 * The matrix is processed in smaller blocks of size BS x BS.
 */
void blocked_lu_parallel() {
    // Iterate over the diagonal blocks
    for (int k = 0; k < N; k += BS) {
        int kend = min(k + BS, N);

        // === Step 1: Factorize the diagonal block (A_kk) ===
        // This part is sequential because each calculation depends on the previous one.
        for (int i = k; i < kend; ++i) {
            // Calculate U for the current row within the block
            for (int j = i; j < kend; ++j) {
                double sum = 0;
                for (int s = k; s < i; ++s) {
                    sum += L[i][s] * U[s][j];
                }
                U[i][j] = A[i][j] - sum;
            }

            // Calculate L for the current column within the block
            L[i][i] = 1.0; // Diagonal of L is always 1
            for (int j = i + 1; j < kend; ++j) {
                double sum = 0;
                for (int s = k; s < i; ++s) {
                    sum += L[j][s] * U[s][i];
                }
                L[j][i] = (A[j][i] - sum) / U[i][i];
            }
        }

        // === Step 2 & 3: Solve for U_k* and L_*k (the row and column of blocks) ===
        // These two loops are independent and can be parallelized.
        #pragma omp parallel for
        for (int j = kend; j < N; ++j) { // Update U blocks to the right
            for (int i = k; i < kend; ++i) {
                double sum = 0;
                for (int s = k; s < i; ++s) {
                    sum += L[i][s] * U[s][j];
                }
                U[i][j] = A[i][j] - sum;
            }
        }

        #pragma omp parallel for
        for (int i = kend; i < N; ++i) { // Update L blocks below
            for (int j = k; j < kend; ++j) {
                double sum = 0;
                for (int s = k; s < j; ++s) {
                    sum += L[i][s] * U[s][j];
                }
                L[i][j] = (A[i][j] - sum) / U[j][j];
            }
        }

        // === Step 4: Update the trailing submatrix ===
        // This is the most computationally intensive part and benefits most from parallelization.
        // The 'collapse(2)' clause tells OpenMP to parallelize the outer two loops together.
        #pragma omp parallel for collapse(2)
        for (int i = kend; i < N; ++i) {
            for (int j = kend; j < N; ++j) {
                double sum = 0;
                for (int s = k; s < kend; ++s) {
                    sum += L[i][s] * U[s][j];
                }
                A[i][j] -= sum;
            }
        }
    }
}

int main() {
    cout << "=================================================" << endl;
    cout << "      LU Decomposition Performance Comparison      " << endl;
    cout << "=================================================" << endl;
    cout << "CPU: AMD Ryzen 5 2600 (12 Threads)" << endl;
    cout << "Matrix Size (N): " << N << endl;
    cout << "Block Size (BS): " << BS << endl;
    cout << "-------------------------------------------------" << endl;

    // --- Single-Threaded Execution ---
    initialize_matrices();
    cout << "\nRunning Blocked LU with 1 thread..." << endl;
    omp_set_num_threads(1); // Force OpenMP to use only one thread
    
    double start_single = omp_get_wtime();
    blocked_lu_parallel();
    double end_single = omp_get_wtime();
    
    double time_single = (end_single - start_single) * 1000.0;
    cout << "Single-Threaded Time: " << time_single << " ms" << endl;

    cout << "-------------------------------------------------" << endl;

    // --- Multi-Threaded Execution ---
    initialize_matrices(); // Reset matrices
    int num_threads = 12; // For AMD Ryzen 5 2600 (6 cores, 12 threads)
    cout << "\nRunning Blocked LU with " << num_threads << " threads..." << endl;
    omp_set_num_threads(num_threads); // Use all available threads

    double start_multi = omp_get_wtime();
    blocked_lu_parallel();
    double end_multi = omp_get_wtime();

    double time_multi = (end_multi - start_multi) * 1000.0;
    cout << "Multi-Threaded Time: " << time_multi << " ms" << endl;

    // --- Final Comparison ---
    cout << "=================================================" << endl;
    cout.precision(2);
    cout << fixed << "Speedup: " << (time_single / time_multi) << "x" << endl;
    cout << "=================================================" << endl;

    return 0;
}
