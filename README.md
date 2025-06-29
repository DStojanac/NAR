# Blocked LU Decomposition with OpenMP

This C++ program performs a blocked LU decomposition of a matrix and benchmarks the performance of a parallel implementation using OpenMP.

It compares the execution time of the algorithm running on a single thread versus multiple threads to demonstrate the speedup achieved through parallelization.

## Requirements

- A C++ compiler that supports OpenMP (e.g., GCC/g++).

## How to Compile

To compile the program, you need to enable OpenMP support by adding the `-fopenmp` flag. For best performance, it's also recommended to use optimization flags like `-O3`.

Open a terminal and run the following command:

```bash
g++ -o lu_decomposition.exe lu_decomposition.cpp -fopenmp -O3
```

## How to Run

After successful compilation, an executable file named `lu_decomposition.exe` will be created. To run the benchmark, simply execute this file from your terminal:

```bash
./lu_decomposition.exe
```

Or on Windows Command Prompt:

```cmd
lu_decomposition.exe
```

### Example Output

The program will print the time taken for both single-threaded and multi-threaded execution, along with the calculated speedup.

```
=================================================
      LU Decomposition Performance Comparison
=================================================
CPU: AMD Ryzen 5 2600 (12 Threads)
Matrix Size (N): 1024
Block Size (BS): 64
-------------------------------------------------

Initializing 1024x1024 matrices...

Running Blocked LU with 1 thread...
Single-Threaded Time: 4170 ms
-------------------------------------------------

Initializing 1024x1024 matrices...

Running Blocked LU with 12 threads...
Multi-Threaded Time: 605 ms
=================================================
Speedup: 6.89x
=================================================
```
