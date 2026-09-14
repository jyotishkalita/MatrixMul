#include <stdio.h>
#include <stdlib.h>
#include "../include/matrix_mul.h"

#define MATRIX_SIZE 1024

int main(void) {
    printf("====================================================\n");
    printf("  N x N Matrix Multiplication Performance & Cache Analysis\n");
    printf("  Matrix Size: %d x %d (Double Precision)\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("====================================================\n\n");

    size_t N = MATRIX_SIZE;
    size_t total_elements = N * N;
    size_t memory_bytes = total_elements * sizeof(double_t_mat);

    printf("[1] Memory Footprint:\n");
    printf("    Per Matrix : %.2f MB\n", (double)memory_bytes / (1024.0 * 1024.0));
    printf("    Total (3x) : %.2f MB\n\n", (3.0 * (double)memory_bytes) / (1024.0 * 1024.0));

    Matrix *A = create_matrix(N, N);
    Matrix *B = create_matrix(N, N);
    Matrix *C_naive = create_matrix(N, N);
    Matrix *C_opt = create_matrix(N, N);

    if (!A || !B || !C_naive || !C_opt) {
        fprintf(stderr, "Error: Matrix allocation failed.\n");
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_naive);
        free_matrix(C_opt);
        return EXIT_FAILURE;
    }

    // Seed for reproducible random input
    srand(42);
    init_matrix_random(A, 0.0, 5.0);
    init_matrix_random(B, 0.0, 5.0);

    // Total floating point operations = 2 * N^3
    double flop = 2.0 * (double)N * (double)N * (double)N;

    // -------------------------------------------------------------
    // Benchmark 1: Naive (i-j-k) Algorithm (High Cache Miss Rate)
    // -------------------------------------------------------------
    printf("[2] Running Naive (i-j-k) Matrix Multiplication...\n");
    double start_naive = get_time_sec();
    matrix_multiply_naive(A, B, C_naive);
    double end_naive = get_time_sec();
    double time_naive = end_naive - start_naive;
    double gflops_naive = (flop / time_naive) / 1e9;
    printf("    Time Taken : %.4f seconds\n", time_naive);
    printf("    Performance: %.2f GFLOPS\n\n", gflops_naive);

    // -------------------------------------------------------------
    // Benchmark 2: Optimized (i-k-j + Tiled) Algorithm (Low Cache Misses)
    // -------------------------------------------------------------
    printf("[3] Running Optimized (i-k-j Tiled) Matrix Multiplication...\n");
    double start_opt = get_time_sec();
    matrix_multiply_optimized(A, B, C_opt);
    double end_opt = get_time_sec();
    double time_opt = end_opt - start_opt;
    double gflops_opt = (flop / time_opt) / 1e9;
    printf("    Time Taken : %.4f seconds\n", time_opt);
    printf("    Performance: %.2f GFLOPS\n\n", gflops_opt);

    // -------------------------------------------------------------
    // Comparison Summary & Speedup
    // -------------------------------------------------------------
    double speedup = time_naive / time_opt;
    printf("====================================================\n");
    printf("                 PERFORMANCE SUMMARY                \n");
    printf("====================================================\n");
    printf(" Algorithm         | Time (s) | GFLOPS  | Speedup  \n");
    printf("-------------------|----------|---------|----------\n");
    printf(" Naive (i-j-k)     | %8.4f | %7.2f |  1.00x   \n", time_naive, gflops_naive);
    printf(" Optimized (Tiled) | %8.4f | %7.2f | %6.2fx   \n", time_opt, gflops_opt, speedup);
    printf("====================================================\n\n");

    // Verification check
    printf("[4] Verifying Correctness of Results...\n");
    bool naive_ok = verify_matrix_multiply(A, B, C_naive, 1e-5);
    bool opt_ok = verify_matrix_multiply(A, B, C_opt, 1e-5);

    if (naive_ok && opt_ok) {
        printf("    [PASS] Both implementations produced identical & correct results!\n");
    } else {
        printf("    [FAIL] Verification mismatch detected!\n");
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C_naive);
    free_matrix(C_opt);

    return EXIT_SUCCESS;
}
