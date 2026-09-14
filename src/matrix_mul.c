#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "../include/matrix_mul.h"

Matrix *create_matrix(size_t rows, size_t cols) {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    if (!mat) {
        perror("Failed to allocate Matrix struct");
        return NULL;
    }
    mat->rows = rows;
    mat->cols = cols;

    // 64-byte aligned allocation for SIMD efficiency
    size_t size = rows * cols * sizeof(double_t_mat);
    int ret = posix_memalign((void **)&(mat->data), 64, size);
    if (ret != 0 || !mat->data) {
        perror("Failed to allocate matrix memory");
        free(mat);
        return NULL;
    }

    return mat;
}

void free_matrix(Matrix *mat) {
    if (mat) {
        if (mat->data) {
            free(mat->data);
        }
        free(mat);
    }
}

void init_matrix_random(Matrix *mat, double min_val, double max_val) {
    if (!mat || !mat->data) return;
    size_t total = mat->rows * mat->cols;
    double range = max_val - min_val;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] = min_val + ((double)rand() / (double)RAND_MAX) * range;
    }
}

void init_matrix_identity(Matrix *mat) {
    if (!mat || !mat->data) return;
    size_t rows = mat->rows;
    size_t cols = mat->cols;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat->data[i * cols + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void init_matrix_constant(Matrix *mat, double val) {
    if (!mat || !mat->data) return;
    size_t total = mat->rows * mat->cols;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] = val;
    }
}

// Standard naive O(N^3) Matrix Multiplication (i-j-k loop order)
int matrix_multiply_naive(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        return -1; // Invalid dimensions
    }

    size_t N = A->rows;
    size_t K = A->cols;
    size_t M = B->cols;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            double_t_mat sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += A->data[i * K + k] * B->data[k * M + j];
            }
            C->data[i * M + j] = sum;
        }
    }
    return 0;
}

// Cache-optimized Matrix Multiplication (i-k-j loop order + tiling)
int matrix_multiply_optimized(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        return -1;
    }

    size_t N = A->rows;
    size_t K = A->cols;
    size_t M = B->cols;

    // Initialize C to 0
    for (size_t i = 0; i < N * M; i++) {
        C->data[i] = 0.0;
    }

    // Tiling / Block size for L1/L2 cache
    const size_t BLOCK_SIZE = 64;

    for (size_t ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (size_t jj = 0; jj < M; jj += BLOCK_SIZE) {

                size_t i_end = (ii + BLOCK_SIZE < N) ? ii + BLOCK_SIZE : N;
                size_t k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;
                size_t j_end = (jj + BLOCK_SIZE < M) ? jj + BLOCK_SIZE : M;

                for (size_t i = ii; i < i_end; i++) {
                    for (size_t k = kk; k < k_end; k++) {
                        double_t_mat r_a = A->data[i * K + k];
                        size_t c_idx = i * M;
                        size_t b_idx = k * M;
                        for (size_t j = jj; j < j_end; j++) {
                            C->data[c_idx + j] += r_a * B->data[b_idx + j];
                        }
                    }
                }

            }
        }
    }

    return 0;
}

// Verify output C against reference calculation on sample points / full matrix
bool verify_matrix_multiply(const Matrix *A, const Matrix *B, const Matrix *C, double tolerance) {
    if (!A || !B || !C) return false;

    size_t N = A->rows;
    size_t K = A->cols;
    size_t M = B->cols;

    // Sample check at representative elements (corners, center, random points)
    size_t sample_indices[][2] = {
        {0, 0}, {0, M - 1}, {N - 1, 0}, {N - 1, M - 1},
        {N / 2, M / 2}, {N / 4, M / 3}, {3 * N / 4, 2 * M / 3}
    };
    size_t num_samples = sizeof(sample_indices) / sizeof(sample_indices[0]);

    for (size_t idx = 0; idx < num_samples; idx++) {
        size_t i = sample_indices[idx][0];
        size_t j = sample_indices[idx][1];

        double_t_mat expected = 0.0;
        for (size_t k = 0; k < K; k++) {
            expected += A->data[i * K + k] * B->data[k * M + j];
        }

        double_t_mat actual = C->data[i * M + j];
        double diff = fabs(actual - expected);

        if (diff > tolerance) {
            fprintf(stderr, "Verification failed at C[%zu][%zu]: Expected %f, Actual %f (Diff: %e)\n",
                    i, j, expected, actual, diff);
            return false;
        }
    }

    return true;
}

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
