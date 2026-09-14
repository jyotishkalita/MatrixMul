#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include <stddef.h>
#include <stdbool.h>

// Matrix data type
typedef double double_t_mat;

typedef struct {
    size_t rows;
    size_t cols;
    double_t_mat *data;
} Matrix;

// Function prototypes
Matrix *create_matrix(size_t rows, size_t cols);
void free_matrix(Matrix *mat);

void init_matrix_random(Matrix *mat, double min_val, double max_val);
void init_matrix_identity(Matrix *mat);
void init_matrix_constant(Matrix *mat, double val);

// Matrix multiplication algorithms
int matrix_multiply_naive(const Matrix *A, const Matrix *B, Matrix *C);
int matrix_multiply_optimized(const Matrix *A, const Matrix *B, Matrix *C);

// Verification and Utility
bool verify_matrix_multiply(const Matrix *A, const Matrix *B, const Matrix *C, double tolerance);
double get_time_sec(void);

#endif // MATRIX_MUL_H
