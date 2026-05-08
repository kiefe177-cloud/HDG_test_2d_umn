#pragma once
#include <Eigen/Sparse>
#include <complex>

// Drop sparse-matrix entries with |value| <= tol (in place).
// Modifies M's storage; reduces nonZeros() count.
void prune(Eigen::SparseMatrix<double>& M, double tol);
void prune(Eigen::SparseMatrix<std::complex<double>>& M, double tol);