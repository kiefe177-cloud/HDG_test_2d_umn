#include "hdg/sparse_utils.h"
#include <cmath>

void prune(Eigen::SparseMatrix<double>& M, double tol) {
    M.prune([tol](int /*row*/, int /*col*/, double value) {
        return std::abs(value) > tol;
    });
}

void prune(Eigen::SparseMatrix<std::complex<double>>& M, double tol) {
    M.prune([tol](int /*row*/, int /*col*/, std::complex<double> value) {
        return std::abs(value) > tol;
    });
}