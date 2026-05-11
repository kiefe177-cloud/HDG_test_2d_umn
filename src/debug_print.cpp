#include "hdg/debug_print.h"
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================
// Single-element accessors
// ============================================================

double dat(const Eigen::SparseMatrix<double>& M, int i, int j) {
    return M.coeff(i, j);
}

std::complex<double> dat(const Eigen::SparseMatrix<std::complex<double>>& M,
                         int i, int j)
{
    return M.coeff(i, j);
}

double dat(const Eigen::MatrixXd& M, int i, int j) {
    return M(i, j);
}

double dat(const Eigen::VectorXd& v, int i) {
    return v(i);
}

// ============================================================
// Stats / metadata — real
// ============================================================

int dnnz(const Eigen::SparseMatrix<double>& M) {
    return static_cast<int>(M.nonZeros());
}

int drows(const Eigen::SparseMatrix<double>& M) {
    return static_cast<int>(M.rows());
}

int dcols(const Eigen::SparseMatrix<double>& M) {
    return static_cast<int>(M.cols());
}

double dmaxabs(const Eigen::SparseMatrix<double>& M) {
    double mx = 0.0;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            mx = std::max(mx, std::abs(it.value()));
        }
    }
    return mx;
}

double dminabs(const Eigen::SparseMatrix<double>& M) {
    double mn = std::numeric_limits<double>::infinity();
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            const double a = std::abs(it.value());
            if (a > 0.0) mn = std::min(mn, a);
        }
    }
    return std::isinf(mn) ? 0.0 : mn;
}

double dnorm(const Eigen::SparseMatrix<double>& M) {
    return M.norm();   // Frobenius norm
}

// ============================================================
// Stats / metadata — complex
// ============================================================

int dnnz(const Eigen::SparseMatrix<std::complex<double>>& M) {
    return static_cast<int>(M.nonZeros());
}

int drows(const Eigen::SparseMatrix<std::complex<double>>& M) {
    return static_cast<int>(M.rows());
}

int dcols(const Eigen::SparseMatrix<std::complex<double>>& M) {
    return static_cast<int>(M.cols());
}

double dmaxabs(const Eigen::SparseMatrix<std::complex<double>>& M) {
    double mx = 0.0;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, k);
             it; ++it)
        {
            mx = std::max(mx, std::abs(it.value()));
        }
    }
    return mx;
}

double dminabs(const Eigen::SparseMatrix<std::complex<double>>& M) {
    double mn = std::numeric_limits<double>::infinity();
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, k);
             it; ++it)
        {
            const double a = std::abs(it.value());
            if (a > 0.0) mn = std::min(mn, a);
        }
    }
    return std::isinf(mn) ? 0.0 : mn;
}

double dnorm(const Eigen::SparseMatrix<std::complex<double>>& M) {
    return M.norm();   // Frobenius norm
}