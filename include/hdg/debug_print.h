#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <fstream>

// ============================================================
// Single-element accessors (defined in debug_print.cpp)
// ============================================================
//
// These are non-inline so they exist as callable symbols in the binary,
// which lets LLDB invoke them via `expr dat(M, i, j)`.
int    dnnz(const Eigen::SparseMatrix<double>& M);
int    dnnz(const Eigen::SparseMatrix<std::complex<double>>& M);
int    drows(const Eigen::SparseMatrix<double>& M);
int    dcols(const Eigen::SparseMatrix<double>& M);
double dmaxabs(const Eigen::SparseMatrix<double>& M);
double dminabs(const Eigen::SparseMatrix<double>& M);
double dnorm(const Eigen::SparseMatrix<double>& M);
double dat(const Eigen::SparseMatrix<double>& M, int i, int j);
std::complex<double> dat(const Eigen::SparseMatrix<std::complex<double>>& M,
                         int i, int j);
double dat(const Eigen::MatrixXd& M, int i, int j);
double dat(const Eigen::VectorXd& v, int i);

// ============================================================
// Full dense-grid print
// ============================================================

inline void dprint(const Eigen::SparseMatrix<double>& M) {
    std::cout << "(" << M.rows() << "x" << M.cols()
              << ", nnz=" << M.nonZeros() << ")\n"
              << Eigen::MatrixXd(M) << std::endl;
}

inline void dprint(const Eigen::SparseMatrix<std::complex<double>>& M) {
    std::cout << "(" << M.rows() << "x" << M.cols()
              << ", nnz=" << M.nonZeros() << ")\n"
              << Eigen::MatrixXcd(M) << std::endl;
}

inline void dprint(const Eigen::MatrixXd& M) {
    std::cout << "(" << M.rows() << "x" << M.cols() << ")\n"
              << M << std::endl;
}

inline void dprint(const Eigen::MatrixXcd& M) {
    std::cout << "(" << M.rows() << "x" << M.cols() << ")\n"
              << M << std::endl;
}

inline void dprint(const Eigen::VectorXd& v) {
    std::cout << "(size " << v.size() << ")\n"
              << v.transpose() << std::endl;
}

inline void dprint(const Eigen::VectorXcd& v) {
    std::cout << "(size " << v.size() << ")\n"
              << v.transpose() << std::endl;
}

// ============================================================
// Sparse matrix as (row, col, value) triples — all nonzeros
// ============================================================

inline void dprint_nnz(const Eigen::SparseMatrix<double>& M) {
    std::cout << "(" << M.rows() << "x" << M.cols()
              << ", nnz=" << M.nonZeros() << ") nonzeros:\n";
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            std::cout << "  (" << it.row() << ", " << it.col()
                      << ") = " << it.value() << "\n";
        }
    }
    std::cout.flush();
}

inline void dprint_nnz(const Eigen::SparseMatrix<std::complex<double>>& M) {
    std::cout << "(" << M.rows() << "x" << M.cols()
              << ", nnz=" << M.nonZeros() << ") nonzeros:\n";
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, k);
             it; ++it)
        {
            const auto val = it.value();
            std::cout << "  (" << it.row() << ", " << it.col()
                      << ") = " << val.real() << " + " << val.imag() << "i\n";
        }
    }
    std::cout.flush();
}

// ============================================================
// Write to file (for matrices too big for stdout)
// ============================================================

inline void dwrite(const Eigen::SparseMatrix<double>& M, const char* path) {
    std::ofstream f(path);
    f << Eigen::MatrixXd(M) << std::endl;
}

inline void dwrite(const Eigen::SparseMatrix<std::complex<double>>& M,
                   const char* path)
{
    std::ofstream f(path);
    f << Eigen::MatrixXcd(M) << std::endl;
}

inline void dwrite(const Eigen::MatrixXd& M, const char* path) {
    std::ofstream f(path);
    f << M << std::endl;
}