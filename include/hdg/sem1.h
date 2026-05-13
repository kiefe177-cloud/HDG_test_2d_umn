#pragma once
#include <Eigen/Sparse> // Required for SparseMatrix
#include <Eigen/Dense>  // Required for VectorXd

// 1. Define the output struct:
struct Sem1Op {
    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> S;    // <-- was MatrixXd
    Eigen::SparseMatrix<double> Lm;
    Eigen::SparseMatrix<double> Lp;
};

// 2. Declare Function:
// Take the integer p (polynomial degree) and return Sem1Op
Sem1Op sem1(int p);