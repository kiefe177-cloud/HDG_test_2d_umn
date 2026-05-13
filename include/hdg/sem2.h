#pragma once
#include <vector>
#include <Eigen/Sparse> // Required for SparseMatrix
#include <Eigen/Dense>  // Required for VectorXd
#include "sem1.h"

// 1. Define the output struct:
struct Sem2Op {
    Eigen::SparseMatrix<double> M;
    std::vector<Eigen::SparseMatrix<double>> S;   // <-- sparse, not MatrixXd
    std::vector<Eigen::SparseMatrix<double>> L;
    std::vector<Eigen::SparseMatrix<double>> T;
};

// 2. Declare Function:
// Take the integer p (polynomial degree) and return Sem2Op
Sem2Op sem2(int p);