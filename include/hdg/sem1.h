#pragma once
#include <Eigen/Sparse> // Required for SparseMatrix
#include <Eigen/Dense>  // Required for VectorXd

// 1. Define the output struct:
struct Sem1Op{
    Eigen::SparseMatrix<double> M;  // Mass Matrix (Diagonal)
    Eigen::MatrixXd S;              // Stiffness Matrix 
    Eigen::SparseMatrix<double> Lm; // Left Boundary operator
    Eigen::SparseMatrix<double> Lp; // Right Boundary operator
};

// 2. Declare Function:
// Take the integer p (polynomial degree) and return Sem1Op
Sem1Op sem1(int p);