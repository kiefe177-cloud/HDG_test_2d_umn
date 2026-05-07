#pragma once
#include <Eigen/Sparse> // Required for SparseMatrix
#include <Eigen/Dense>  // Required for VectorXd

// Declaration:
// Takes a dense vector 'x' and returns a Sparse Matrix with 'x' on the diagonal.
Eigen::SparseMatrix<double> sd(const Eigen::VectorXd& x);