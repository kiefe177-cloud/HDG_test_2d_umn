#pragma once
#include <Eigen/Sparse> // Required for SparseMatrix
#include <Eigen/Dense>  // Required for VectorXd

// Computes the n-th order Legendre polynomial P_n at points x using recurrence relation
Eigen::VectorXd legendreP(int n, const Eigen::VectorXd& x);
