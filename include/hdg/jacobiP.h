#pragma once
#include <Eigen/Dense>  // Required for VectorXd

// Computes the values of the N-th order Jacobi Polynomial P_N^(alpha,beta) at points x
Eigen::VectorXd jacobiP(int N, double alpha, double beta, const Eigen::VectorXd &x);