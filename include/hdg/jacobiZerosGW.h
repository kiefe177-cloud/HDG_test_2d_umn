#pragma once
#include <Eigen/Dense>  // Required for VectorXd

// Computes zeros of the Jacobi polynomial P_n^(alpha,beta) using the Golub-Welsch algorithm
Eigen::VectorXd jacobiZerosGW(int n, double alpha, double beta);