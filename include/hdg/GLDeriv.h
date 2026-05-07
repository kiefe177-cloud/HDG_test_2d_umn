#pragma once
#include <Eigen/Dense>  // Required for VectorXd
#include <cmath>

// Computes the N-th order Gauss-Lobatto Derivative matrix
Eigen::MatrixXd GLDeriv(int N);