#pragma once
#include <Eigen/Dense>  // Required for VectorXd

// Computes the N-th order Gauss-Lobatto-Legendre quadrature points and weights
void GLQuad(int N, Eigen::VectorXd &x, Eigen::VectorXd &w);

