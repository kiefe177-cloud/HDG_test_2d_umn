#pragma once
#include <Eigen/Dense>

// Must match src/vec.cpp EXACTLY:
// Use MatrixXd (which covers both matrices and vectors)
Eigen::VectorXd vec(const Eigen::MatrixXd& u);