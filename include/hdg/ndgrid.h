#pragma once 
#include <Eigen/Dense>  // Required for MatrixXd
#include <iostream>
#include <utility>    // Required for std::pair

// Replicates grid vectros to produce a rectangular grid (like MATLAB's ndgrid)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ndgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
