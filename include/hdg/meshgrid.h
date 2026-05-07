#pragma once
#include <Eigen/Dense>
#include <utility>

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y);