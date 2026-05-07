#pragma once

#include <Eigen/Dense>
#include <string>

void save_grid_diagnostics(
    const std::string& path,
    const Eigen::MatrixXd& Phi_tfi,
    const Eigen::MatrixXd& Psi_tfi,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const Eigen::VectorXd& x_left,    const Eigen::VectorXd& y_left,
    const Eigen::VectorXd& x_right,   const Eigen::VectorXd& y_right,
    const Eigen::VectorXd& x_bottom,  const Eigen::VectorXd& y_bottom,
    const Eigen::VectorXd& x_top,     const Eigen::VectorXd& y_top,
    const double& x_p,                const double& y_p,
    const double& phi_p,              const double& psi_p,
    const Eigen::MatrixXd& skewness,
    const Eigen::MatrixXd& jacobian);