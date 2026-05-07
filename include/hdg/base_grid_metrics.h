#pragma once

#include <Eigen/Dense>

struct BaseGridMetrics {
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
    Eigen::VectorXd phi_vec;
    Eigen::VectorXd psi_vec;
    int    nx;
    int    ny;
    double dphi;
    double dpsi;
};

BaseGridMetrics base_grid_metrics(
    const Eigen::MatrixXd& X_ellip_grid,
    const Eigen::MatrixXd& Y_ellip_grid,
    const Eigen::VectorXd& phi_comp_vec,
    const Eigen::VectorXd& psi_comp_vec);