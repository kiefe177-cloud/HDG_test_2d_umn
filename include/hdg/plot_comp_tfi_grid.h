#pragma once
#include <Eigen/Dense>

// Function Declaration
void plot_comp_tfi_grid(
    const Eigen::MatrixXd& Phi_tfi, 
    const Eigen::MatrixXd& Psi_tfi, 
    const Eigen::MatrixXd& X, 
    const Eigen::MatrixXd& Y, 
    double x_p, double y_p, 
    double phi_val, double psi_val
);