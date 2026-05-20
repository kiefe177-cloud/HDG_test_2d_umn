#pragma once

#include "hdg/Msh.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Complex>
#include <string>

void save_outputs_h5(
    const std::string& path,
    const Eigen::MatrixXd& Phi_tfi,
    const Eigen::MatrixXd& Psi_tfi,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const Msh& msh,
    const Eigen::MatrixXd& brho,
    const Eigen::MatrixXd& brhou,
    const Eigen::MatrixXd& brhov,
    const Eigen::MatrixXd& brhow,
    const Eigen::MatrixXd& bE,
    const int Ne,
    const int Nf,
    const int Me,
    const int Mf,
    const int p,
    const Eigen::VectorXcd& u_e,
    const Eigen::VectorXcd& u_f);
