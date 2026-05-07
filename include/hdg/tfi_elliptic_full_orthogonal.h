#pragma once

#include <Eigen/Dense>

#include "hdg/elliptic_result.h"
#include "hdg/ortho_opts.h"

EllipticResult tfi_elliptic_full_orthogonal(
    const Eigen::MatrixXd& X_tfi,
    const Eigen::MatrixXd& Y_tfi,
    const Eigen::VectorXd& phi_comp_vec,
    const Eigen::VectorXd& psi_comp_vec,
    int n_grid_pts,
    int n_elliptic_iter,
    double sor_omega,
    double convergence_tol,
    const OrthoOpts& ortho_opts);