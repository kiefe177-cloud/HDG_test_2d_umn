// include/hdg/elliptic_result.h
#pragma once
#include <Eigen/Dense>

struct Diagnostics {
    Eigen::MatrixXd skewness;
    Eigen::MatrixXd jacobian;
};

struct EllipticResult {
    Eigen::MatrixXd X_ellip_grid;
    Eigen::MatrixXd Y_ellip_grid;
    Diagnostics     diagnost;
};