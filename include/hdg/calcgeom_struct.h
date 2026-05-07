#pragma once

#include <Eigen/Dense>

struct calc_geom_output
{
    Eigen::MatrixXd x1;
    Eigen::MatrixXd y1;
    Eigen::MatrixXd J;
    Eigen::MatrixXd Ja11;
    Eigen::MatrixXd Ja12;
    Eigen::MatrixXd Ja21;
    Eigen::MatrixXd Ja22;
    Eigen::MatrixXd phi;
    Eigen::MatrixXd psi;
    Eigen::VectorXd Jf_1;
    Eigen::VectorXd Jf_2;
    Eigen::VectorXd Jf_3;
    Eigen::VectorXd Jf_4;  
};
