#pragma once

#include "hdg/Msh.h"
#include "parameters/params_struct.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::SparseMatrix<double> get_tau_visc(const Msh& msh,
                                        const Eigen::MatrixXd& brho_face,
                                        const Eigen::MatrixXd& brhou_face,
                                        const Eigen::MatrixXd& brhov_face,
                                        const Eigen::MatrixXd& brhow_face,
                                        const Eigen::MatrixXd& bE_face,
                                        const SimulationParams& params,
                                        const double c_v,
                                        const int i2,
                                        const int i1);