#pragma once
#include "parameters/params_struct.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

struct flux_split_jacobians
{
    using SparseD = Eigen::SparseMatrix<double>;

    SparseD Ap;
    SparseD Am;
};

flux_split_jacobians flux_split(const Eigen::VectorXd& U,
                                const Eigen::SparseMatrix<double>& nx,
                                const Eigen::SparseMatrix<double>& ny,
                                const Eigen::SparseMatrix<double>& nn,
                                const SimulationParams& params,
                                const int nvar);