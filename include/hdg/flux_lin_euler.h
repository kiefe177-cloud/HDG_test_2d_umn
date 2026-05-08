#pragma once
#include "parameters/params_struct.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

struct inviscid_fluxes{
    using SparseD   = Eigen::SparseMatrix<double>;
    
    SparseD F;
    SparseD G;
    SparseD H;
};

inviscid_fluxes flux_lin_euler(const Eigen::VectorXd& bU, const SimulationParams& params, const int nvar);

