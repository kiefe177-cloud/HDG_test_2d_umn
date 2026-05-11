#pragma once
#include "parameters/params_struct.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>

struct source_terms
{
    using SparseCD  = Eigen::SparseMatrix<std::complex<double>>;
    using SparseD   = Eigen::SparseMatrix<double>;

    SparseCD Q;
    SparseD Qx;
    SparseD Qy;
    SparseD Qz;
    
};

source_terms source_terms_Q(const Eigen::VectorXd& bU,
                            const Eigen::VectorXd& bUx,
                            const Eigen::VectorXd& bUy,
                            const Eigen::VectorXd& bUz,
                            const SimulationParams& params,
                            const int nvar,
                            const int m,
                            const Eigen::MatrixXd& invr_in);
