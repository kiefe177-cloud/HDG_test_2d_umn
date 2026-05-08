#pragma once

#include "parameters/params_struct.h"
#include "hdg/msh_amr_curv.h"
#include "hdg/sem1.h"
#include "hdg/sem2.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <vector>

struct dg
{
    using SparseCD  = Eigen::SparseMatrix<std::complex<double>>;
    using SparseD   = Eigen::SparseMatrix<double>;
    
    SparseCD A;
    SparseD B;
    SparseD C;
    SparseD D;

    std::vector<SparseCD>   Ik;
    std::vector<SparseD>    QB;
    std::vector<SparseD>    QC;
    std::vector<SparseD>    QiD;
    
};

dg dg_mat_2Roe(const SimulationParams& params, const double omega,
    const Msh& msh, const int nvar, const Sem1Op& op1, const Sem2Op& op2,
    const Eigen::MatrixXd& bU, const Eigen::MatrixXd& bUx, const Eigen::MatrixXd& bUy, const Eigen::MatrixXd& bUz,
    const Eigen::MatrixXd& bU_f, const Eigen::MatrixXd& bU_fx, const Eigen::MatrixXd& bU_fy, const Eigen::MatrixXd& bU_fz,
    int m);
