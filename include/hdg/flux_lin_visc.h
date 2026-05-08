#pragma once
#include "parameters/params_struct.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>

struct viscous_fluxes {
    Eigen::SparseMatrix<std::complex<double>> F;
    Eigen::SparseMatrix<double>               Fx;
    Eigen::SparseMatrix<double>               Fy;
    Eigen::SparseMatrix<double>               Fz;

    Eigen::SparseMatrix<std::complex<double>> G;
    Eigen::SparseMatrix<double>               Gx;
    Eigen::SparseMatrix<double>               Gy;
    Eigen::SparseMatrix<double>               Gz;

    Eigen::SparseMatrix<std::complex<double>> H;
    Eigen::SparseMatrix<double>               Hx;
    Eigen::SparseMatrix<double>               Hy;
    Eigen::SparseMatrix<double>               Hz;
};

viscous_fluxes flux_lin_visc(const Eigen::VectorXd& bU,const Eigen::VectorXd& bUx,const Eigen::VectorXd& bUy,const Eigen::VectorXd& bUz,
                            const SimulationParams& params, const int nvar, const int m, const Eigen::SparseMatrix<double>& invr);
