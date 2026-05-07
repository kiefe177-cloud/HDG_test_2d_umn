// multidof_ops.h
#pragma once

#include "hdg/sem1.h"
#include "hdg/sem2.h"
#include <Eigen/Sparse>
#include <array>

struct MultidofOps {
    Eigen::SparseMatrix<double> M0;
    Eigen::SparseMatrix<double> M;
    std::array<Eigen::SparseMatrix<double>, 2>  S;
    std::array<Eigen::SparseMatrix<double>, 12> L;
    std::array<Eigen::SparseMatrix<double>, 12> T;
};

MultidofOps multidof_ops(int nvar, const Sem1Op& sem1, const Sem2Op& sem2);
