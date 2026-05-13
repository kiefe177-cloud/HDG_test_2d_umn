#include "hdg/multidof_ops.h"
#include <unsupported/Eigen/KroneckerProduct>

MultidofOps multidof_ops(const int nvar, const Sem1Op& sem1, const Sem2Op& sem2){
    MultidofOps out;

    Eigen::SparseMatrix<double> I(nvar,nvar);
    I.setIdentity();

    out.M0  = Eigen::kroneckerProduct(I, sem1.M);
    out.M = Eigen::kroneckerProduct(I, sem2.M);
    out.M.prune(0.0, 1e-14);

    for (int i1 = 0; i1 < 2; ++i1){
        out.S[i1] = Eigen::kroneckerProduct(I, sem2.S[i1]);
        out.S[i1].prune(0.0, 1e-14);
    }

    for (int i1 = 0; i1 < 12; ++i1){
        out.L[i1] = Eigen::kroneckerProduct(I, sem2.L[i1]);
        out.L[i1].prune(0.0, 1e-14);
        out.T[i1] = Eigen::kroneckerProduct(I, sem2.T[i1]);
        out.T[i1].prune(0.0, 1e-14);
    }
    
    return out;
}