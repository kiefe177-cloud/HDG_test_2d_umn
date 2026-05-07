#include "hdg/sd.h"

Eigen::SparseMatrix<double> sd(const Eigen::VectorXd& x) {
    // In MATLAB: A = spdiags(x, 0, N, N);
    // In Eigen, we can use the asDiagonal() method to create a sparse diagonal matrix.
    
    long N = x.size();

    Eigen::SparseMatrix<double> A(N, N);
    A = x.asDiagonal();

    return A;
}
