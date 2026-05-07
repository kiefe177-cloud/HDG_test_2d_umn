#include <iostream>
#include <cassert>  // Required for assert()
#include "hdg/sem1.h"

int main() {
    int p = 3;
    Sem1Op op = sem1(p);
    double tol = 1e-4;

    // 1. Verify Mass Matrix (Sparse)
    Eigen::SparseMatrix<double> expected_M(4,4);
    expected_M.insert(0,0) = 1.0/6.0;
    expected_M.insert(1,1) = 5.0/6.0;
    expected_M.insert(2,2) = 5.0/6.0;
    expected_M.insert(3,3) = 1.0/6.0;

    // IF this is false, the program crashes immediately with an error message
    assert(op.M.isApprox(expected_M, tol) && "Mass Matrix M mismatch");

    // 2. Verify Stiffness Matrix (Dense)
    Eigen::MatrixXd expected_S(4,4);
    expected_S << -0.5, -0.6742, 0.2575, -0.0833,
                   0.6742, 0.0, -0.9317, 0.2575,
                  -0.2575, 0.9317, 0.0, -0.6742,
                   0.0833, -0.2575, 0.6742, 0.5;

    assert(op.S.isApprox(expected_S, tol) && "Stiffness Matrix S mismatch");

    // 3. Verify Boundary Operators
    Eigen::SparseMatrix<double> expected_Lm(4,4);
    expected_Lm.insert(0,0) = 1.0;
    assert(op.Lm.isApprox(expected_Lm, tol) && "Lm mismatch");

    Eigen::SparseMatrix<double> expected_Lp(4,4);
    expected_Lp.insert(3,3) = 1.0;
    assert(op.Lp.isApprox(expected_Lp, tol) && "Lp mismatch");

    std::cout << "[PASS] All sem1 tests passed!" << std::endl;
    return 0;
}
