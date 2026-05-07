#include "hdg/fd_coeffs_uniform.h"
#include <iostream>
#include <Eigen/Dense> // Ensure this is included for colPivHouseholderQr

unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

Eigen::VectorXd fd_coeffs_uniform(const Eigen::VectorXd& s, const int& d) {
    
    int N = s.size();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N,N); 

    // Assemble the Vandermonde matrix
    for (int i1 = 0; i1 < N; ++i1){
        A.row(i1) = s.array().pow(i1).transpose();
    }

    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
    
    b[d] = factorial(d);

    // FIX 4: Removed the trailing .transpose() so it matches the VectorXd return type
    Eigen::VectorXd c = A.partialPivLu().solve(b);

    return c;
}