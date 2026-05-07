#include <iostream>
#include <cmath>
#include <cassert>

#include "hdg/legendreP.h"
#include "hdg/jacobiZerosGW.h"

int main(){
    // 1. The Function Call
    int N = 4;
    int alpha = 1;
    int beta = 1;
    Eigen::VectorXd roots = jacobiZerosGW(N-1, alpha, beta);

    Eigen::VectorXd x(roots.size() + 2);
    x << -1.0, roots, 1.0;
    Eigen::VectorXd P = legendreP(N, x);

    // 2. The Expected Results
    Eigen::VectorXd expected_P(x.size());

    expected_P(0) = 1.0;
    expected_P(1) = -.4286;
    expected_P(2) = 0.3750;
    expected_P(3) = -.4286;
    expected_P(4) = 1.0;

    double tol = 1e-4;

    // 3. The Assertions
    for (int i = 0; i < x.size(); ++i){
        assert(std::abs(P(i) - expected_P(i)) < tol);
    }
    std::cout << "[PASS] legendreP Test for N=4, alpha=1, beta=1" << std::endl;

    // Second Test: Weights Check
    // 1. The Function Call
    N = 8;
    alpha = 1;
    beta = 1;
    tol = 1e-4;
    roots = jacobiZerosGW(N-1, alpha, beta);

    x.resize(roots.size() + 2);
    x << -1.0, roots, 1.0;
    P = legendreP(N, x);

    Eigen::VectorXd w_i(x.size());
    for (int i = 0; i < x.size(); ++i){
        w_i(i) = 2.0 / (N*(N+1)*std::pow(P(i),2));
    }

    // 2. The Expected Results
    // Weights Expected from Table
    Eigen::VectorXd w_expected(x.size());
    w_expected << 0.0278,.1655,0.2745,0.3464,0.3715,0.3464,0.2745,0.1655,0.0278;

    // 3. The Assertions
    for (int i = 0; i < x.size(); ++i){
        w_i(i) = 2.0 / (N*(N+1)*std::pow(P(i),2));
        assert(std::abs(w_i(i) - w_expected(i)) < tol);
    }
    std::cout << "[PASS] legendreP Weights Test for N=8, alpha=1, beta=1" << std::endl;

    return 0;
}
