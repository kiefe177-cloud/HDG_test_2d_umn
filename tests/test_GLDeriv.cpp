#include <iostream>
#include <cmath>
#include <cassert>

#include "hdg/jacobiP.h"
#include "hdg/GLQuad.h"
#include "hdg/GLDeriv.h"

int main(){
    // Test GLDeriv for N=4

    // 1. Compute D
    int N = 4;
    Eigen::MatrixXd D = GLDeriv(N);

    // 2. Expected Results
    Eigen::MatrixXd expected_D(N+1, N+1);
    expected_D <<  -5.0, 6.7565, -2.6667, 1.4102, -0.5,
                   -1.2410, 0.0, 1.7457, -0.7638, 0.2590,
                    0.3750, -1.3366, 0.0, 1.3366, -0.3750,
                    -0.2590, 0.7638, -1.7457, 0.0, 1.2410,
                    0.5, -1.4102, 2.6667, -6.7565, 5.0;

    double tol = 1e-4;
    // 3. Verify Results
    for (int i=0; i<=N; i++){
        for (int j=0; j<=N; j++){
            assert( std::abs(D(i,j) - expected_D(i,j)) < tol );
        }
    }   

    std::cout << "[PASS] GLDeriv Test for N=4" << std::endl;

    // Test GLDeriv for N=2
    N = 2;
    D = GLDeriv(N);

    // 2. Expected Results
    expected_D.resize(N+1, N+1);
    expected_D << -1.5, 2.0, -0.5,
                   -0.5, 0.0, 0.5,
                    0.5, -2.0, 1.5;

    // 3. Verify Results
    for (int i=0; i<=N; i++){
        for (int j=0; j<=N; j++){
            assert( std::abs(D(i,j) - expected_D(i,j)) < tol );
        }   
    }
    std::cout << "[PASS] GLDeriv Test for N=2" << std::endl;
    return 0;
}
