#include <iostream>
#include <cmath>
#include <cassert>
#include "hdg/jacobiP.h"
#include "hdg/GLQuad.h"

int main(){

    // Test 1: Jacobi Polynomial P_1^(0,0)(x) at GL points for N=4
    // Function Call
    int j = 1;
    Eigen::VectorXd x,w;
    GLQuad(4,x,w);

    Eigen::VectorXd P = jacobiP(j,0.0,0.0,x);

    // Expected Results
    Eigen::VectorXd expected_P(x.size());
    expected_P << -1.0, -0.6547, 0.0, 0.6547, 1.0;

    // Verification
    double tol = 1e-4;
    for (int i=0; i<x.size(); i++){
        assert( std::abs(P(i) - expected_P(i)) < tol );
    }
    std::cout << "[PASS] jacobiP Test for N=1" << std::endl;


    // Test 2: Jacobi Polynomial P_4^(0,0)(x) at GL points for N=4, j=4
    // Function Call
    j = 4;
    P = jacobiP(j,0.0,0.0,x);
    
    // Expected Results
    expected_P << 1.0, -0.4286, 0.3750, -0.4286, 1.0; 

    for (int i=0; i<x.size(); i++){
        assert( std::abs(P(i) - expected_P(i)) < tol );
    }

    std::cout << "[PASS] jacobiP Test for N=4" << std::endl;

    // Test 3: Jacobi Polynomial P_4^(2,2)(x) at GL points for N=4, j=4
    // Function Call
    j = 4;
    P = jacobiP(j,2.0,2.0,x);
    
    expected_P << 15.0, -0.6122, 0.9375, -0.6122, 15.0;
    for (int i=0; i<x.size(); i++){
        assert( std::abs(P(i) - expected_P(i)) < tol );
    }
    std::cout << "[PASS] jacobiP Test for N=4, alpha=2, beta=2" << std::endl;

    // Test 4: Jacobi Polynomial P_0^(0,0)(x) at GL points for N=8, j=0
    // Function Call
    j = 0;
    GLQuad(8,x,w);
    P = jacobiP(j,0.0,0.0,x);

    // Expected Results
    expected_P.resize(x.size());
    expected_P << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    for (int i=0; i<x.size(); i++){
        assert( std::abs(P(i) - expected_P(i)) < tol );
    }
    std::cout << "[PASS] jacobiP Test for N=8" << std::endl;

    return 0;
}
