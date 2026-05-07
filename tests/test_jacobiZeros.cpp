#include <iostream>
#include <cmath>
#include <cassert>

#include "hdg/jacobiZerosGW.h"

int main() {
    // 1. The Function Call
    Eigen::VectorXd roots = jacobiZerosGW(2,0,0);

    // 2. The Expected Results
    double expected_root1 = -1.0 / std::sqrt(3.0);
    double expected_root2 =  1.0 / std::sqrt(3.0);
    double tol = 1e-5;

    // 3. The Assertions
    // "Assert that the distance between result and expected is tiny"
    assert(std::abs(roots(0) - (expected_root1)) < tol);
    assert(std::abs(roots(1) - (expected_root2)) < tol);

    // If the program reaches this line, it means all assertions passed.
    std::cout << "[PASS] jacobiZerosGW Test for n=2, alpha=0, beta=0" << std::endl;

    Eigen::VectorXd roots_7 = jacobiZerosGW(7,1,1);

    double expected_roots[7] = {
        -0.8998,
        -0.6772,
        -0.3631,
         0.0,
         0.3631,
         0.6772,
         0.8998,
    };

    tol = 1e-4;

    for (int i = 0; i < 7; ++i) {
        assert(std::abs(roots_7(i) - expected_roots[i]) < tol);
    }

    std::cout << "[PASS] jacobiZerosGW Test for n=7, alpha=1, beta=1" << std::endl;

    Eigen::VectorXd roots_asym = jacobiZerosGW(2, 1, 0);
    
    double expected_asym2 = 0.2899;
    double expected_asym1 =  -0.6899;
    
    std::cout << "\n[TEST 3] Asymmetric Check (n=2, alpha=1, beta=0)" << std::endl;
    std::cout << "Computed: " << roots_asym.transpose() << std::endl;
    std::cout << "Expected: " << expected_asym1 << "  " << expected_asym2 << std::endl;

    assert(std::abs(roots_asym(0) - expected_asym1) < 1e-4);
    assert(std::abs(roots_asym(1) - expected_asym2) < 1e-4);
    
    std::cout << "[PASS] Asymmetric Test" << std::endl;

    return 0;
}
