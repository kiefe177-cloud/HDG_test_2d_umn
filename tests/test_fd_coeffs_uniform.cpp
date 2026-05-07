#include "hdg/fd_coeffs_uniform.h"
#include <iostream>
#include <Eigen/Dense>

int test_fd_coeffs_uniform(){
    std::cout << "\n========== Test A: Coefficients ==========\n";
    const int d = 1;
    
    // FIX 1: Explicitly set the size to 7 before using the comma initializer
    Eigen::VectorXd s(7);
    Eigen::VectorXd c_expected(7);

    s          << -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0; 
    c_expected << -0.0167, 0.1500, -0.7500, 0.0000, 0.7500, -0.1500, 0.0167; 

    Eigen::VectorXd c = fd_coeffs_uniform(s, d);

    // FIX 2: Use double quotes, newlines, and print the correct variables
    std::cout << "s:\n" << s << "\n\n";
    std::cout << "d:\n" << d << "\n\n";
    std::cout << "c:\n" << c << "\n\n";

    // FIX 3: Error is a VectorXd
    Eigen::VectorXd Error = (c - c_expected).array().abs();

    std::cout << "Error:\n" << Error << "\n\n";

    // FIX 4: Automate the pass/fail check and return an integer
    // Since c_expected is truncated to 4 decimal places, we need a tolerance
    double max_error = Error.maxCoeff();
    double tolerance = 1e-4; 

    if (max_error <= tolerance) {
        std::cout << "--> TEST PASSED! Max error: " << max_error << "\n";
        return 0; // 0 indicates success
    } else {
        std::cout << "--> TEST FAILED! Max error: " << max_error << "\n";
        return 1; // 1 indicates failure
    }
}

int main(){
    int rc = 0;
    
    // Bitwise OR will accumulate any failures. 
    // If any test returns 1, rc becomes 1.
    rc |= test_fd_coeffs_uniform();
    
    if (rc == 0) {
        std::cout << "\nALL TESTS PASSED.\n";
    } else {
        std::cout << "\nSOME TESTS FAILED.\n";
    }
    
    return rc;
}