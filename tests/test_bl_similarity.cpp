#include "hdg/bl_similarity.h"
#include "parameters/MalikSpallParams.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

int test_bl_similarity() {
    std::cout << "\n========== Test B: Boundary Layer Similarity ==========\n";
    
    SimulationParams params = get_malik_spall_params();

    int N = 350; 
    BlProfiles bl_profiles = bl_similarity(params, 10*N);

    Eigen::VectorXd y_expected(10);
    Eigen::VectorXd u_expected(10);
    Eigen::VectorXd v_expected(10);
    Eigen::VectorXd T_expected(10); 
    
    y_expected << 0, 4.4464e-4, 8.8929e-4, 0.0013, 0.0018, 0.0022, 0.0027, 0.0031, 0.0036, 0.0040; 
    u_expected << 0, 0.0011,    0.0022,    0.0033, 0.0044, 0.0055, 0.0066, 0.0077, 0.0088, 0.0099; 
    v_expected << 0, 1.2257e-7, 4.9029e-7, 1.1031e-6, 1.9612e-6, 3.0643e-6, 4.4126e-6, 6.0061e-6, 7.8447e-6, 9.9285e-6; 
    T_expected << 5.1714, 5.1714, 5.1714,  5.1714, 5.1713, 5.1713, 5.1713, 5.1712, 5.1711, 5.1711; 

    // --- YOUR ORIGINAL DEBUG PRINTS ---
    std::cout << "Size y: " << bl_profiles.y.size() << "\n";
    std::cout << "Size u: " << bl_profiles.u.size() << "\n";
    std::cout << "Size v: " << bl_profiles.v.size() << "\n";
    std::cout << "Size T: " << bl_profiles.T.size() << "\n\n";

    std::cout << "y (first 10):\n" << bl_profiles.y.head(10) << "\n\n";
    std::cout << "u (first 10):\n" << bl_profiles.u.head(10) << "\n\n";
    std::cout << "v (first 10):\n" << bl_profiles.v.head(10) << "\n\n";
    std::cout << "T (first 10):\n" << bl_profiles.T.head(10) << "\n\n";
    // ----------------------------------

    // CRITICAL FIX: Only subtract against the first 10 elements of the computed profiles!
    Eigen::VectorXd Error_y = (bl_profiles.y.head(10) - y_expected).array().abs();
    Eigen::VectorXd Error_u = (bl_profiles.u.head(10) - u_expected).array().abs();
    Eigen::VectorXd Error_v = (bl_profiles.v.head(10) - v_expected).array().abs();
    Eigen::VectorXd Error_T = (bl_profiles.T.head(10) - T_expected).array().abs();

    // --- YOUR ORIGINAL ERROR PRINTS ---
    std::cout << "Error y:\n" << Error_y << "\n\n";
    std::cout << "Error u:\n" << Error_u << "\n\n";
    std::cout << "Error v:\n" << Error_v << "\n\n";
    std::cout << "Error T:\n" << Error_T << "\n\n";
    // ----------------------------------

    double max_error_y = Error_y.maxCoeff();
    double max_error_u = Error_u.maxCoeff();
    double max_error_v = Error_v.maxCoeff();
    double max_error_T = Error_T.maxCoeff();

    // Set tolerances
    double tol_y = 1e-4;
    double tol_u = 1e-4;
    double tol_v = 1e-7;
    double tol_T = 1e-4; 

    bool all_passed = true;

    // Streamlined pass/fail logic
    if (max_error_y > tol_y) {
        std::cout << "--> TEST FAILED! Max error y: " << max_error_y << "\n";
        all_passed = false;
    }
    if (max_error_u > tol_u) {
        std::cout << "--> TEST FAILED! Max error u: " << max_error_u << "\n";
        all_passed = false;
    }
    if (max_error_v > tol_v) {
        std::cout << "--> TEST FAILED! Max error v: " << max_error_v << "\n";
        all_passed = false;
    }
    if (max_error_T > tol_T) {
        std::cout << "--> TEST FAILED! Max error T: " << max_error_T << "\n";
        all_passed = false;
    }

    if (all_passed) {
        std::cout << "--> ALL PROFILES PASSED!\n";
        return 0; // Success
    }

    return 1; // Failure
}

int main() {
    int rc = 0;
    
    // Bitwise OR will accumulate any failures. 
    rc |= test_bl_similarity();
    
    if (rc == 0) {
        std::cout << "\nALL TESTS PASSED.\n";
    } else {
        std::cout << "\nSOME TESTS FAILED.\n";
    }
    
    return rc;
}