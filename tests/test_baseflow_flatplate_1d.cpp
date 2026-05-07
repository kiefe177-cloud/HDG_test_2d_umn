#include "hdg/baseflow_flatplate_1d.h"
#include "parameters/MalikSpallParams.h"
#include "hdg/baseflow.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

int test_baseflow_flatplate_1d() {
    std::cout << "\n========== Test B: Baseflow_flatplate_1d ==========\n";
    
    SimulationParams params = get_malik_spall_params();

    int Ny = 350; 
    int i1 = 1;

    Baseflow baseflow_copy = baseflow_flatplate_1d(params,Ny,i1);
    
    int Nx_expected = 1;
    int Ny_expected = 350;
    int Nz_expected = 1;
    int size_expected = 350;

    Eigen::VectorXd y_expected(10);
    Eigen::VectorXd x_expected(10);
    Eigen::VectorXd z_expected(10);

    x_expected << 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
    z_expected << 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;

    y_expected << 0.0000, 0.0652, 0.1308, 0.1967, 0.2629, 0.3295, 0.3964, 0.4636, 0.5312, 0.5991;

    Eigen::VectorXd rhou_expected(10);
    Eigen::VectorXd rhov_expected(10);
    Eigen::VectorXd rhow_expected(10);

    rhou_expected << 6.7089e-19,    8.3181e-4,  0.0017, 0.0025, 0.0034, 0.0042, 0.0051, 0.0059, 0.0068, 0.0076;
    rhov_expected << 0.0000,        0.0000,     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
    rhow_expected << 0.0000,        0.0000,     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
    
    Eigen::VectorXd E_expected(10);
    Eigen::VectorXd p_expected(10);
    Eigen::VectorXd T_expected(10);
    Eigen::VectorXd mu_expected(10);

    E_expected  << 0.0714, 0.0714, 0.0714, 0.0714, 0.0715, 0.0715, 0.0715, 0.0715, 0.0715, 0.0716;
    p_expected  << 0.0286, 0.0286, 0.0286, 0.0286, 0.0286, 0.0286, 0.0286, 0.0286, 0.0286, 0.0286;
    T_expected  << 5.1714, 5.1714, 5.1712, 5.1708, 5.1704, 5.1698, 5.1690, 5.1681, 5.1671, 5.1660;
    mu_expected << 5.1052, 5.1052, 5.1050, 5.1047, 5.1044, 5.1039, 5.1033, 5.1026, 5.1018, 5.1009;
    

    Eigen::VectorXd y = baseflow_copy.x[1];
    Eigen::VectorXd x = baseflow_copy.x[0];
    Eigen::VectorXd z = baseflow_copy.x[2];

    
    Eigen::VectorXd rhou = baseflow_copy.rhou[0];
    Eigen::VectorXd rhov = baseflow_copy.rhou[1];
    Eigen::VectorXd rhow = baseflow_copy.rhou[2];

    
    Eigen::VectorXd E = baseflow_copy.E;
    Eigen::VectorXd P = baseflow_copy.p();
    Eigen::VectorXd T = baseflow_copy.T();
    Eigen::VectorXd mu= baseflow_copy.mu();

    // --- YOUR ORIGINAL DEBUG PRINTS ---
    std::cout << "Size y: " << y.size() << "\n";
    std::cout << "Size x: " << x.size() << "\n";
    std::cout << "Size z: " << z.size() << "\n";

    std::cout << "Size rhou: " << rhou.size() << "\n";
    std::cout << "Size rhov: " << rhov.size() << "\n";
    std::cout << "Size rhow: " << rhow.size() << "\n";

    std::cout << "Size E: " << E.size() << "\n";
    std::cout << "Size p: " << P.size() << "\n";
    std::cout << "Size mu: " << mu.size() << "\n";
    std::cout << "Size T: " << T.size() << "\n\n";

    std::cout << "y (first 10):\n" << y.head(10) << "\n\n";
    std::cout << "x (first 10):\n" << x.head(10) << "\n\n";
    std::cout << "z (first 10):\n" << z.head(10) << "\n\n";


    std::cout << "rhou (first 10):\n" << rhou.head(10) << "\n\n";
    std::cout << "rhov (first 10):\n" << rhov.head(10) << "\n\n";
    std::cout << "rhow (first 10):\n" << rhow.head(10) << "\n\n";

    
    std::cout << "E (first 10):\n" << E.head(10) << "\n\n";
    std::cout << "P (first 10):\n" << P.head(10) << "\n\n";
    std::cout << "mu (first 10):\n" << mu.head(10) << "\n\n";
    std::cout << "T (first 10):\n" << T.head(10) << "\n\n";
    // ----------------------------------

    // CRITICAL FIX: Only subtract against the first 10 elements of the computed profiles!
    Eigen::VectorXd Error_y = (y.head(10) - y_expected).array().abs();
    Eigen::VectorXd Error_x = (x.head(10) - x_expected).array().abs();
    Eigen::VectorXd Error_z = (z.head(10) - z_expected).array().abs();
    
    Eigen::VectorXd Error_rhou = (rhou.head(10) - rhou_expected).array().abs();
    Eigen::VectorXd Error_rhov = (rhov.head(10) - rhov_expected).array().abs();
    Eigen::VectorXd Error_rhow = (rhow.head(10) - rhow_expected).array().abs();

    
    Eigen::VectorXd Error_E = (E.head(10) - E_expected).array().abs();
    Eigen::VectorXd Error_P = (P.head(10) - p_expected).array().abs();
    Eigen::VectorXd Error_mu = (mu.head(10) - mu_expected).array().abs();
    Eigen::VectorXd Error_T = (T.head(10) - T_expected).array().abs();

    // --- YOUR ORIGINAL ERROR PRINTS ---
    std::cout << "Error y:\n" << Error_y << "\n\n";
    std::cout << "Error rhou:\n" << Error_rhou << "\n\n";
    std::cout << "Error E:\n" << Error_E << "\n\n";
    std::cout << "Error T:\n" << Error_T << "\n\n";
    // ----------------------------------

    double max_error_y = Error_y.maxCoeff();
    double max_error_rhou = Error_rhou.maxCoeff();
    double max_error_E = Error_E.maxCoeff();
    double max_error_T = Error_T.maxCoeff();

    // Set tolerances
    double tol_y = 1e-4;
    double tol_rhou = 1e-4;
    double tol_E = 1e-4;
    double tol_T = 1e-4; 

    bool all_passed = true;

    // Streamlined pass/fail logic
    if (max_error_y > tol_y) {
        std::cout << "--> TEST FAILED! Max error y: " << max_error_y << "\n";
        all_passed = false;
    }
    if (max_error_rhou > tol_rhou) {
        std::cout << "--> TEST FAILED! Max error u: " << max_error_rhou << "\n";
        all_passed = false;
    }
    if (max_error_E > tol_E) {
        std::cout << "--> TEST FAILED! Max error v: " << max_error_E << "\n";
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
    rc |= test_baseflow_flatplate_1d();
    
    if (rc == 0) {
        std::cout << "\nALL TESTS PASSED.\n";
    } else {
        std::cout << "\nSOME TESTS FAILED.\n";
    }
    
    return rc;
}