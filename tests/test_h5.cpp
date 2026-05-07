#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <vector>
#include "parameters/M5p8_Params.h"

#include <Eigen/Dense>
#include <highfive/H5File.hpp>

int main() {

    std::string filename = std::string(PROJECT_ROOT) + "/baseflows/baseflow_pst_b3p6.h5";
    std::cout << "Reading HDF5 file: " << filename << std::endl;

    HighFive::File file(filename, HighFive::File::ReadOnly);

    // Read into nested vector first
    std::vector<std::vector<double>> x_raw, y_raw;
    file.getDataSet("/mesh/xno").read(x_raw);
    file.getDataSet("/mesh/yno").read(y_raw);

    int nrows = static_cast<int>(x_raw.size());
    int ncols = static_cast<int>(x_raw[0].size());
    std::cout << "Raw shape: " << nrows << " x " << ncols << std::endl;

    // Copy into Eigen matrices
    Eigen::MatrixXd x_h5(nrows, ncols);
    Eigen::MatrixXd y_h5(nrows, ncols);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            x_h5(i, j) = x_raw[i][j];
            y_h5(i, j) = y_raw[i][j];
        }
    }
    SimulationParams params = get_M5p8_params();
    // ---- Apply MATLAB operations: transpose, scale, flip, normalize ----
    double scale_factor = 1.0;
    double L = params.L;  // Use the L parameter from the simulation params

    Eigen::MatrixXd x_h5t = x_h5.transpose();
    Eigen::MatrixXd y_h5t = y_h5.transpose();

    std::cout << "Transposed shape: " << x_h5t.rows() << " x " << x_h5t.cols() << std::endl;

    Eigen::MatrixXd X = scale_factor * x_h5t / L;
    Eigen::MatrixXd Y = scale_factor * y_h5t / L;

    std::cout << "Final X shape: " << X.rows() << " x " << X.cols() << std::endl;
    std::cout << "Final Y shape: " << Y.rows() << " x " << Y.cols() << std::endl;

    // ---- Print corners for comparison with MATLAB ----
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "\n--- Corner values (compare with MATLAB) ---" << std::endl;
    std::cout << "X(0,0)     = " << X(0,0) << std::endl;
    std::cout << "X(0,end)   = " << X(0, X.cols()-1) << std::endl;
    std::cout << "X(end,0)   = " << X(X.rows()-1, 0) << std::endl;
    std::cout << "X(end,end) = " << X(X.rows()-1, X.cols()-1) << std::endl;
    std::cout << std::endl;
    std::cout << "Y(0,0)     = " << Y(0,0) << std::endl;
    std::cout << "Y(0,end)   = " << Y(0, Y.cols()-1) << std::endl;
    std::cout << "Y(end,0)   = " << Y(Y.rows()-1, 0) << std::endl;
    std::cout << "Y(end,end) = " << Y(Y.rows()-1, Y.cols()-1) << std::endl;

    // MATLAB: X(1:280, 50:1201)  →  rows 1-280, cols 50-1201
    // C++:    rows 0-279, cols 49-1200

    Eigen::MatrixXd X_cut = X.block(0, 49, 280, 1152).eval();
    Eigen::MatrixXd Y_cut = Y.block(0, 49, 280, 1152).eval();

    std::cout << "\n" << std::endl;

    std::cout << "X shape: " << X_cut.rows() << " x " << X_cut.cols() << std::endl;
    std::cout << "Y shape: " << Y_cut.rows() << " x " << Y_cut.cols() << std::endl;

    // ---- Print corners for comparison with MATLAB ----
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "\n--- Corner values (compare with MATLAB) ---" << std::endl;
    std::cout << "X(0,49)     = " << X(0,49) << std::endl;
    std::cout << "X(0,1200)   = " << X(0, 1200) << std::endl;
    std::cout << "X(279,49)   = " << X(279, 49) << std::endl;
    std::cout << "X(279,1200) = " << X(279, 1200) << std::endl;
    std::cout << std::endl;
    std::cout << "Y(0,49)     = " << Y(0,49) << std::endl;
    std::cout << "Y(0,1200)   = " << Y(0, 1200) << std::endl;
    std::cout << "Y(279,49)   = " << Y(279, 49) << std::endl;
    std::cout << "Y(279,1200) = " << Y(279, 1200) << std::endl;

    // ---- Print corners for comparison with MATLAB ----
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "\n--- Corner values (compare with MATLAB) ---" << std::endl;
    std::cout << "X_cut(0,0)     = " << X_cut(0,0) << std::endl;
    std::cout << "X_cut(0,end)   = " << X_cut(0, X_cut.cols()-1) << std::endl;
    std::cout << "X_cut(end,0)   = " << X_cut(X_cut.rows()-1, 0) << std::endl;
    std::cout << "X_cut(end,end) = " << X_cut(X_cut.rows()-1, X_cut.cols()-1) << std::endl;
    std::cout << std::endl;
    std::cout << "Y_cut(0,0)     = " << Y_cut(0,0) << std::endl;
    std::cout << "Y_cut(0,end)   = " << Y_cut(0, Y_cut.cols()-1) << std::endl;
    std::cout << "Y_cut(end,0)   = " << Y_cut(Y_cut.rows()-1, 0) << std::endl;
    std::cout << "Y_cut(end,end) = " << Y_cut(Y_cut.rows()-1, Y_cut.cols()-1) << std::endl;



    // ---- Sanity checks ----
    assert(!X.hasNaN() && "X contains NaN!");
    assert(!Y.hasNaN() && "Y contains NaN!");
    assert(X.allFinite() && "X contains Inf!");
    assert(Y.allFinite() && "Y contains Inf!");

    double tol = 1e-4;
    assert(X_cut(0,0) - 12.8713 < tol && "X_cut(0,0) does not match expected value!");
    assert(Y_cut(0,0) - 7.05710 < tol && "Y_cut(0,0) does not match expected value!");
    
    assert(X_cut(0,X_cut.cols()-1) - 924.2516 < tol && "X_cut(0,end) does not match expected value!");
    assert(Y_cut(0,Y_cut.cols()-1) - 118.96056 < tol && "Y_cut(0,end) does not match expected value!");

    assert(X_cut(X_cut.rows()-1,0) - 11.0656 < tol && "X_cut(end,0) does not match expected value!");
    assert(Y_cut(Y_cut.rows()-1,0) - 14.6058 < tol && "Y_cut(end,0) does not match expected value!");
    
    assert(X_cut(X_cut.rows()-1,X_cut.cols()-1) - 909.3880 < tol && "X_cut(end,end) does not match expected value!");
    assert(Y_cut(Y_cut.rows()-1,Y_cut.cols()-1) - 211.8634 < tol && "Y_cut(end,end) does not match expected value!");

    std::cout << "\nALL ASSERTIONS PASSED." << std::endl;
    return 0;
}