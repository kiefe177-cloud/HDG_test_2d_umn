#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <highfive/H5File.hpp>

#include "hdg/generate_tfi.h"
#include "hdg/Spline.h" 
#include "parameters/M5p8_Params.h"
#include <iomanip>

// --- HELPER FOR FLOATING POINT ---
bool is_approx(double a, double b, double eps = 1e-4) {
    return std::abs(a - b) < eps;
}

/**
 * TEST 1: Grid Stretching
 */
void test_stretching() {
    std::cout << "Running: test_stretching..." << std::endl;
    
    Eigen::VectorXd uniform = Eigen::VectorXd::LinSpaced(10, -1.0, 1.0);
    
    // Testing exported helper from generate_tfi.h
    Eigen::VectorXd start_str = apply_stretching(uniform, 2.0, "start");
    assert(is_approx(start_str(0), -1.0));
    assert(is_approx(start_str(9), 1.0));
    assert(is_approx(start_str(1),-0.54641)); 
    assert(is_approx(start_str(8),0.95939549)); 


    std::cout << "  [PASS] Stretching logic." << std::endl;
}

/**
 * TEST 2: TFI Identity (The Unit Square)
 */
void test_tfi_unit_square() {
    std::cout << "Running: test_tfi_unit_square..." << std::endl;

    // 3x3 logical grid in [-1, 1]
    Eigen::MatrixXd Phi(3, 3), Psi(3, 3);
    Phi << -1,  0,  1,
           -1,  0,  1,
           -1,  0,  1;
    Psi << -1, -1, -1,
            0,  0,  0,
            1,  1,  1;

    // 2. Use the REAL Spline class now
    Spline xb_x, xb_y, xt_x, xt_y, xl_x, xl_y, xr_x, xr_y;
    
    // Linear parameter vector t in [0, 1]
    Eigen::VectorXd t_vec = Eigen::VectorXd::LinSpaced(2, 0.0, 1.0);
    Eigen::VectorXd pos_neg = Eigen::VectorXd::LinSpaced(2, -1.0, 1.0);
    Eigen::VectorXd ones = Eigen::VectorXd::Constant(2, 1.0);
    Eigen::VectorXd neg_ones = Eigen::VectorXd::Constant(2, -1.0);

    // Fit boundaries for a [-1, 1] square
    xb_x.fit(t_vec, pos_neg);  xb_y.fit(t_vec, neg_ones); // Bottom
    xt_x.fit(t_vec, pos_neg);  xt_y.fit(t_vec, ones);     // Top
    xl_x.fit(t_vec, neg_ones); xl_y.fit(t_vec, pos_neg);  // Left
    xr_x.fit(t_vec, ones);     xr_y.fit(t_vec, pos_neg);  // Right

    // Wrap each spline pair as a BoundaryFunc lambda. Spline.eval expects
    // t in [0,1], so we map the [-1,1] computational coordinate ourselves.
    auto eval_spline = [](double v, const Spline& sx, const Spline& sy) {
        double t = std::clamp((v + 1.0) / 2.0, 0.0, 1.0);
        return std::pair<double,double>{sx.eval(t), sy.eval(t)};
    };
    BoundaryFunc leftB   = [&, xl_x, xl_y](double v) { return eval_spline(v, xl_x, xl_y); };
    BoundaryFunc rightB  = [&, xr_x, xr_y](double v) { return eval_spline(v, xr_x, xr_y); };
    BoundaryFunc bottomB = [&, xb_x, xb_y](double v) { return eval_spline(v, xb_x, xb_y); };
    BoundaryFunc topB    = [&, xt_x, xt_y](double v) { return eval_spline(v, xt_x, xt_y); };

    Eigen::MatrixXd OutX, OutY;

    // 3. Call with corrected corner arguments from TFI_Mesh::Boundaries logic
    compute_tfi_matrices(Phi, Psi,
                         leftB, rightB, bottomB, topB,
                         -1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0, 
                         OutX, OutY);

    // Assert Corners
    assert(is_approx(OutX(0,0), -1.0)); // SW
    assert(is_approx(OutY(2,2),  1.0)); // NE

    // Assert Center point (should be 0,0)
    assert(is_approx(OutX(1,1), 0.0)); 
    assert(is_approx(OutY(1,1), 0.0));

    std::cout << "  [PASS] Unit square TFI." << std::endl;
}
// Test 3: Full TFI Mesh Generation (matches MATLAB generate_tfi_mesh)
void test_full_tfi_mesh() {
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

    Eigen::MatrixXd X = scale_factor * x_h5t / L;
    Eigen::MatrixXd Y = scale_factor * y_h5t / L;

    Eigen::MatrixXd X_cut = X.block(0, 49, 280, 1152).eval();
    Eigen::MatrixXd Y_cut = Y.block(0, 49, 280, 1152).eval();

    std::cout << "\n" << std::endl;

    std::cout << "X shape: " << X_cut.rows() << " x " << X_cut.cols() << std::endl;
    std::cout << "Y shape: " << Y_cut.rows() << " x " << Y_cut.cols() << std::endl;

    int Ne_x = 16;
    TFI_Mesh tfi = generate_tfi_mesh(X_cut, Y_cut, Ne_x);

    int n = tfi.n_grid_pts;
    std::cout << std::scientific << std::setprecision(10);

    // ---- Grid shape ----
    std::cout << "X_tfi shape: " << tfi.X_tfi.rows() << " x " << tfi.X_tfi.cols() << std::endl;
    assert(tfi.X_tfi.rows() == n && "Wrong row count!");
    assert(tfi.X_tfi.cols() == n && "Wrong col count!");

    // ---- No NaN/Inf ----
    assert(!tfi.X_tfi.hasNaN() && "X_tfi has NaN!");
    assert(!tfi.Y_tfi.hasNaN() && "Y_tfi has NaN!");
    assert(tfi.X_tfi.allFinite() && "X_tfi has Inf!");
    assert(tfi.Y_tfi.allFinite() && "Y_tfi has Inf!");

    // ---- Corners should match boundary struct ----
    double tol = 1e-4;
    assert(std::abs(tfi.X_tfi(0, 0)   - tfi.f.x_sw) < tol && "SW x mismatch!");
    assert(std::abs(tfi.Y_tfi(0, 0)   - tfi.f.y_sw) < tol && "SW y mismatch!");
    assert(std::abs(tfi.X_tfi(0, n-1) - tfi.f.x_se) < tol && "SE x mismatch!");
    assert(std::abs(tfi.Y_tfi(0, n-1) - tfi.f.y_se) < tol && "SE y mismatch!");
    assert(std::abs(tfi.X_tfi(n-1, 0) - tfi.f.x_nw) < tol && "NW x mismatch!");
    assert(std::abs(tfi.Y_tfi(n-1, 0) - tfi.f.y_nw) < tol && "NW y mismatch!");
    assert(std::abs(tfi.X_tfi(n-1, n-1) - tfi.f.x_ne) < tol && "NE x mismatch!");
    assert(std::abs(tfi.Y_tfi(n-1, n-1) - tfi.f.y_ne) < tol && "NE y mismatch!");

    // ---- Test point (phi=1, psi=-1) should equal SE corner ----
    assert(std::abs(tfi.x_p - tfi.f.x_se) < tol && "Test point x != SE!");
    assert(std::abs(tfi.y_p - tfi.f.y_se) < tol && "Test point y != SE!");

    // ---- Print values for MATLAB comparison ----
    std::cout << "\n--- Corners ---" << std::endl;
    std::cout << "SW: (" << tfi.f.x_sw << ", " << tfi.f.y_sw << ")" << std::endl;
    std::cout << "SE: (" << tfi.f.x_se << ", " << tfi.f.y_se << ")" << std::endl;
    std::cout << "NW: (" << tfi.f.x_nw << ", " << tfi.f.y_nw << ")" << std::endl;
    std::cout << "NE: (" << tfi.f.x_ne << ", " << tfi.f.y_ne << ")" << std::endl;

    std::cout << "\n--- Grid Corners ---" << std::endl;
    std::cout << "X_tfi(1,1)       = " << tfi.X_tfi(0, 0) << std::endl;
    std::cout << "X_tfi(1,end)     = " << tfi.X_tfi(0, n-1) << std::endl;
    std::cout << "X_tfi(end,1)     = " << tfi.X_tfi(n-1, 0) << std::endl;
    std::cout << "X_tfi(end,end)   = " << tfi.X_tfi(n-1, n-1) << std::endl;
    std::cout << "Y_tfi(1,1)       = " << tfi.Y_tfi(0, 0) << std::endl;
    std::cout << "Y_tfi(1,end)     = " << tfi.Y_tfi(0, n-1) << std::endl;
    std::cout << "Y_tfi(end,1)     = " << tfi.Y_tfi(n-1, 0) << std::endl;
    std::cout << "Y_tfi(end,end)   = " << tfi.Y_tfi(n-1, n-1) << std::endl;

    int mid = n / 2;
    std::cout << "\n--- Mid Point ---" << std::endl;
    std::cout << "X_tfi(" << mid+1 << "," << mid+1 << ") = " << tfi.X_tfi(mid, mid) << std::endl;
    std::cout << "Y_tfi(" << mid+1 << "," << mid+1 << ") = " << tfi.Y_tfi(mid, mid) << std::endl;

    std::cout << "\n--- Test Point ---" << std::endl;
    std::cout << "x_p = " << tfi.x_p << std::endl;
    std::cout << "y_p = " << tfi.y_p << std::endl;

    std::cout << "X_tfi(mid,mid)   = " << tfi.X_tfi(mid, mid) << std::endl;
    std::cout << "Y_tfi(mid,mid)   = " << tfi.Y_tfi(mid, mid) << std::endl;

    assert(std::abs(tfi.X_tfi(mid,mid) - 4.646899e+2) < tol && "Mid X mismatch!");
    assert(std::abs(tfi.Y_tfi(mid,mid) - 89.696375e+0) < tol && "Mid Y mismatch!");

    assert(std::abs(tfi.X_tfi(3,3) - 1.829118e+2) < tol && "(4,4) X mismatch!");
    assert(std::abs(tfi.Y_tfi(3,3) - 33.58669e+0) < tol && "(4,4) Y mismatch!");

    // ---- Paste these into MATLAB to compare ----
    // fprintf('SW: (%.10e, %.10e)\n', f.x_sw, f.y_sw)
    // fprintf('SE: (%.10e, %.10e)\n', f.x_se, f.y_se)
    // fprintf('NW: (%.10e, %.10e)\n', f.x_nw, f.y_nw)
    // fprintf('NE: (%.10e, %.10e)\n', f.x_ne, f.y_ne)
    // fprintf('X_tfi(1,1)=%.10e\n', X_tfi(1,1))
    // fprintf('X_tfi(1,end)=%.10e\n', X_tfi(1,end))
    // fprintf('X_tfi(end,1)=%.10e\n', X_tfi(end,1))
    // fprintf('X_tfi(end,end)=%.10e\n', X_tfi(end,end))
    // mid = ceil(n_grid_pts/2);
    // fprintf('X_tfi(%d,%d)=%.10e\n', mid, mid, X_tfi(mid,mid))
    // fprintf('Y_tfi(%d,%d)=%.10e\n', mid, mid, Y_tfi(mid,mid))
    // fprintf('x_p=%.10e\n', x_p)
    // fprintf('y_p=%.10e\n', y_p)

    std::cout << "  [PASS] Full TFI mesh generation." << std::endl;
}

int main() {
    test_stretching();
    test_tfi_unit_square();
    test_full_tfi_mesh();
    std::cout << "\nALL TESTS PASSED!" << std::endl;
    return 0;
}