#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "hdg/comp_space_metrics.h"
#include "hdg/generate_tfi.h"  // for TFI_Mesh::Boundaries

// Helper to save CSV for Python plotting
void save_csv(const std::string& filename, const Eigen::MatrixXd& mat) {
    std::ofstream file(filename);
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    file << mat.format(CSVFormat);
}

// True math function for Physical Y coordinate
// A "Hump" shape: Linear in Y, but curved in X
double exact_y(double phi, double psi) {
    return psi + 0.5 * std::sin(phi * 3.14159265359);
}

int main() {
    std::cout << "1. Generating 5x5 Coarse Grid..." << std::endl;
    
    int rows = 5; // Phi direction
    int cols = 5; // Psi direction
    
    Eigen::VectorXd phi_vec = Eigen::VectorXd::LinSpaced(rows, 0.0, 1.0);
    Eigen::VectorXd psi_vec = Eigen::VectorXd::LinSpaced(cols, 0.0, 1.0);
    
    Eigen::MatrixXd X_grid(rows, cols);
    Eigen::MatrixXd Y_grid(rows, cols);

    // Fill grid with known data
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            // Physical X is just linear phi
            X_grid(i,j) = phi_vec(i); 
            // Physical Y is our "Hump" function
            Y_grid(i,j) = exact_y(phi_vec(i), psi_vec(j)); 
        }
    }

    std::cout << "2. Running comp_space_metrics..." << std::endl;
    // This test exercises only the interpolation, so we pass a default-
    // constructed Boundaries (zero corners, empty boundary lambdas). The
    // value is stored on geom.f but never used in this test.
    TFI_Mesh::Boundaries f_dummy{};

    // We pass p=1 for the derivative order (doesn't matter for this interpolation test)
    GeomMetrics metrics = comp_space_metrics(X_grid, Y_grid, 1, phi_vec, psi_vec, f_dummy);

    std::cout << "3. Testing Single Point Interpolation..." << std::endl;
    
    // Pick a point NOT on the grid nodes (e.g., 0.5, 0.5)
    // At 5x5, indices are 0, 0.25, 0.5, 0.75, 1.0. 
    // Let's pick 0.33, 0.33 to be safe and force actual interpolation.
    double test_phi = 0.33;
    double test_psi = 0.33;

    double interp_x = metrics.Fx_interpolant.eval(test_phi, test_psi);
    double interp_y = metrics.Fy_interpolant.eval(test_phi, test_psi);
    
    double true_x = test_phi;
    double true_y = exact_y(test_phi, test_psi);

    std::cout << std::fixed << std::setprecision(15); // Show 15 decimal places

    std::cout << "   Test Point (phi=" << test_phi << ", psi=" << test_psi << ")" << std::endl;
    std::cout << "   Interp Y: " << interp_y << std::endl;
    std::cout << "   Exact Y:  " << true_y << std::endl;
    std::cout << "   Error:    " << std::abs(interp_y - true_y) << std::endl;

    // Splines are approximations, so error won't be zero, but should be small (e.g., < 1e-3)
    assert(std::abs(interp_y - true_y) < 1e-2 && "Interpolation error too high!");
    std::cout << "[PASS] Interpolation accuracy is good." << std::endl;

    // ---------------------------------------------------------
    // 4. Generate Fine Surface for Visualization
    // ---------------------------------------------------------
    int fine_pts = 50;
    Eigen::VectorXd phi_fine = Eigen::VectorXd::LinSpaced(fine_pts, 0, 1);
    Eigen::VectorXd psi_fine = Eigen::VectorXd::LinSpaced(fine_pts, 0, 1);
    
    Eigen::MatrixXd Z_output(fine_pts, fine_pts); // Storing the interpolated Y values

    for(int i=0; i<fine_pts; ++i) {
        for(int j=0; j<fine_pts; ++j) {
            Z_output(i,j) = metrics.Fy_interpolant.eval(phi_fine(i), psi_fine(j));
        }
    }
    
    save_csv(std::string(OUTPUT_DIR_TEST) + "/interp_surface.csv", Z_output);
    save_csv(std::string(OUTPUT_DIR_TEST) + "/orig_grid_y.csv", Y_grid);
    std::cout << "Saved CSV data for plotting." << std::endl;

    return 0;
}