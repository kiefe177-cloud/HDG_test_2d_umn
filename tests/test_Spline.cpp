#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>  // <--- Required for assert()
#include "hdg/Spline.h"

// Helper to check floating point equality with tolerance
bool is_close(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) < tol;
}

void save_csv(const std::string& filename, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(15);
    for (int i = 0; i < x.size(); ++i) {
        file << x(i) << "," << y(i) << "\n";
    }
}

int main() {
    
    // Test 1: Cubic Polynomial Reconstruction
    std::cout << "Running Test 1: Cubic Polynomial Reconstruction..." << std::endl;
    
    // 1. Make liine
    Eigen::VectorXd x_poly(5);
    Eigen::VectorXd y_poly(5);
    for(int i=0; i<5; ++i) {
        x_poly(i) = static_cast<double>(i);      // 0, 1, 2, 3, 4
        y_poly(i) = std::pow(x_poly(i), 3.0);    // 0, 1, 8, 27, 64
    }

    Spline s_poly;
    s_poly.fit(x_poly, y_poly);

    // Check a value "in between" knots (e.g., x = 1.5)
    // Expected: 1.5^3 = 3.375
    double val_check = s_poly.eval(1.5);
    double val_exact = 3.375;

    std::cout << "  Eval(1.5): " << val_check << " (Expected: " << val_exact << ")" << std::endl;
    
    // ASSERTION
    assert(is_close(val_check, val_exact) && "Cubic Spline failed to fit x^3!");


    // Test 2: Lower bound check
    std::cout << "Running Test 2: Linear Reconstruction..." << std::endl;
    Eigen::VectorXd x_lin(3); x_lin << 0, 5, 10;
    Eigen::VectorXd y_lin(3); y_lin << 0, 10, 20; // y = 2x
    
    Spline s_lin;
    s_lin.fit(x_lin, y_lin);
    
    double lin_check = s_lin.eval(2.5); // Should be 5.0
    assert(is_close(lin_check, 5.0) && "Spline failed to fit straight line!");

    std::cout << "ALL INTERNAL ASSERTIONS PASSED." << std::endl;

    // Generate sin wave
    int n_knots = 6;
    Eigen::VectorXd x_knots(n_knots);
    Eigen::VectorXd y_knots(n_knots);
    for(int i = 0; i < n_knots; ++i) {
        x_knots(i) = static_cast<double>(i);
        y_knots(i) = std::sin(x_knots(i)); 
    }

    Spline s;
    s.fit(x_knots, y_knots);

    int n_query = 101; 
    Eigen::VectorXd x_query = Eigen::VectorXd::LinSpaced(n_query, 0.0, 5.0);
    Eigen::VectorXd y_query = s.eval_vector(x_query);

    save_csv(std::string(OUTPUT_DIR_TEST) + "/knots_data.csv", x_knots, y_knots);
    save_csv(std::string(OUTPUT_DIR_TEST) + "/cpp_spline_output.csv", x_query, y_query);
    std::cout << "Generated CSV files for MATLAB verification." << std::endl;

    return 0;
}
