#pragma once
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>

class Spline;

// Matches MATLAB function handle: @(val) -> [x, y]
using BoundaryFunc = std::function<std::pair<double,double>(double)>;

struct TFI_Mesh {
    Eigen::MatrixXd X_tfi, Y_tfi;

    Eigen::VectorXd phi_vec, psi_vec;
    Eigen::VectorXd phi_comp_vec, psi_comp_vec;

    Eigen::MatrixXd Phi_tfi, Psi_tfi;

    double x_p, y_p;

    int n_grid_pts;
    double phi_val, psi_val;

    // Mirrors MATLAB's `f` struct: physical corners + the four boundary
    // function handles. Storing the lambdas here (with by-value capture in
    // generate_tfi_mesh) lets the driver evaluate the curved walls at any
    // resolution after the mesh is built — same pattern as MATLAB.
    struct Boundaries {
        double x_sw, y_sw, x_nw, y_nw;
        double x_se, y_se, x_ne, y_ne;

        BoundaryFunc left_boundary;
        BoundaryFunc right_boundary;
        BoundaryFunc bottom_boundary;
        BoundaryFunc top_boundary;
    } f;
};

// --- Exported Helper Functions ---

Eigen::VectorXd apply_stretching(const Eigen::VectorXd& uniform, double strength, std::string target);

// Bilinear TFI with corner correction. Takes boundary function handles —
// matches MATLAB compute_tfi.
void compute_tfi_matrices(
    const Eigen::MatrixXd& Phi, const Eigen::MatrixXd& Psi,
    BoundaryFunc leftB, BoundaryFunc rightB,
    BoundaryFunc bottomB, BoundaryFunc topB,
    double x_sw, double y_sw, double x_se, double y_se,
    double x_ne, double y_ne, double x_nw, double y_nw,
    Eigen::MatrixXd& OutX, Eigen::MatrixXd& OutY
);

// Main entry point — takes Ne_x (number of elements).
TFI_Mesh generate_tfi_mesh(const Eigen::MatrixXd& X_in, const Eigen::MatrixXd& Y_in, int Ne_x);