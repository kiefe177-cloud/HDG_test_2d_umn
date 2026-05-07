#include <Eigen/Dense>
#include <iostream>
#include <highfive/H5File.hpp>
#include <string>

#include "hdg/all.h"
#include "parameters/M5p8_Params.h"

using namespace Eigen;

int main(){
    
    // #######################
    // 0: Scenario Parameters
    // #######################
    SimulationParams params = get_M5p8_params();

    int nvar = 5;
    int p = 6;
    int Ne_x = 16;

    // VectorXd idx_vec = VectorXd::LinSpaced(50,1,50);
    // VectorXd F = VectorXd::LinSpaced(50,2e-5,1.5e-4);

    VectorXd idx_vec(1);  idx_vec << 37;
    VectorXd F = VectorXd::LinSpaced(50, 2e-5, 1.5e-4);

    double gam = params.gam;

    std::vector<int> m_vec = {0};

    for (int m : m_vec){

        for (int i = 0; i <idx_vec.size(); ++i){

            int idx = static_cast<int>(idx_vec(i)) - 1;

            double F_final = F(idx);

            double omega = F_final * params.Re_L;
            double omega_dim = omega * params.ue / params.L;
            double F_dim = omega_dim / (2* M_PI);

            // Test print statements
            std::cout << "Re_L: " << params.Re_L << std::endl;
            std::cout << "Omega_dim (e5): " << omega_dim/1e5 << std::endl;
            std::cout << "F_dim (kHz): " << F_dim/1000 << std::endl;
            std::cout << "Idx: " << idx+1 << " | Omega: " << omega << " | F: "<< F_final << std::endl;
            
            double scale_factor = 1.0;
            
            // #######################
            // 1: Grid Generation
            // #######################

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

            double L = params.L;  // Use the L parameter from the simulation params

            Eigen::MatrixXd x_h5t = x_h5.transpose();
            Eigen::MatrixXd y_h5t = y_h5.transpose();

            std::cout << "Transposed shape: " << x_h5t.rows() << " x " << x_h5t.cols() << std::endl;

            Eigen::MatrixXd X = scale_factor * x_h5t / L;
            Eigen::MatrixXd Y = scale_factor * y_h5t / L;

            // MATLAB: X(1:280, 50:1201)  →  rows 1-280, cols 50-1201
            // C++:    rows 0-279, cols 49-1200

            X = X.block(0, 49, 280, 1152).eval();
            Y = Y.block(0, 49, 280, 1152).eval();
            
            std::cout << "X shape: " << X.rows() << " x " << X.cols() << std::endl;
            std::cout << "Y shape: " << Y.rows() << " x " << Y.cols() << std::endl;

            TFI_Mesh mesh = generate_tfi_mesh(X, Y, Ne_x);
            
            // Access results like:
            std::cout << "TFI grid: " << mesh.X_tfi.rows() << " x " << mesh.X_tfi.cols() << std::endl;
            std::cout << "Test point: (" << mesh.x_p << ", " << mesh.y_p << ")" << std::endl;

            // Four Elliptic Options: 
            //                      solve (TTM),
            //                      ortho (Wall Ortho only),
            //                      full Ortho,
            //                      Steger-Sorenson

            EllipticMethod method = EllipticMethod::Ortho;

            int n_elliptic_iter = 100000;
            double sor_omega = 1.5;
            double convergence_tol = 1e-12;

            OrthoOpts ortho_opts;
            ortho_opts.ortho_tol = convergence_tol;
            ortho_opts.march_tol = convergence_tol;

            EllipticResult result;
            /*
            switch (method) {
                case EllipticMethod::TTM:
                    result = tfi_elliptic_solve(
                        mesh.X_tfi, mesh.Y_tfi, mesh.phi_comp_vec, mesh.psi_comp_vec,
                        mesh.n_grid_pts, n_elliptic_iter, sor_omega, convergence_tol);
                    break;

                case EllipticMethod::Ortho:
                    result = tfi_elliptic_orthogonal(
                        mesh.X_tfi, mesh.Y_tfi, mesh.phi_comp_vec, mesh.psi_comp_vec,
                        mesh.n_grid_pts, n_elliptic_iter, sor_omega, convergence_tol, ortho_opts);
                    break;

                case EllipticMethod::FullOrtho:
                    result = tfi_elliptic_full_orthogonal(
                        mesh.X_tfi, mesh.Y_tfi, mesh.phi_comp_vec, mesh.psi_comp_vec,
                        mesh.n_grid_pts, n_elliptic_iter, sor_omega, convergence_tol, ortho_opts);
                    break;

                case EllipticMethod::StegerSorenson:
                    result = tfi_elliptic_steger_sorenson_v2(
                        mesh.X_tfi, mesh.Y_tfi, mesh.phi_comp_vec, mesh.psi_comp_vec,
                        mesh.n_grid_pts, n_elliptic_iter, sor_omega, convergence_tol, ortho_opts);
                    break;
            } */
        }
    }


    return 0;
}
