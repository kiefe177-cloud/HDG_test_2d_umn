#pragma once
#include <Eigen/Dense>
#include <vector>
#include "Spline.h"
#include "hdg/generate_tfi.h"   // for TFI_Mesh::Boundaries


class BicubicSpline {
private:
    std::vector<Spline> row_splines;
    Eigen::VectorXd phi_grid;
    bool is_initialized = false;

public:
    BicubicSpline() = default;


    void fit(const Eigen::VectorXd& phi_vec, const Eigen::VectorXd& psi_vec, const Eigen::MatrixXd& data) {
        if (data.rows() != phi_vec.size() || data.cols() != psi_vec.size()) {
            throw std::invalid_argument("BicubicSpline: Data dimensions do not match grid vectors.");
        }

        phi_grid = phi_vec;
        int n_rows = data.rows();

        row_splines.clear();
        row_splines.resize(n_rows);

        for (int i = 0; i < n_rows; ++i) {
            Eigen::VectorXd row_data = data.row(i);
            row_splines[i].fit(psi_vec, row_data);
        }
        is_initialized = true;
    }

    double eval(double phi, double psi) const {
        if (!is_initialized) return 0.0;

        int n_rows = row_splines.size();
        Eigen::VectorXd vals_at_phi_nodes(n_rows);

        for (int i = 0; i < n_rows; ++i) {
            vals_at_phi_nodes(i) = row_splines[i].eval(psi);
        }


        Spline phi_spline;
        phi_spline.fit(phi_grid, vals_at_phi_nodes);

        return phi_spline.eval(phi);
    }
};


struct GeomMetrics {
    Eigen::VectorXd phi_global_vec;
    Eigen::VectorXd psi_global_vec;

    BicubicSpline Fx_interpolant;
    BicubicSpline Fy_interpolant;

    Eigen::MatrixXd Dr1d;

    // Boundary struct (corners + boundary function handles), copied
    // from TFI_Mesh::f. Mirrors MATLAB: geom.f = f.
    TFI_Mesh::Boundaries f;
};

GeomMetrics comp_space_metrics(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    int p,
    const Eigen::VectorXd& phi_s,
    const Eigen::VectorXd& psi_s,
    const TFI_Mesh::Boundaries& f);