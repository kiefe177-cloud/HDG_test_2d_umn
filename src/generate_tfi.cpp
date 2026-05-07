#include <iostream>
#include <cmath>
#include <algorithm>

#include "hdg/generate_tfi.h"
#include "hdg/Spline.h"
#include "hdg/meshgrid.h"

// =========================================================================
//  boundarySpline
//  Match MATLAB: unique-stable, then flip if y(0) > y(end), then
//  arc-length-parameterized cubic spline.
// =========================================================================
static std::pair<Spline, Spline> boundarySpline(
    const Eigen::VectorXd& x_in, const Eigen::VectorXd& y_in)
{
    const int N = static_cast<int>(x_in.size());

    // Remove exact duplicate (x,y) rows while preserving order.
    std::vector<double> x_uniq, y_uniq;
    x_uniq.reserve(N);
    y_uniq.reserve(N);
    for (int i = 0; i < N; ++i) {
        bool is_dup = false;
        for (int j = 0; j < static_cast<int>(x_uniq.size()); ++j) {
            if (x_in(i) == x_uniq[j] && y_in(i) == y_uniq[j]) {
                is_dup = true;
                break;
            }
        }
        if (!is_dup) {
            x_uniq.push_back(x_in(i));
            y_uniq.push_back(y_in(i));
        }
    }

    const int M = static_cast<int>(x_uniq.size());

    // Flip if endpoints are reversed in y.
    if (y_uniq[0] > y_uniq[M - 1]) {
        std::reverse(x_uniq.begin(), x_uniq.end());
        std::reverse(y_uniq.begin(), y_uniq.end());
    }

    // Cumulative arc length, normalized to [0,1].
    std::vector<double> s_vals(M);
    s_vals[0] = 0.0;
    for (int i = 1; i < M; ++i) {
        const double dx = x_uniq[i] - x_uniq[i - 1];
        const double dy = y_uniq[i] - y_uniq[i - 1];
        s_vals[i] = s_vals[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    double total = s_vals.back();
    if (total == 0.0) total = 1.0;

    Eigen::VectorXd t(M), x(M), y(M);
    for (int i = 0; i < M; ++i) {
        t(i) = s_vals[i] / total;
        x(i) = x_uniq[i];
        y(i) = y_uniq[i];
    }

    Spline sx, sy;
    sx.fit(t, x);
    sy.fit(t, y);
    return {sx, sy};
}

// =========================================================================
//  boundaryEval — map [-1,1] -> [0,1], evaluate the two splines.
// =========================================================================
static std::pair<double, double> boundaryEval(
    double val, const Spline& sx, const Spline& sy)
{
    const double t = std::clamp((val + 1.0) / 2.0, 0.0, 1.0);
    return {sx.eval(t), sy.eval(t)};
}

// =========================================================================
//  apply_stretching — vector form (used internally).
// =========================================================================
Eigen::VectorXd apply_stretching(
    const Eigen::VectorXd& uniform, double strength, std::string target)
{
    Eigen::VectorXd s = uniform;
    const int n = static_cast<int>(s.size());

    if (strength == 0.0 || target == "none") return s;
    std::transform(target.begin(), target.end(), target.begin(), ::tolower);

    for (int i = 0; i < n; ++i) {
        const double val = s(i);
        const double t = (val + 1.0) / 2.0;

        if (target == "start") {
            const double t_str = (strength > 0)
                ? std::tanh(strength * t) / std::tanh(strength)
                : std::sinh(strength * t) / std::sinh(strength);
            s(i) = 2.0 * t_str - 1.0;
        }
        else if (target == "end") {
            const double rev_t = (strength > 0)
                ? std::tanh(strength * (1.0 - t)) / std::tanh(strength)
                : std::sinh(strength * (1.0 - t)) / std::sinh(strength);
            s(i) = 2.0 * (1.0 - rev_t) - 1.0;
        }
        else if (target == "center") {
            const double beta = std::abs(strength);
            s(i) = (strength > 0)
                ? std::asinh(beta * val) / std::asinh(beta)
                : std::tanh(strength * val) / std::tanh(strength + 1e-16);
        }
        else if (target == "both") {
            if (strength > 0) {
                s(i) = std::tanh(strength * val) / std::tanh(strength);
            } else {
                const double beta = std::abs(strength);
                s(i) = std::asinh(beta * val) / std::asinh(beta);
            }
        }
        s(i) = std::clamp(s(i), -1.0, 1.0);
    }
    return s;
}

// Scalar convenience used inside the boundary lambdas.
static double apply_stretching_scalar(
    double val, double strength, const std::string& target)
{
    if (strength == 0.0 || target == "none") return val;
    Eigen::VectorXd v(1);
    v << val;
    return apply_stretching(v, strength, target)(0);
}

// =========================================================================
//  compute_tfi_matrices — bilinear TFI with corner correction.
//  Single overload taking BoundaryFunc lambdas; matches MATLAB compute_tfi.
// =========================================================================
void compute_tfi_matrices(
    const Eigen::MatrixXd& Phi,
    const Eigen::MatrixXd& Psi,
    BoundaryFunc leftB, BoundaryFunc rightB,
    BoundaryFunc bottomB, BoundaryFunc topB,
    double x_sw, double y_sw, double x_se, double y_se,
    double x_ne, double y_ne, double x_nw, double y_nw,
    Eigen::MatrixXd& OutX, Eigen::MatrixXd& OutY)
{
    const long rows = Phi.rows();
    const long cols = Phi.cols();
    OutX.resize(rows, cols);
    OutY.resize(rows, cols);

    for (long r = 0; r < rows; ++r) {
        for (long c = 0; c < cols; ++c) {
            const double phi = Phi(r, c);
            const double psi = Psi(r, c);

            auto [xL, yL] = leftB  (psi);
            auto [xR, yR] = rightB (psi);
            auto [xB, yB] = bottomB(phi);
            auto [xT, yT] = topB   (phi);

            const double term1_x = (1.0 - phi) / 2.0 * xL + (1.0 + phi) / 2.0 * xR;
            const double term1_y = (1.0 - phi) / 2.0 * yL + (1.0 + phi) / 2.0 * yR;

            const double term2_x = (1.0 - psi) / 2.0 * xB + (1.0 + psi) / 2.0 * xT;
            const double term2_y = (1.0 - psi) / 2.0 * yB + (1.0 + psi) / 2.0 * yT;

            const double term3_x = (1.0 - phi) * (1.0 - psi) / 4.0 * x_sw
                                 + (1.0 - phi) * (1.0 + psi) / 4.0 * x_nw
                                 + (1.0 + phi) * (1.0 - psi) / 4.0 * x_se
                                 + (1.0 + phi) * (1.0 + psi) / 4.0 * x_ne;

            const double term3_y = (1.0 - phi) * (1.0 - psi) / 4.0 * y_sw
                                 + (1.0 - phi) * (1.0 + psi) / 4.0 * y_nw
                                 + (1.0 + phi) * (1.0 - psi) / 4.0 * y_se
                                 + (1.0 + phi) * (1.0 + psi) / 4.0 * y_ne;

            OutX(r, c) = term1_x + term2_x - term3_x;
            OutY(r, c) = term1_y + term2_y - term3_y;
        }
    }
}

// =========================================================================
//  generate_tfi_mesh — match MATLAB generate_tfi_mesh(X, Y, Ne_x).
//  Stores the four boundary lambdas on res.f so the driver can plot the
//  curved walls at high resolution after the solve.
// =========================================================================
TFI_Mesh generate_tfi_mesh(
    const Eigen::MatrixXd& X_in, const Eigen::MatrixXd& Y_in, int Ne_x)
{
    TFI_Mesh res;

    const int n_grid_pts = Ne_x + 1;
    res.n_grid_pts = n_grid_pts;

    // Stretching parameters (match MATLAB).
    const double      strength_phi = 0.0;
    const std::string target_phi   = "end";
    const double      strength_psi = 0.0;
    const std::string target_psi   = "end";

    // Boundary curve splines (left/right run in psi, bottom/top run in phi).
    auto [xl_x, xl_y] = boundarySpline(X_in.col(0),                  Y_in.col(0));
    auto [xr_x, xr_y] = boundarySpline(X_in.col(X_in.cols() - 1),    Y_in.col(Y_in.cols() - 1));
    auto [xt_x, xt_y] = boundarySpline(X_in.row(X_in.rows() - 1),    Y_in.row(Y_in.rows() - 1));
    auto [xb_x, xb_y] = boundarySpline(X_in.row(0),                  Y_in.row(0));

    // Uniform computational vectors.
    res.phi_vec      = Eigen::VectorXd::LinSpaced(n_grid_pts, -1.0, 1.0);
    res.psi_vec      = Eigen::VectorXd::LinSpaced(n_grid_pts, -1.0, 1.0);
    res.phi_comp_vec = res.phi_vec;
    res.psi_comp_vec = res.psi_vec;

    auto [Phi, Psi] = meshgrid(res.phi_comp_vec, res.psi_comp_vec);
    res.Phi_tfi = Phi;
    res.Psi_tfi = Psi;

    // Bake stretching into boundary functions. Capture by VALUE so the
    // lambdas remain valid after generate_tfi_mesh returns — they're stored
    // on res.f and called later by the driver / plotting code.
    res.f.left_boundary =
        [xl_x, xl_y, strength_psi, target_psi](double psi) {
            return boundaryEval(
                apply_stretching_scalar(psi, strength_psi, target_psi),
                xl_x, xl_y);
        };
    res.f.right_boundary =
        [xr_x, xr_y, strength_psi, target_psi](double psi) {
            return boundaryEval(
                apply_stretching_scalar(psi, strength_psi, target_psi),
                xr_x, xr_y);
        };
    res.f.bottom_boundary =
        [xb_x, xb_y, strength_phi, target_phi](double phi) {
            return boundaryEval(
                apply_stretching_scalar(phi, strength_phi, target_phi),
                xb_x, xb_y);
        };
    res.f.top_boundary =
        [xt_x, xt_y, strength_phi, target_phi](double phi) {
            return boundaryEval(
                apply_stretching_scalar(phi, strength_phi, target_phi),
                xt_x, xt_y);
        };

    // Physical corners.
    std::tie(res.f.x_sw, res.f.y_sw) = res.f.left_boundary (-1.0);
    std::tie(res.f.x_nw, res.f.y_nw) = res.f.left_boundary ( 1.0);
    std::tie(res.f.x_se, res.f.y_se) = res.f.right_boundary(-1.0);
    std::tie(res.f.x_ne, res.f.y_ne) = res.f.right_boundary( 1.0);

    // TFI grid.
    compute_tfi_matrices(
        res.Phi_tfi, res.Psi_tfi,
        res.f.left_boundary, res.f.right_boundary,
        res.f.bottom_boundary, res.f.top_boundary,
        res.f.x_sw, res.f.y_sw, res.f.x_se, res.f.y_se,
        res.f.x_ne, res.f.y_ne, res.f.x_nw, res.f.y_nw,
        res.X_tfi, res.Y_tfi);

    // Test point at (phi=1, psi=-1).
    res.phi_val = 1.0;
    res.psi_val = -1.0;

    Eigen::MatrixXd sP(1, 1), sS(1, 1);
    sP << res.phi_val;
    sS << res.psi_val;

    Eigen::MatrixXd sX, sY;
    compute_tfi_matrices(
        sP, sS,
        res.f.left_boundary, res.f.right_boundary,
        res.f.bottom_boundary, res.f.top_boundary,
        res.f.x_sw, res.f.y_sw, res.f.x_se, res.f.y_se,
        res.f.x_ne, res.f.y_ne, res.f.x_nw, res.f.y_nw,
        sX, sY);

    res.x_p = sX(0, 0);
    res.y_p = sY(0, 0);

    return res;
}