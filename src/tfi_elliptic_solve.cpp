#include <iostream>
#include <cmath>
#include <algorithm>

#include "hdg/tfi_elliptic_solve.h"

// =====================================================================
//  Skewness: deviation from orthogonality in degrees (0 = orthogonal)
// =====================================================================
static Eigen::MatrixXd compute_skewness(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    Eigen::MatrixXd skew = Eigen::MatrixXd::Zero(N, N);

    // Interior points: central differences in both directions
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double xp = (X(k, l+1) - X(k, l-1)) / (2.0 * dphi);
            double xs = (X(k+1, l) - X(k-1, l)) / (2.0 * dpsi);
            double yp = (Y(k, l+1) - Y(k, l-1)) / (2.0 * dphi);
            double ys = (Y(k+1, l) - Y(k-1, l)) / (2.0 * dpsi);

            double mp = std::sqrt(xp*xp + yp*yp);
            double ms = std::sqrt(xs*xs + ys*ys);
            if (mp < 1e-14 || ms < 1e-14) continue;

            double ca = std::clamp((xp*xs + yp*ys) / (mp * ms), -1.0, 1.0);
            skew(k, l) = std::abs(90.0 - std::acos(ca) * 180.0 / M_PI);
        }
    }

    // Bottom wall (k=0): one-sided psi derivative
    for (int l = 1; l < N - 1; ++l) {
        double xp = (X(0, l+1) - X(0, l-1)) / (2.0 * dphi);
        double xs = (X(1, l) - X(0, l)) / dpsi;
        double yp = (Y(0, l+1) - Y(0, l-1)) / (2.0 * dphi);
        double ys = (Y(1, l) - Y(0, l)) / dpsi;

        double mp = std::sqrt(xp*xp + yp*yp);
        double ms = std::sqrt(xs*xs + ys*ys);
        if (mp > 1e-14 && ms > 1e-14) {
            double ca = std::clamp((xp*xs + yp*ys) / (mp * ms), -1.0, 1.0);
            skew(0, l) = std::abs(90.0 - std::acos(ca) * 180.0 / M_PI);
        }
    }

    // Top wall (k=N-1): one-sided psi derivative
    for (int l = 1; l < N - 1; ++l) {
        int kk = N - 1;
        double xp = (X(kk, l+1) - X(kk, l-1)) / (2.0 * dphi);
        double xs = (X(kk, l) - X(kk-1, l)) / dpsi;
        double yp = (Y(kk, l+1) - Y(kk, l-1)) / (2.0 * dphi);
        double ys = (Y(kk, l) - Y(kk-1, l)) / dpsi;

        double mp = std::sqrt(xp*xp + yp*yp);
        double ms = std::sqrt(xs*xs + ys*ys);
        if (mp > 1e-14 && ms > 1e-14) {
            double ca = std::clamp((xp*xs + yp*ys) / (mp * ms), -1.0, 1.0);
            skew(kk, l) = std::abs(90.0 - std::acos(ca) * 180.0 / M_PI);
        }
    }

    return skew;
}

// =====================================================================
//  Jacobian field
// =====================================================================
static Eigen::MatrixXd compute_jacobian_field(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    Eigen::MatrixXd J_field = Eigen::MatrixXd::Zero(N, N);

    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double xp = (X(k, l+1) - X(k, l-1)) / (2.0 * dphi);
            double xs = (X(k+1, l) - X(k-1, l)) / (2.0 * dpsi);
            double yp = (Y(k, l+1) - Y(k, l-1)) / (2.0 * dphi);
            double ys = (Y(k+1, l) - Y(k-1, l)) / (2.0 * dpsi);
            J_field(k, l) = xp * ys - xs * yp;
        }
    }

    return J_field;
}

// =====================================================================
//  Elliptic Solver on TFI grid (Thomas-Middlecoff Poisson system)
// =====================================================================
EllipticResult tfi_elliptic_solve(
    const Eigen::MatrixXd& X_tfi,
    const Eigen::MatrixXd& Y_tfi,
    const Eigen::VectorXd& phi_comp_vec,
    const Eigen::VectorXd& psi_comp_vec,
    int n_grid_pts,
    int n_elliptic_iter,
    double sor_omega,
    double convergence_tol)
{
    std::printf("Starting elliptic solver (Poisson system) for %d iterations...\n", n_elliptic_iter);

    double dphi  = phi_comp_vec(1) - phi_comp_vec(0);
    double dpsi  = psi_comp_vec(1) - psi_comp_vec(0);
    double dphi2 = dphi * dphi;
    double dpsi2 = dpsi * dpsi;
    double twodphidpsi = 2.0 * dphi * dpsi;

    // Working copies (modified in-place by Gauss-Seidel)
    Eigen::MatrixXd X = X_tfi;
    Eigen::MatrixXd Y = Y_tfi;

    const int N   = n_grid_pts;
    const int end = N - 1;  // convenience alias matching MATLAB's "end"

    // =========================================================================
    // THOMAS-MIDDLECOFF CONTROL FUNCTION INITIALIZATION (P and Q)
    // Computed ONCE from the initial TFI grid and frozen for all iterations
    // =========================================================================
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);

    // --- P: extract from bottom (row 0) and top (row end) walls ---
    for (int l = 1; l < end; ++l) {
        // Bottom wall
        double xp  = (X(0, l+1) - X(0, l-1)) / (2.0 * dphi);
        double yp  = (Y(0, l+1) - Y(0, l-1)) / (2.0 * dphi);
        double xpp = (X(0, l+1) - 2.0*X(0, l) + X(0, l-1)) / dphi2;
        double ypp = (Y(0, l+1) - 2.0*Y(0, l) + Y(0, l-1)) / dphi2;
        P(0, l) = -(xp*xpp + yp*ypp) / (xp*xp + yp*yp + 1e-14);

        // Top wall
        xp  = (X(end, l+1) - X(end, l-1)) / (2.0 * dphi);
        yp  = (Y(end, l+1) - Y(end, l-1)) / (2.0 * dphi);
        xpp = (X(end, l+1) - 2.0*X(end, l) + X(end, l-1)) / dphi2;
        ypp = (Y(end, l+1) - 2.0*Y(end, l) + Y(end, l-1)) / dphi2;
        P(end, l) = -(xp*xpp + yp*ypp) / (xp*xp + yp*yp + 1e-14);
    }
    P(0, 0)     = P(0, 1);       P(0, end)     = P(0, end-1);
    P(end, 0)   = P(end, 1);     P(end, end)   = P(end, end-1);

    // --- Q: extract from left (col 0) and right (col end) walls ---
    for (int k = 1; k < end; ++k) {
        // Left wall
        double xq  = (X(k+1, 0) - X(k-1, 0)) / (2.0 * dpsi);
        double yq  = (Y(k+1, 0) - Y(k-1, 0)) / (2.0 * dpsi);
        double xqq = (X(k+1, 0) - 2.0*X(k, 0) + X(k-1, 0)) / dpsi2;
        double yqq = (Y(k+1, 0) - 2.0*Y(k, 0) + Y(k-1, 0)) / dpsi2;
        Q(k, 0) = -(xq*xqq + yq*yqq) / (xq*xq + yq*yq + 1e-14);

        // Right wall
        xq  = (X(k+1, end) - X(k-1, end)) / (2.0 * dpsi);
        yq  = (Y(k+1, end) - Y(k-1, end)) / (2.0 * dpsi);
        xqq = (X(k+1, end) - 2.0*X(k, end) + X(k-1, end)) / dpsi2;
        yqq = (Y(k+1, end) - 2.0*Y(k, end) + Y(k-1, end)) / dpsi2;
        Q(k, end) = -(xq*xqq + yq*yqq) / (xq*xq + yq*yq + 1e-14);
    }
    Q(0, 0)     = Q(1, 0);       Q(end, 0)     = Q(end-1, 0);
    Q(0, end)   = Q(1, end);     Q(end, end)   = Q(end-1, end);

    // --- Interpolate P and Q into the interior volume ---
    for (int k = 1; k < end; ++k) {
        for (int l = 1; l < end; ++l) {
            double s_psi = static_cast<double>(k) / end;
            double s_phi = static_cast<double>(l) / end;
            P(k, l) = (1.0 - s_psi) * P(0, l)  + s_psi * P(end, l);
            Q(k, l) = (1.0 - s_phi) * Q(k, 0)  + s_phi * Q(k, end);
        }
    }

    // =========================================================================
    //  Gauss-Seidel / SOR iteration
    // =========================================================================
    double final_max_update = 0.0;
    int last_iter = 0;

    for (int iter = 1; iter <= n_elliptic_iter; ++iter) {
        double max_update_iter = 0.0;

        for (int k = 1; k < end; ++k) {
            for (int l = 1; l < end; ++l) {

                double x_old = X(k, l);
                double y_old = Y(k, l);

                // Central differences (using latest Gauss-Seidel values)
                double x_phi = (X(k, l+1) - X(k, l-1)) / (2.0 * dphi);
                double x_psi = (X(k+1, l) - X(k-1, l)) / (2.0 * dpsi);
                double y_phi = (Y(k, l+1) - Y(k, l-1)) / (2.0 * dphi);
                double y_psi = (Y(k+1, l) - Y(k-1, l)) / (2.0 * dpsi);

                // Metric coefficients
                double alpha_m = x_psi*x_psi + y_psi*y_psi;
                double beta_m  = x_phi*x_psi + y_phi*y_psi;
                double gamma_m = x_phi*x_phi + y_phi*y_phi;

                double denom = 2.0 * (alpha_m / dphi2 + gamma_m / dpsi2);
                if (std::abs(denom) < 1e-14) continue;

                // Thomas-Middlecoff source terms (alpha_m / gamma_m weighting)
                double source_x = alpha_m * P(k, l) * x_phi + gamma_m * Q(k, l) * x_psi;
                double source_y = alpha_m * P(k, l) * y_phi + gamma_m * Q(k, l) * y_psi;

                // Cross-derivative stencils
                double mixed_x = X(k+1, l+1) - X(k+1, l-1) - X(k-1, l+1) + X(k-1, l-1);
                double mixed_y = Y(k+1, l+1) - Y(k+1, l-1) - Y(k-1, l+1) + Y(k-1, l-1);

                // Right-hand sides
                double rhs_x = alpha_m * (X(k, l+1) + X(k, l-1)) / dphi2
                              + gamma_m * (X(k+1, l) + X(k-1, l)) / dpsi2
                              - beta_m  * mixed_x / twodphidpsi
                              + source_x;

                double rhs_y = alpha_m * (Y(k, l+1) + Y(k, l-1)) / dphi2
                              + gamma_m * (Y(k+1, l) + Y(k-1, l)) / dpsi2
                              - beta_m  * mixed_y / twodphidpsi
                              + source_y;

                double x_new_val = rhs_x / denom;
                double y_new_val = rhs_y / denom;

                // SOR update in-place
                X(k, l) = (1.0 - sor_omega) * x_old + sor_omega * x_new_val;
                Y(k, l) = (1.0 - sor_omega) * y_old + sor_omega * y_new_val;

                max_update_iter = std::max(max_update_iter, std::abs(X(k, l) - x_old));
                max_update_iter = std::max(max_update_iter, std::abs(Y(k, l) - y_old));
            }
        }

        final_max_update = max_update_iter;
        last_iter = iter;

        if (iter % 20 == 0 || iter == 1 || iter == n_elliptic_iter) {
            std::printf("Elliptic Iter: %4d / %4d, Max Update: %e\n",
                        iter, n_elliptic_iter, max_update_iter);
        }

        if (max_update_iter < convergence_tol) {
            std::printf("Elliptic solver converged in %d iterations.\n", iter);
            break;
        }
    }

    if (last_iter == n_elliptic_iter && final_max_update >= convergence_tol) {
        std::printf("Elliptic solver reached max iterations (%d) without full convergence (Max Update: %e).\n",
                    n_elliptic_iter, final_max_update);
    }

    // =====================================================================
    //  Diagnostics
    // =====================================================================
    EllipticResult result;
    result.X_ellip_grid = X;
    result.Y_ellip_grid = Y;
    result.diagnost.skewness = compute_skewness(X, Y, dphi, dpsi, N);
    result.diagnost.jacobian = compute_jacobian_field(X, Y, dphi, dpsi, N);

    const auto& skew    = result.diagnost.skewness;
    const auto& J_field = result.diagnost.jacobian;

    // Bot wall max skewness (row 0, interior cols)
    double bot_max = 0.0;
    for (int l = 1; l < end; ++l) bot_max = std::max(bot_max, std::abs(skew(0, l)));

    // Top wall max skewness (row end, interior cols)
    double top_max = 0.0;
    for (int l = 1; l < end; ++l) top_max = std::max(top_max, std::abs(skew(end, l)));

    // Interior max skewness
    double int_max = 0.0;
    for (int k = 1; k < end; ++k)
        for (int l = 1; l < end; ++l)
            int_max = std::max(int_max, std::abs(skew(k, l)));

    std::printf("  Skewness: bot wall max=%.3f, top wall max=%.3f, interior max=%.3f deg\n",
                bot_max, top_max, int_max);

    // Jacobian min/max over the interior block.
    // Boundary entries are not computed by compute_jacobian_field and remain
    // zero, so we look only at the interior to get a meaningful range.
    double j_min = J_field.block(1, 1, N - 2, N - 2).minCoeff();
    double j_max = J_field.block(1, 1, N - 2, N - 2).maxCoeff();
    std::printf("  Jacobian (interior): min=%.4e, max=%.4e\n", j_min, j_max);

    return result;
}