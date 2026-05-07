#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

#include "hdg/tfi_elliptic_steger_sorenson_v2.h"

namespace {

constexpr double kRad2Deg = 180.0 / M_PI;
constexpr double kEps     = 1e-14;

// =====================================================================
//  Inward-pointing wall normals
//  k_wall: 0 for bottom, N-1 for top
//  interior_dir: +1 (interior at k_wall+1) or -1 (interior at k_wall-1)
// =====================================================================
void compute_wall_normals(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    int k_wall, double dphi, int N, int interior_dir,
    Eigen::VectorXd& nx, Eigen::VectorXd& ny)
{
    nx.setZero(N);
    ny.setZero(N);
    int k_int = k_wall + interior_dir;

    for (int l = 1; l < N - 1; ++l) {
        double tx = (X(k_wall, l + 1) - X(k_wall, l - 1)) / (2.0 * dphi);
        double ty = (Y(k_wall, l + 1) - Y(k_wall, l - 1)) / (2.0 * dphi);
        double tm = std::sqrt(tx * tx + ty * ty);
        if (tm < kEps) continue;
        tx /= tm; ty /= tm;
        nx(l) = -ty; ny(l) = tx;

        // Make sure it points toward the interior
        double dx = X(k_int, l) - X(k_wall, l);
        double dy = Y(k_int, l) - Y(k_wall, l);
        if ((nx(l) * dx + ny(l) * dy) < 0.0) {
            nx(l) = -nx(l); ny(l) = -ny(l);
        }
    }
    nx(0)     = nx(1);     ny(0)     = ny(1);
    nx(N - 1) = nx(N - 2); ny(N - 1) = ny(N - 2);
}

// =====================================================================
//  Quick wall skewness (max deviation from 90 deg, in degrees)
// =====================================================================
void quick_wall_skewness(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N,
    double& skew_bot, double& skew_top)
{
    skew_bot = 0.0; skew_top = 0.0;
    for (int l = 1; l < N - 1; ++l) {
        // Bottom (k=0)
        {
            double xp = (X(0, l + 1) - X(0, l - 1)) / (2.0 * dphi);
            double yp = (Y(0, l + 1) - Y(0, l - 1)) / (2.0 * dphi);
            double xs = (X(1, l) - X(0, l)) / dpsi;
            double ys = (Y(1, l) - Y(0, l)) / dpsi;
            double mp = std::sqrt(xp * xp + yp * yp);
            double ms = std::sqrt(xs * xs + ys * ys);
            if (mp > kEps && ms > kEps) {
                double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                skew_bot = std::max(skew_bot, std::abs(90.0 - std::acos(ca) * kRad2Deg));
            }
        }
        // Top (k=N-1)
        {
            int kk = N - 1;
            double xp = (X(kk, l + 1) - X(kk, l - 1)) / (2.0 * dphi);
            double yp = (Y(kk, l + 1) - Y(kk, l - 1)) / (2.0 * dphi);
            double xs = (X(kk, l) - X(kk - 1, l)) / dpsi;
            double ys = (Y(kk, l) - Y(kk - 1, l)) / dpsi;
            double mp = std::sqrt(xp * xp + yp * yp);
            double ms = std::sqrt(xs * xs + ys * ys);
            if (mp > kEps && ms > kEps) {
                double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                skew_top = std::max(skew_top, std::abs(90.0 - std::acos(ca) * kRad2Deg));
            }
        }
    }
}

// =====================================================================
//  Full skewness field (deviation from 90 deg, in degrees)
// =====================================================================
Eigen::MatrixXd compute_skewness(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    Eigen::MatrixXd skew = Eigen::MatrixXd::Zero(N, N);

    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double x_phi = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
            double x_psi = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
            double y_phi = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
            double y_psi = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);
            double mp = std::sqrt(x_phi * x_phi + y_phi * y_phi);
            double ms = std::sqrt(x_psi * x_psi + y_psi * y_psi);
            if (mp < kEps || ms < kEps) continue;
            double ca = std::clamp(
                (x_phi * x_psi + y_phi * y_psi) / (mp * ms), -1.0, 1.0);
            skew(k, l) = std::abs(90.0 - std::acos(ca) * kRad2Deg);
        }
    }

    for (int l = 1; l < N - 1; ++l) {
        // Bottom
        {
            double xp = (X(0, l + 1) - X(0, l - 1)) / (2.0 * dphi);
            double yp = (Y(0, l + 1) - Y(0, l - 1)) / (2.0 * dphi);
            double xs = (X(1, l) - X(0, l)) / dpsi;
            double ys = (Y(1, l) - Y(0, l)) / dpsi;
            double mp = std::sqrt(xp * xp + yp * yp);
            double ms = std::sqrt(xs * xs + ys * ys);
            if (mp > kEps && ms > kEps) {
                double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                skew(0, l) = std::abs(90.0 - std::acos(ca) * kRad2Deg);
            }
        }
        // Top
        {
            int kk = N - 1;
            double xp = (X(kk, l + 1) - X(kk, l - 1)) / (2.0 * dphi);
            double yp = (Y(kk, l + 1) - Y(kk, l - 1)) / (2.0 * dphi);
            double xs = (X(kk, l) - X(kk - 1, l)) / dpsi;
            double ys = (Y(kk, l) - Y(kk - 1, l)) / dpsi;
            double mp = std::sqrt(xp * xp + yp * yp);
            double ms = std::sqrt(xs * xs + ys * ys);
            if (mp > kEps && ms > kEps) {
                double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                skew(kk, l) = std::abs(90.0 - std::acos(ca) * kRad2Deg);
            }
        }
    }

    return skew;
}

// =====================================================================
//  Jacobian field (interior only; boundary entries left at 0)
// =====================================================================
Eigen::MatrixXd compute_jacobian_field(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(N, N);
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double x_phi = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
            double x_psi = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
            double y_phi = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
            double y_psi = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);
            J(k, l) = x_phi * y_psi - x_psi * y_phi;
        }
    }
    return J;
}

} // anonymous namespace


// =====================================================================
//  Main solver: coupled Winslow/TTM + Steger-Sorenson orthogonality
// =====================================================================
EllipticResult tfi_elliptic_steger_sorenson_v2(
    const Eigen::MatrixXd& X_tfi,
    const Eigen::MatrixXd& Y_tfi,
    const Eigen::VectorXd& phi_comp_vec,
    const Eigen::VectorXd& psi_comp_vec,
    int n_grid_pts,
    int n_elliptic_iter,
    double sor_omega,
    double convergence_tol,
    const OrthoOpts& ortho_opts)
{
    std::printf("Steger-Sorenson v2: coupled TTM + orthogonality solver\n");

    // Working copies (mutated in place)
    Eigen::MatrixXd X = X_tfi;
    Eigen::MatrixXd Y = Y_tfi;

    const bool   ortho_enable = ortho_opts.enable;
    const bool   ortho_walls  = ortho_opts.walls;
    const bool   ortho_outer  = ortho_opts.outer;
    // The MATLAB v2 calls this `relax`. The shared OrthoOpts struct uses
    // `wall_relax` for the same Phase-1 wall-orthogonality relaxation, so
    // we reuse that field here.
    const double ortho_relax  = ortho_opts.wall_relax;

    const int N = n_grid_pts;

    const double dphi  = phi_comp_vec(1) - phi_comp_vec(0);
    const double dpsi  = psi_comp_vec(1) - psi_comp_vec(0);
    const double dphi2 = dphi * dphi;
    const double dpsi2 = dpsi * dpsi;
    const double twodphidpsi = 2.0 * dphi * dpsi;

    // SOR interior range.
    // MATLAB: k_start=2, k_end=N-1; with ortho: k_start=3, k_end=N-2.
    // C++ 0-indexed: k_start=1, k_end=N-2; with ortho: k_start=2, k_end=N-3.
    int k_start = 1;
    int k_end   = N - 2;
    if (ortho_enable && ortho_walls) k_start = 2;
    if (ortho_enable && ortho_outer) k_end   = N - 3;

    std::printf("  N=%d, SOR range: k=%d to k=%d, omega=%.2f\n",
                N, k_start, k_end, sor_omega);
    std::printf("  Ortho: walls=%d, outer=%d, relax=%.2f\n",
                static_cast<int>(ortho_walls),
                static_cast<int>(ortho_outer),
                ortho_relax);

    // =====================================================================
    //  1. Thomas-Middlecoff source terms
    // =====================================================================
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);

    // P along bottom (k=0) and top (k=N-1)
    for (int l = 1; l < N - 1; ++l) {
        // Bottom
        {
            double xp  = (X(0, l + 1) - X(0, l - 1)) / (2.0 * dphi);
            double yp  = (Y(0, l + 1) - Y(0, l - 1)) / (2.0 * dphi);
            double xpp = (X(0, l + 1) - 2.0 * X(0, l) + X(0, l - 1)) / dphi2;
            double ypp = (Y(0, l + 1) - 2.0 * Y(0, l) + Y(0, l - 1)) / dphi2;
            P(0, l) = -(xp * xpp + yp * ypp) / (xp * xp + yp * yp + kEps);
        }
        // Top
        {
            int kk = N - 1;
            double xp  = (X(kk, l + 1) - X(kk, l - 1)) / (2.0 * dphi);
            double yp  = (Y(kk, l + 1) - Y(kk, l - 1)) / (2.0 * dphi);
            double xpp = (X(kk, l + 1) - 2.0 * X(kk, l) + X(kk, l - 1)) / dphi2;
            double ypp = (Y(kk, l + 1) - 2.0 * Y(kk, l) + Y(kk, l - 1)) / dphi2;
            P(kk, l) = -(xp * xpp + yp * ypp) / (xp * xp + yp * yp + kEps);
        }
    }
    P(0, 0)         = P(0, 1);
    P(0, N - 1)     = P(0, N - 2);
    P(N - 1, 0)     = P(N - 1, 1);
    P(N - 1, N - 1) = P(N - 1, N - 2);

    // Q along left (l=0) and right (l=N-1)
    for (int k = 1; k < N - 1; ++k) {
        // Left
        {
            double xq  = (X(k + 1, 0) - X(k - 1, 0)) / (2.0 * dpsi);
            double yq  = (Y(k + 1, 0) - Y(k - 1, 0)) / (2.0 * dpsi);
            double xqq = (X(k + 1, 0) - 2.0 * X(k, 0) + X(k - 1, 0)) / dpsi2;
            double yqq = (Y(k + 1, 0) - 2.0 * Y(k, 0) + Y(k - 1, 0)) / dpsi2;
            Q(k, 0) = -(xq * xqq + yq * yqq) / (xq * xq + yq * yq + kEps);
        }
        // Right
        {
            int ll = N - 1;
            double xq  = (X(k + 1, ll) - X(k - 1, ll)) / (2.0 * dpsi);
            double yq  = (Y(k + 1, ll) - Y(k - 1, ll)) / (2.0 * dpsi);
            double xqq = (X(k + 1, ll) - 2.0 * X(k, ll) + X(k - 1, ll)) / dpsi2;
            double yqq = (Y(k + 1, ll) - 2.0 * Y(k, ll) + Y(k - 1, ll)) / dpsi2;
            Q(k, ll) = -(xq * xqq + yq * yqq) / (xq * xq + yq * yq + kEps);
        }
    }
    Q(0, 0)         = Q(1, 0);
    Q(N - 1, 0)     = Q(N - 2, 0);
    Q(0, N - 1)     = Q(1, N - 1);
    Q(N - 1, N - 1) = Q(N - 2, N - 1);

    // Linear interpolation into interior.
    // MATLAB: s_psi = (k-1)/(N-1) (1-indexed) -> s_psi = k/(N-1) (0-indexed)
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double s_psi = static_cast<double>(k) / static_cast<double>(N - 1);
            double s_phi = static_cast<double>(l) / static_cast<double>(N - 1);
            P(k, l) = (1.0 - s_psi) * P(0, l) + s_psi * P(N - 1, l);
            Q(k, l) = (1.0 - s_phi) * Q(k, 0) + s_phi * Q(k, N - 1);
        }
    }

    // =====================================================================
    //  2. Precompute wall normals
    // =====================================================================
    Eigen::VectorXd nx_bot, ny_bot, nx_top, ny_top;
    compute_wall_normals(X, Y, 0,     dphi, N, +1, nx_bot, ny_bot);
    compute_wall_normals(X, Y, N - 1, dphi, N, -1, nx_top, ny_top);

    // =====================================================================
    //  3. Coupled iteration loop
    // =====================================================================
    std::vector<double> conv_history;
    conv_history.reserve(n_elliptic_iter);
    int final_iter = n_elliptic_iter;
    double max_update_iter = 0.0;
    bool converged = false;

    for (int iter = 1; iter <= n_elliptic_iter; ++iter) {
        max_update_iter = 0.0;

        // -------- 3a. Ortho BC: set k=1 (MATLAB k=2) from bottom wall -------
        if (ortho_enable && ortho_walls) {
            for (int l = 1; l < N - 1; ++l) {
                double dx = X(1, l) - X(0, l);
                double dy = Y(1, l) - Y(0, l);
                double spacing = std::sqrt(dx * dx + dy * dy);
                if (spacing < kEps) continue;

                double x_ortho = X(0, l) + spacing * nx_bot(l);
                double y_ortho = Y(0, l) + spacing * ny_bot(l);

                double x_new = (1.0 - ortho_relax) * X(1, l) + ortho_relax * x_ortho;
                double y_new = (1.0 - ortho_relax) * Y(1, l) + ortho_relax * y_ortho;

                max_update_iter = std::max(max_update_iter, std::abs(x_new - X(1, l)));
                max_update_iter = std::max(max_update_iter, std::abs(y_new - Y(1, l)));

                X(1, l) = x_new;
                Y(1, l) = y_new;
            }
        }

        // -------- 3b. Ortho BC: set k=N-2 (MATLAB k=N-1) from top wall ------
        if (ortho_enable && ortho_outer) {
            for (int l = 1; l < N - 1; ++l) {
                double dx = X(N - 2, l) - X(N - 1, l);
                double dy = Y(N - 2, l) - Y(N - 1, l);
                double spacing = std::sqrt(dx * dx + dy * dy);
                if (spacing < kEps) continue;

                double x_ortho = X(N - 1, l) + spacing * nx_top(l);
                double y_ortho = Y(N - 1, l) + spacing * ny_top(l);

                double x_new = (1.0 - ortho_relax) * X(N - 2, l) + ortho_relax * x_ortho;
                double y_new = (1.0 - ortho_relax) * Y(N - 2, l) + ortho_relax * y_ortho;

                max_update_iter = std::max(max_update_iter, std::abs(x_new - X(N - 2, l)));
                max_update_iter = std::max(max_update_iter, std::abs(y_new - Y(N - 2, l)));

                X(N - 2, l) = x_new;
                Y(N - 2, l) = y_new;
            }
        }

        // -------- 3c. SOR sweep on true interior --------
        for (int k = k_start; k <= k_end; ++k) {
            for (int l = 1; l <= N - 2; ++l) {
                double x_old = X(k, l), y_old = Y(k, l);

                double x_phi = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
                double x_psi = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
                double y_phi = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
                double y_psi = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);

                double alpha_m = x_psi * x_psi + y_psi * y_psi;
                double beta_m  = x_phi * x_psi + y_phi * y_psi;
                double gamma_m = x_phi * x_phi + y_phi * y_phi;

                double denom = 2.0 * (alpha_m / dphi2 + gamma_m / dpsi2);
                if (std::abs(denom) < kEps) continue;

                double source_x = alpha_m * P(k, l) * x_phi + gamma_m * Q(k, l) * x_psi;
                double source_y = alpha_m * P(k, l) * y_phi + gamma_m * Q(k, l) * y_psi;

                double mixed_x = (X(k + 1, l + 1) - X(k + 1, l - 1)
                                - X(k - 1, l + 1) + X(k - 1, l - 1));
                double mixed_y = (Y(k + 1, l + 1) - Y(k + 1, l - 1)
                                - Y(k - 1, l + 1) + Y(k - 1, l - 1));

                double rhs_x = alpha_m * (X(k, l + 1) + X(k, l - 1)) / dphi2
                             + gamma_m * (X(k + 1, l) + X(k - 1, l)) / dpsi2
                             - beta_m * mixed_x / twodphidpsi
                             + source_x;

                double rhs_y = alpha_m * (Y(k, l + 1) + Y(k, l - 1)) / dphi2
                             + gamma_m * (Y(k + 1, l) + Y(k - 1, l)) / dpsi2
                             - beta_m * mixed_y / twodphidpsi
                             + source_y;

                double x_new_val = rhs_x / denom;
                double y_new_val = rhs_y / denom;

                X(k, l) = (1.0 - sor_omega) * x_old + sor_omega * x_new_val;
                Y(k, l) = (1.0 - sor_omega) * y_old + sor_omega * y_new_val;

                max_update_iter = std::max(max_update_iter, std::abs(X(k, l) - x_old));
                max_update_iter = std::max(max_update_iter, std::abs(Y(k, l) - y_old));
            }
        }

        // --- Convergence ---
        conv_history.push_back(max_update_iter);

        if (iter % 500 == 0 || iter == 1 || iter == n_elliptic_iter) {
            double sb, st;
            quick_wall_skewness(X, Y, dphi, dpsi, N, sb, st);
            std::printf("  iter %6d: max_update=%.4e  wall_skew=[%.2f, %.2f] deg\n",
                        iter, max_update_iter, sb, st);
        }

        if (max_update_iter < convergence_tol) {
            final_iter = iter;
            converged = true;
            double sb, st;
            quick_wall_skewness(X, Y, dphi, dpsi, N, sb, st);
            std::printf("  Converged at iter %d (max_update=%.4e, wall_skew=[%.2f, %.2f])\n",
                        iter, max_update_iter, sb, st);
            break;
        }
    }

    if (!converged) {
        std::printf("  WARNING: max iterations reached. Final max_update = %.4e\n",
                    max_update_iter);
        final_iter = n_elliptic_iter;
    }

    // =====================================================================
    //  4. Diagnostics
    // =====================================================================
    EllipticResult result;
    result.X_ellip_grid = X;
    result.Y_ellip_grid = Y;
    result.diagnost.skewness = compute_skewness(X, Y, dphi, dpsi, N);
    result.diagnost.jacobian = compute_jacobian_field(X, Y, dphi, dpsi, N);

    const auto& skew = result.diagnost.skewness;
    const auto& J    = result.diagnost.jacobian;

    double skew_bot_max = 0.0, skew_top_max = 0.0, skew_int_max = 0.0;
    for (int l = 1; l < N - 1; ++l) {
        skew_bot_max = std::max(skew_bot_max, std::abs(skew(0,     l)));
        skew_top_max = std::max(skew_top_max, std::abs(skew(N - 1, l)));
    }
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            skew_int_max = std::max(skew_int_max, std::abs(skew(k, l)));
        }
    }

    // Jacobian min/max over interior block (boundary entries are 0).
    double jmin = J.block(1, 1, N - 2, N - 2).minCoeff();
    double jmax = J.block(1, 1, N - 2, N - 2).maxCoeff();

    std::printf("  --- Final ---\n");
    std::printf("  Skewness: bot wall max=%.3f, top wall max=%.3f, interior max=%.3f deg\n",
                skew_bot_max, skew_top_max, skew_int_max);
    std::printf("  Jacobian (interior): min=%.4e, max=%.4e\n", jmin, jmax);

    // Suppress unused-variable warning if final_iter isn't recorded in
    // EllipticResult (the conv_history field may not exist on your struct).
    (void)final_iter;
    (void)conv_history;

    return result;
}