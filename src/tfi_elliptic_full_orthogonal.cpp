#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

#include "hdg/tfi_elliptic_full_orthogonal.h"

namespace {

constexpr double kRad2Deg = 180.0 / M_PI;
constexpr double kEps     = 1e-14;

// =====================================================================
//  Interpolate point on wall curve at arc-length s
//  (piecewise-linear interpolation matching the MATLAB helper)
// =====================================================================
void interp_wall_curve(
    double s,
    const Eigen::VectorXd& s_wall,
    const Eigen::VectorXd& X_wall,
    const Eigen::VectorXd& Y_wall,
    double& xi, double& yi)
{
    const int N = static_cast<int>(s_wall.size());

    // Clamp to endpoints
    if (s <= s_wall(0)) {
        xi = X_wall(0); yi = Y_wall(0);
        return;
    }
    if (s >= s_wall(N - 1)) {
        xi = X_wall(N - 1); yi = Y_wall(N - 1);
        return;
    }

    // Find segment: s_wall(idx) <= s < s_wall(idx+1)
    int idx = 0;
    for (int i = 0; i < N - 1; ++i) {
        if (s >= s_wall(i) && s < s_wall(i + 1)) {
            idx = i;
            break;
        }
    }

    double ds = s_wall(idx + 1) - s_wall(idx);
    if (ds < kEps) {
        xi = X_wall(idx); yi = Y_wall(idx);
        return;
    }

    double t = (s - s_wall(idx)) / ds;
    xi = (1.0 - t) * X_wall(idx) + t * X_wall(idx + 1);
    yi = (1.0 - t) * Y_wall(idx) + t * Y_wall(idx + 1);
}

// =====================================================================
//  Wall normals (inward-pointing)
//  k_wall: 0 for bottom, N-1 for top
//  int_dir: +1 (interior is k_wall+1) or -1 (interior is k_wall-1)
// =====================================================================
void compute_wall_normals(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    int k_wall, double dphi, int N, int int_dir,
    Eigen::VectorXd& nx, Eigen::VectorXd& ny)
{
    nx.setZero(N);
    ny.setZero(N);
    int k_int = k_wall + int_dir;

    for (int l = 1; l < N - 1; ++l) {
        double tx = (X(k_wall, l + 1) - X(k_wall, l - 1)) / (2.0 * dphi);
        double ty = (Y(k_wall, l + 1) - Y(k_wall, l - 1)) / (2.0 * dphi);
        double tm = std::sqrt(tx * tx + ty * ty);
        if (tm < kEps) continue;
        tx /= tm; ty /= tm;
        nx(l) = -ty; ny(l) = tx;
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
//  Returns pair {bottom, top}
// =====================================================================
void quick_wall_skew(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N,
    double& sb, double& st)
{
    sb = 0.0; st = 0.0;
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
                sb = std::max(sb, std::abs(90.0 - std::acos(ca) * kRad2Deg));
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
                st = std::max(st, std::abs(90.0 - std::acos(ca) * kRad2Deg));
            }
        }
    }
}

// =====================================================================
//  Quick interior max skewness
// =====================================================================
double quick_interior_skew(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    double s_max = 0.0;
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double xp = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
            double xs = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
            double yp = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
            double ys = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);
            double mp = std::sqrt(xp * xp + yp * yp);
            double ms = std::sqrt(xs * xs + ys * ys);
            if (mp > kEps && ms > kEps) {
                double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                s_max = std::max(s_max, std::abs(90.0 - std::acos(ca) * kRad2Deg));
            }
        }
    }
    return s_max;
}

// =====================================================================
//  Skewness: deviation from orthogonality in degrees (0 = orthogonal)
// =====================================================================
Eigen::MatrixXd compute_skewness(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    Eigen::MatrixXd skew = Eigen::MatrixXd::Zero(N, N);

    // Interior: central differences in both directions
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double xp = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
            double xs = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
            double yp = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
            double ys = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);

            double mp = std::sqrt(xp * xp + yp * yp);
            double ms = std::sqrt(xs * xs + ys * ys);
            if (mp < kEps || ms < kEps) continue;

            double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
            skew(k, l) = std::abs(90.0 - std::acos(ca) * kRad2Deg);
        }
    }

    // Bottom (k=0) and top (k=N-1) walls: one-sided psi derivative
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
//  Jacobian field
// =====================================================================
Eigen::MatrixXd compute_jacobian_field(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
    double dphi, double dpsi, int N)
{
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(N, N);
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double xp = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
            double xs = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
            double yp = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
            double ys = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);
            J(k, l) = xp * ys - xs * yp;
        }
    }
    return J;
}

} // anonymous namespace

// =====================================================================
//  Main solver
// =====================================================================
EllipticResult tfi_elliptic_full_orthogonal(
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
    std::printf("=== Full Orthogonal Solver (Steger-Sorenson + Marching + Wall Slide) ===\n");

    // Working copies (mutated in place)
    Eigen::MatrixXd X = X_tfi;
    Eigen::MatrixXd Y = Y_tfi;

    const bool   ortho_enable = ortho_opts.enable;
    const bool   ortho_walls  = ortho_opts.walls;
    const bool   ortho_outer  = ortho_opts.outer;
    const double wall_relax   = ortho_opts.wall_relax;
    const double march_relax  = ortho_opts.march_relax;
    const int    n_march      = ortho_opts.n_march_sweeps;
    const double march_tol    = ortho_opts.march_tol;
    const bool   wall_slide   = ortho_opts.wall_slide;
    const double slide_relax  = ortho_opts.slide_relax;

    const int N = n_grid_pts;

    const double dphi  = phi_comp_vec(1) - phi_comp_vec(0);
    const double dpsi  = psi_comp_vec(1) - psi_comp_vec(0);
    const double dphi2 = dphi * dphi;
    const double dpsi2 = dpsi * dpsi;
    const double twodphidpsi = 2.0 * dphi * dpsi;

    // MATLAB: k_start=2, k_end=N-1; with ortho: k_start=3, k_end=N-2
    // C++ 0-indexed: k_start=1, k_end=N-2; with ortho: k_start=2, k_end=N-3
    int k_start = 1;
    int k_end   = N - 2;
    if (ortho_enable && ortho_walls) k_start = 2;
    if (ortho_enable && ortho_outer) k_end   = N - 3;

    // =====================================================================
    //  Parameterize bottom and top wall curves by arc length (for sliding)
    // =====================================================================
    Eigen::VectorXd X_wall_bot_orig = X.row(0).transpose();
    Eigen::VectorXd Y_wall_bot_orig = Y.row(0).transpose();
    Eigen::VectorXd s_wall_bot = Eigen::VectorXd::Zero(N);
    for (int l = 1; l < N; ++l) {
        double dx = X_wall_bot_orig(l) - X_wall_bot_orig(l - 1);
        double dy = Y_wall_bot_orig(l) - Y_wall_bot_orig(l - 1);
        s_wall_bot(l) = s_wall_bot(l - 1) + std::sqrt(dx * dx + dy * dy);
    }
    double s_total_bot = s_wall_bot(N - 1);
    Eigen::VectorXd s_current_bot = s_wall_bot;

    Eigen::VectorXd X_wall_top_orig = X.row(N - 1).transpose();
    Eigen::VectorXd Y_wall_top_orig = Y.row(N - 1).transpose();
    Eigen::VectorXd s_wall_top = Eigen::VectorXd::Zero(N);
    for (int l = 1; l < N; ++l) {
        double dx = X_wall_top_orig(l) - X_wall_top_orig(l - 1);
        double dy = Y_wall_top_orig(l) - Y_wall_top_orig(l - 1);
        s_wall_top(l) = s_wall_top(l - 1) + std::sqrt(dx * dx + dy * dy);
    }
    double s_total_top = s_wall_top(N - 1);
    Eigen::VectorXd s_current_top = s_wall_top;

    // =====================================================================
    //  PHASE 1: Coupled Steger-Sorenson + TTM
    // =====================================================================
    std::printf("  Phase 1: Steger-Sorenson (SOR k=%d:%d)\n", k_start, k_end);

    // --- TTM source terms ---
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);

    // P along bottom and top walls (varying with l)
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
    P(0, 0) = P(0, 1);
    P(0, N - 1) = P(0, N - 2);
    P(N - 1, 0) = P(N - 1, 1);
    P(N - 1, N - 1) = P(N - 1, N - 2);

    // Q along left and right boundaries (varying with k)
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
    Q(0, 0) = Q(1, 0);
    Q(N - 1, 0) = Q(N - 2, 0);
    Q(0, N - 1) = Q(1, N - 1);
    Q(N - 1, N - 1) = Q(N - 2, N - 1);

    // Bilinear blend of P, Q into the interior
    // MATLAB: s_psi = (k-1)/(N-1), s_phi = (l-1)/(N-1) with k,l in 1-indexed
    // In 0-indexed C++: s_psi = k/(N-1), s_phi = l/(N-1)
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            double s_psi = static_cast<double>(k) / static_cast<double>(N - 1);
            double s_phi = static_cast<double>(l) / static_cast<double>(N - 1);
            P(k, l) = (1.0 - s_psi) * P(0, l) + s_psi * P(N - 1, l);
            Q(k, l) = (1.0 - s_phi) * Q(k, 0) + s_phi * Q(k, N - 1);
        }
    }

    // --- Wall normals ---
    Eigen::VectorXd nx_bot, ny_bot, nx_top, ny_top;
    compute_wall_normals(X, Y, 0,     dphi, N, +1, nx_bot, ny_bot);
    compute_wall_normals(X, Y, N - 1, dphi, N, -1, nx_top, ny_top);

    // --- Coupled iteration ---
    std::vector<double> conv_p1;
    conv_p1.reserve(n_elliptic_iter);
    int p1_final = n_elliptic_iter;

    for (int iter = 1; iter <= n_elliptic_iter; ++iter) {
        double max_update = 0.0;

        // Ortho BC: k=1 (MATLAB k=2)
        if (ortho_enable && ortho_walls) {
            for (int l = 1; l < N - 1; ++l) {
                double dx = X(1, l) - X(0, l);
                double dy = Y(1, l) - Y(0, l);
                double sp = std::sqrt(dx * dx + dy * dy);
                if (sp < kEps) continue;
                double xo = X(0, l) + sp * nx_bot(l);
                double yo = Y(0, l) + sp * ny_bot(l);
                double xn = (1.0 - wall_relax) * X(1, l) + wall_relax * xo;
                double yn = (1.0 - wall_relax) * Y(1, l) + wall_relax * yo;
                max_update = std::max(max_update,
                    std::max(std::abs(xn - X(1, l)), std::abs(yn - Y(1, l))));
                X(1, l) = xn; Y(1, l) = yn;
            }
        }

        // Ortho BC: k=N-2 (MATLAB k=N-1)
        if (ortho_enable && ortho_outer) {
            for (int l = 1; l < N - 1; ++l) {
                double dx = X(N - 2, l) - X(N - 1, l);
                double dy = Y(N - 2, l) - Y(N - 1, l);
                double sp = std::sqrt(dx * dx + dy * dy);
                if (sp < kEps) continue;
                double xo = X(N - 1, l) + sp * nx_top(l);
                double yo = Y(N - 1, l) + sp * ny_top(l);
                double xn = (1.0 - wall_relax) * X(N - 2, l) + wall_relax * xo;
                double yn = (1.0 - wall_relax) * Y(N - 2, l) + wall_relax * yo;
                max_update = std::max(max_update,
                    std::max(std::abs(xn - X(N - 2, l)), std::abs(yn - Y(N - 2, l))));
                X(N - 2, l) = xn; Y(N - 2, l) = yn;
            }
        }

        // SOR interior
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
                double denom   = 2.0 * (alpha_m / dphi2 + gamma_m / dpsi2);
                if (std::abs(denom) < kEps) continue;
                double source_x = alpha_m * P(k, l) * x_phi + gamma_m * Q(k, l) * x_psi;
                double source_y = alpha_m * P(k, l) * y_phi + gamma_m * Q(k, l) * y_psi;
                double mx = (X(k + 1, l + 1) - X(k + 1, l - 1)
                           - X(k - 1, l + 1) + X(k - 1, l - 1));
                double my = (Y(k + 1, l + 1) - Y(k + 1, l - 1)
                           - Y(k - 1, l + 1) + Y(k - 1, l - 1));
                double rhs_x = alpha_m * (X(k, l + 1) + X(k, l - 1)) / dphi2
                             + gamma_m * (X(k + 1, l) + X(k - 1, l)) / dpsi2
                             - beta_m * mx / twodphidpsi + source_x;
                double rhs_y = alpha_m * (Y(k, l + 1) + Y(k, l - 1)) / dphi2
                             + gamma_m * (Y(k + 1, l) + Y(k - 1, l)) / dpsi2
                             - beta_m * my / twodphidpsi + source_y;
                double xnv = rhs_x / denom;
                double ynv = rhs_y / denom;
                X(k, l) = (1.0 - sor_omega) * x_old + sor_omega * xnv;
                Y(k, l) = (1.0 - sor_omega) * y_old + sor_omega * ynv;
                max_update = std::max(max_update,
                    std::max(std::abs(X(k, l) - x_old), std::abs(Y(k, l) - y_old)));
            }
        }

        conv_p1.push_back(max_update);

        if (iter % 500 == 0 || iter == 1) {
            double sb, st;
            quick_wall_skew(X, Y, dphi, dpsi, N, sb, st);
            std::printf("    P1 iter %d: update=%.4e  wall_skew=[%.2f, %.2f]\n",
                        iter, max_update, sb, st);
        }
        if (max_update < convergence_tol) {
            p1_final = iter;
            double sb, st;
            quick_wall_skew(X, Y, dphi, dpsi, N, sb, st);
            std::printf("    P1 converged iter %d (update=%.4e, wall_skew=[%.2f, %.2f])\n",
                        iter, max_update, sb, st);
            break;
        }
    }

    // =====================================================================
    //  PHASE 2: Full-Domain Marching + Wall Sliding
    // =====================================================================
    if (!ortho_enable) {
        std::printf("  Phase 2: Skipped\n");
    } else {
        std::printf("  Phase 2: Marching + wall slide (sweeps=%d, march_relax=%.2f, slide_relax=%.2f)\n",
                    n_march, march_relax, slide_relax);

        // Spacings from Phase 1: spacing(k,l) = distance between (k,l) and (k+1,l)
        // MATLAB allocates (N-1)xN. We use NxN and only access rows 0..N-2.
        Eigen::MatrixXd spacing = Eigen::MatrixXd::Zero(N, N);
        for (int k = 0; k < N - 1; ++k) {
            for (int l = 0; l < N; ++l) {
                double dx = X(k + 1, l) - X(k, l);
                double dy = Y(k + 1, l) - Y(k, l);
                spacing(k, l) = std::sqrt(dx * dx + dy * dy);
            }
        }

        // Cosine blending weights: w_bot(k) = 0.5 * (1 + cos(pi*t)), t = k/(N-1)
        Eigen::VectorXd w_bot(N);
        for (int k = 0; k < N; ++k) {
            double t = static_cast<double>(k) / static_cast<double>(N - 1);
            w_bot(k) = 0.5 * (1.0 + std::cos(M_PI * t));
        }

        for (int sweep = 1; sweep <= n_march; ++sweep) {
            double max_corr = 0.0;

            // ---------- Bottom wall sliding ----------
            if (wall_slide && ortho_walls) {
                for (int l = 1; l < N - 1; ++l) {
                    double tx = (X(0, l + 1) - X(0, l - 1)) / (2.0 * dphi);
                    double ty = (Y(0, l + 1) - Y(0, l - 1)) / (2.0 * dphi);
                    double tmag = std::sqrt(tx * tx + ty * ty);
                    if (tmag < kEps) continue;
                    tx /= tmag; ty /= tmag;

                    double dx = X(1, l) - X(0, l);
                    double dy = Y(1, l) - Y(0, l);
                    double tang_comp = dx * tx + dy * ty;

                    double ds_slide = slide_relax * tang_comp;
                    double s_new = s_current_bot(l) + ds_slide;
                    s_new = std::max(s_wall_bot(0), std::min(s_total_bot, s_new));

                    double x_new, y_new;
                    interp_wall_curve(s_new, s_wall_bot, X_wall_bot_orig, Y_wall_bot_orig,
                                      x_new, y_new);

                    double dxc = x_new - X(0, l);
                    double dyc = y_new - Y(0, l);
                    max_corr = std::max(max_corr, std::sqrt(dxc * dxc + dyc * dyc));
                    s_current_bot(l) = s_new;
                    X(0, l) = x_new;
                    Y(0, l) = y_new;
                }

                // Refresh normals + spacing row 0 after bottom moved
                compute_wall_normals(X, Y, 0, dphi, N, +1, nx_bot, ny_bot);
                for (int l = 0; l < N; ++l) {
                    double dx = X(1, l) - X(0, l);
                    double dy = Y(1, l) - Y(0, l);
                    spacing(0, l) = std::sqrt(dx * dx + dy * dy);
                }
            }

            // ---------- Top wall sliding ----------
            if (wall_slide && ortho_outer) {
                for (int l = 1; l < N - 1; ++l) {
                    double tx = (X(N - 1, l + 1) - X(N - 1, l - 1)) / (2.0 * dphi);
                    double ty = (Y(N - 1, l + 1) - Y(N - 1, l - 1)) / (2.0 * dphi);
                    double tmag = std::sqrt(tx * tx + ty * ty);
                    if (tmag < kEps) continue;
                    tx /= tmag; ty /= tmag;

                    double dx = X(N - 2, l) - X(N - 1, l);
                    double dy = Y(N - 2, l) - Y(N - 1, l);
                    double tang_comp = dx * tx + dy * ty;

                    double ds_slide = slide_relax * tang_comp;
                    double s_new = s_current_top(l) + ds_slide;
                    s_new = std::max(s_wall_top(0), std::min(s_total_top, s_new));

                    double x_new, y_new;
                    interp_wall_curve(s_new, s_wall_top, X_wall_top_orig, Y_wall_top_orig,
                                      x_new, y_new);

                    double dxc = x_new - X(N - 1, l);
                    double dyc = y_new - Y(N - 1, l);
                    max_corr = std::max(max_corr, std::sqrt(dxc * dxc + dyc * dyc));
                    s_current_top(l) = s_new;
                    X(N - 1, l) = x_new;
                    Y(N - 1, l) = y_new;
                }

                compute_wall_normals(X, Y, N - 1, dphi, N, -1, nx_top, ny_top);
                for (int l = 0; l < N; ++l) {
                    double dx = X(N - 1, l) - X(N - 2, l);
                    double dy = Y(N - 1, l) - Y(N - 2, l);
                    spacing(N - 2, l) = std::sqrt(dx * dx + dy * dy);
                }
            }

            // ---------- March from bottom upward ----------
            Eigen::MatrixXd X_bot = X;
            Eigen::MatrixXd Y_bot = Y;
            if (ortho_walls) {
                for (int k = 1; k < N - 1; ++k) {
                    for (int l = 1; l < N - 1; ++l) {
                        double tx = (X_bot(k - 1, l + 1) - X_bot(k - 1, l - 1)) / (2.0 * dphi);
                        double ty = (Y_bot(k - 1, l + 1) - Y_bot(k - 1, l - 1)) / (2.0 * dphi);
                        double tm = std::sqrt(tx * tx + ty * ty);
                        if (tm < kEps) continue;
                        tx /= tm; ty /= tm;
                        double nx = -ty, ny = tx;
                        double dx = X(k, l) - X_bot(k - 1, l);
                        double dy = Y(k, l) - Y_bot(k - 1, l);
                        if ((nx * dx + ny * dy) < 0.0) { nx = -nx; ny = -ny; }
                        double sp = spacing(k - 1, l);
                        X_bot(k, l) = X_bot(k - 1, l) + sp * nx;
                        Y_bot(k, l) = Y_bot(k - 1, l) + sp * ny;
                    }
                }
            }

            // ---------- March from top downward ----------
            Eigen::MatrixXd X_top = X;
            Eigen::MatrixXd Y_top = Y;
            if (ortho_outer) {
                for (int k = N - 2; k >= 1; --k) {
                    for (int l = 1; l < N - 1; ++l) {
                        double tx = (X_top(k + 1, l + 1) - X_top(k + 1, l - 1)) / (2.0 * dphi);
                        double ty = (Y_top(k + 1, l + 1) - Y_top(k + 1, l - 1)) / (2.0 * dphi);
                        double tm = std::sqrt(tx * tx + ty * ty);
                        if (tm < kEps) continue;
                        tx /= tm; ty /= tm;
                        double nx = -ty, ny = tx;
                        double dx = X(k, l) - X_top(k + 1, l);
                        double dy = Y(k, l) - Y_top(k + 1, l);
                        if ((nx * dx + ny * dy) < 0.0) { nx = -nx; ny = -ny; }
                        double sp = spacing(k, l);  // distance between row k and k+1
                        X_top(k, l) = X_top(k + 1, l) + sp * nx;
                        Y_top(k, l) = Y_top(k + 1, l) + sp * ny;
                    }
                }
            }

            // ---------- Blend and apply relaxed correction ----------
            for (int k = 1; k < N - 1; ++k) {
                for (int l = 1; l < N - 1; ++l) {
                    double xt, yt;
                    if (ortho_walls && ortho_outer) {
                        xt = w_bot(k) * X_bot(k, l) + (1.0 - w_bot(k)) * X_top(k, l);
                        yt = w_bot(k) * Y_bot(k, l) + (1.0 - w_bot(k)) * Y_top(k, l);
                    } else if (ortho_walls) {
                        xt = X_bot(k, l); yt = Y_bot(k, l);
                    } else if (ortho_outer) {
                        xt = X_top(k, l); yt = Y_top(k, l);
                    } else {
                        continue;
                    }
                    double cx = march_relax * (xt - X(k, l));
                    double cy = march_relax * (yt - Y(k, l));
                    X(k, l) += cx;
                    Y(k, l) += cy;
                    max_corr = std::max(max_corr, std::sqrt(cx * cx + cy * cy));
                }
            }

            if (sweep % 50 == 0 || sweep == 1) {
                double sb, st;
                quick_wall_skew(X, Y, dphi, dpsi, N, sb, st);
                double si = quick_interior_skew(X, Y, dphi, dpsi, N);
                std::printf("    P2 sweep %d: corr=%.4e  wall=[%.2f, %.2f]  int=%.2f deg\n",
                            sweep, max_corr, sb, st, si);
            }
            if (max_corr < march_tol) {
                std::printf("    P2 converged sweep %d (corr=%.4e)\n", sweep, max_corr);
                break;
            }
        }
    }

    // =====================================================================
    //  Diagnostics
    // =====================================================================
    EllipticResult result;
    result.X_ellip_grid = X;
    result.Y_ellip_grid = Y;

    Eigen::VectorXd conv_p1_vec(static_cast<int>(conv_p1.size()));
    for (int i = 0; i < static_cast<int>(conv_p1.size()); ++i) conv_p1_vec(i) = conv_p1[i];
    result.diagnost.skewness            = compute_skewness(X, Y, dphi, dpsi, N);
    result.diagnost.jacobian            = compute_jacobian_field(X, Y, dphi, dpsi, N);

    const auto& skew = result.diagnost.skewness;
    const auto& J    = result.diagnost.jacobian;

    // Max over l=1..N-2 of skew(0, l), skew(N-1, l), and skew(1..N-2, 1..N-2)
    double skew_bot = 0.0, skew_top = 0.0, skew_int = 0.0;
    for (int l = 1; l < N - 1; ++l) {
        skew_bot = std::max(skew_bot, std::abs(skew(0,     l)));
        skew_top = std::max(skew_top, std::abs(skew(N - 1, l)));
    }
    for (int k = 1; k < N - 1; ++k) {
        for (int l = 1; l < N - 1; ++l) {
            skew_int = std::max(skew_int, std::abs(skew(k, l)));
        }
    }

    std::printf("  === Final ===\n");
    std::printf("  Skewness: bot=%.3f, top=%.3f, interior=%.3f deg\n",
                skew_bot, skew_top, skew_int);
    std::printf("  Jacobian: min=%.4e, max=%.4e\n", J.minCoeff(), J.maxCoeff());

    return result;
}