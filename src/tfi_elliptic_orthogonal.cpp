#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "hdg/tfi_elliptic_orthogonal.h"

namespace {

constexpr double kRad2Deg = 180.0 / M_PI;

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
            if (mp < 1e-14 || ms < 1e-14) continue;

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
            if (mp > 1e-14 && ms > 1e-14) {
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
            if (mp > 1e-14 && ms > 1e-14) {
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
//  Two-phase elliptic + full-domain orthogonality solver
// =====================================================================
EllipticResult tfi_elliptic_orthogonal(
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
    std::printf("Starting elliptic solver with full-domain orthogonality...\n");

    const bool   ortho_enable = ortho_opts.enable;
    const bool   ortho_walls  = ortho_opts.walls;
    const bool   ortho_outer  = ortho_opts.outer;
    const double ortho_relax  = ortho_opts.relax;
    const int    n_ortho      = ortho_opts.n_ortho_sweeps;
    const double ortho_tol    = ortho_opts.ortho_tol;

    const int N   = n_grid_pts;
    const int end = N - 1;  // matches MATLAB's "end"

    const double dphi  = phi_comp_vec(1) - phi_comp_vec(0);
    const double dpsi  = psi_comp_vec(1) - psi_comp_vec(0);
    const double dphi2 = dphi * dphi;
    const double dpsi2 = dpsi * dpsi;
    const double twodphidpsi = 2.0 * dphi * dpsi;

    // Working copies (modified in-place)
    Eigen::MatrixXd X = X_tfi;
    Eigen::MatrixXd Y = Y_tfi;

    // =====================================================================
    //  PHASE 1: Standard TTM elliptic solve
    // =====================================================================
    std::printf("  Phase 1: TTM elliptic solve...\n");

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);

    // P from bottom and top walls
    for (int l = 1; l < end; ++l) {
        double xp  = (X(0, l + 1) - X(0, l - 1)) / (2.0 * dphi);
        double yp  = (Y(0, l + 1) - Y(0, l - 1)) / (2.0 * dphi);
        double xpp = (X(0, l + 1) - 2.0 * X(0, l) + X(0, l - 1)) / dphi2;
        double ypp = (Y(0, l + 1) - 2.0 * Y(0, l) + Y(0, l - 1)) / dphi2;
        P(0, l) = -(xp * xpp + yp * ypp) / (xp * xp + yp * yp + 1e-14);

        xp  = (X(end, l + 1) - X(end, l - 1)) / (2.0 * dphi);
        yp  = (Y(end, l + 1) - Y(end, l - 1)) / (2.0 * dphi);
        xpp = (X(end, l + 1) - 2.0 * X(end, l) + X(end, l - 1)) / dphi2;
        ypp = (Y(end, l + 1) - 2.0 * Y(end, l) + Y(end, l - 1)) / dphi2;
        P(end, l) = -(xp * xpp + yp * ypp) / (xp * xp + yp * yp + 1e-14);
    }
    P(0,   0)   = P(0,   1);     P(0,   end) = P(0,   end - 1);
    P(end, 0)   = P(end, 1);     P(end, end) = P(end, end - 1);

    // Q from left and right walls
    for (int k = 1; k < end; ++k) {
        double xq  = (X(k + 1, 0) - X(k - 1, 0)) / (2.0 * dpsi);
        double yq  = (Y(k + 1, 0) - Y(k - 1, 0)) / (2.0 * dpsi);
        double xqq = (X(k + 1, 0) - 2.0 * X(k, 0) + X(k - 1, 0)) / dpsi2;
        double yqq = (Y(k + 1, 0) - 2.0 * Y(k, 0) + Y(k - 1, 0)) / dpsi2;
        Q(k, 0) = -(xq * xqq + yq * yqq) / (xq * xq + yq * yq + 1e-14);

        xq  = (X(k + 1, end) - X(k - 1, end)) / (2.0 * dpsi);
        yq  = (Y(k + 1, end) - Y(k - 1, end)) / (2.0 * dpsi);
        xqq = (X(k + 1, end) - 2.0 * X(k, end) + X(k - 1, end)) / dpsi2;
        yqq = (Y(k + 1, end) - 2.0 * Y(k, end) + Y(k - 1, end)) / dpsi2;
        Q(k, end) = -(xq * xqq + yq * yqq) / (xq * xq + yq * yq + 1e-14);
    }
    Q(0,   0)   = Q(1,       0); Q(end, 0)   = Q(end - 1, 0);
    Q(0,   end) = Q(1,     end); Q(end, end) = Q(end - 1, end);

    // 1D linear interpolation: P along psi, Q along phi
    for (int k = 1; k < end; ++k) {
        for (int l = 1; l < end; ++l) {
            double s_psi = static_cast<double>(k) / end;
            double s_phi = static_cast<double>(l) / end;
            P(k, l) = (1.0 - s_psi) * P(0, l) + s_psi * P(end, l);
            Q(k, l) = (1.0 - s_phi) * Q(k, 0) + s_phi * Q(k, end);
        }
    }

    // SOR iteration
    for (int iter = 1; iter <= n_elliptic_iter; ++iter) {
        double max_update_iter = 0.0;

        for (int k = 1; k < end; ++k) {
            for (int l = 1; l < end; ++l) {
                double x_old = X(k, l);
                double y_old = Y(k, l);

                double x_phi = (X(k, l + 1) - X(k, l - 1)) / (2.0 * dphi);
                double x_psi = (X(k + 1, l) - X(k - 1, l)) / (2.0 * dpsi);
                double y_phi = (Y(k, l + 1) - Y(k, l - 1)) / (2.0 * dphi);
                double y_psi = (Y(k + 1, l) - Y(k - 1, l)) / (2.0 * dpsi);

                double alpha_m = x_psi * x_psi + y_psi * y_psi;
                double beta_m  = x_phi * x_psi + y_phi * y_psi;
                double gamma_m = x_phi * x_phi + y_phi * y_phi;

                double denom = 2.0 * (alpha_m / dphi2 + gamma_m / dpsi2);
                if (std::abs(denom) < 1e-14) continue;

                double source_x = alpha_m * P(k, l) * x_phi + gamma_m * Q(k, l) * x_psi;
                double source_y = alpha_m * P(k, l) * y_phi + gamma_m * Q(k, l) * y_psi;

                double mx = X(k + 1, l + 1) - X(k + 1, l - 1) - X(k - 1, l + 1) + X(k - 1, l - 1);
                double my = Y(k + 1, l + 1) - Y(k + 1, l - 1) - Y(k - 1, l + 1) + Y(k - 1, l - 1);

                double rhs_x = alpha_m * (X(k, l + 1) + X(k, l - 1)) / dphi2
                             + gamma_m * (X(k + 1, l) + X(k - 1, l)) / dpsi2
                             - beta_m  * mx / twodphidpsi
                             + source_x;
                double rhs_y = alpha_m * (Y(k, l + 1) + Y(k, l - 1)) / dphi2
                             + gamma_m * (Y(k + 1, l) + Y(k - 1, l)) / dpsi2
                             - beta_m  * my / twodphidpsi
                             + source_y;

                double x_new = rhs_x / denom;
                double y_new = rhs_y / denom;

                X(k, l) = (1.0 - sor_omega) * x_old + sor_omega * x_new;
                Y(k, l) = (1.0 - sor_omega) * y_old + sor_omega * y_new;

                max_update_iter = std::max(max_update_iter, std::abs(X(k, l) - x_old));
                max_update_iter = std::max(max_update_iter, std::abs(Y(k, l) - y_old));
            }
        }

        if (iter % 500 == 0 || iter == 1) {
            std::printf("    TTM iter %d: max_update = %.4e\n", iter, max_update_iter);
        }
        if (max_update_iter < convergence_tol) {
            std::printf("    TTM converged at iter %d\n", iter);
            break;
        }
    }

    // =====================================================================
    //  PHASE 2: Full-domain orthogonality correction
    // =====================================================================
    if (!ortho_enable) {
        std::printf("  Phase 2: Skipped (orthogonality disabled)\n");
    } else {
        std::printf("  Phase 2: Orthogonal marching correction (%d sweeps, relax=%.2f)...\n",
                    n_ortho, ortho_relax);

        // Spacing field: distance between row k and row k+1 at column l.
        // Frozen after Phase 1; preserved while Phase 2 corrects directions.
        Eigen::MatrixXd spacing_field(N - 1, N);
        for (int k = 0; k < N - 1; ++k) {
            for (int l = 0; l < N; ++l) {
                double dx = X(k + 1, l) - X(k, l);
                double dy = Y(k + 1, l) - Y(k, l);
                spacing_field(k, l) = std::sqrt(dx * dx + dy * dy);
            }
        }

        // Cosine blending weights: w_bot = 1 at bottom, 0 at top, smooth.
        Eigen::VectorXd w_bot(N);
        for (int k = 0; k < N; ++k) {
            double t = static_cast<double>(k) / end;
            w_bot(k) = 0.5 * (1.0 + std::cos(M_PI * t));
        }

        for (int sweep = 1; sweep <= n_ortho; ++sweep) {
            double max_correction = 0.0;

            // ------ March from bottom wall (k=0) upward ------
            Eigen::MatrixXd X_bot = X;
            Eigen::MatrixXd Y_bot = Y;
            if (ortho_walls) {
                for (int k = 1; k < end; ++k) {
                    for (int l = 1; l < end; ++l) {
                        // Tangent along the previous row (k-1)
                        double tx = (X_bot(k - 1, l + 1) - X_bot(k - 1, l - 1)) / (2.0 * dphi);
                        double ty = (Y_bot(k - 1, l + 1) - Y_bot(k - 1, l - 1)) / (2.0 * dphi);
                        double tmag = std::sqrt(tx * tx + ty * ty);
                        if (tmag < 1e-14) continue;
                        tx /= tmag;
                        ty /= tmag;

                        // Normal (perpendicular to tangent)
                        double nx = -ty;
                        double ny =  tx;

                        // Ensure normal points from k-1 toward k (use current X/Y)
                        double dx = X(k, l) - X_bot(k - 1, l);
                        double dy = Y(k, l) - Y_bot(k - 1, l);
                        if (nx * dx + ny * dy < 0.0) { nx = -nx; ny = -ny; }

                        double sp = spacing_field(k - 1, l);
                        X_bot(k, l) = X_bot(k - 1, l) + sp * nx;
                        Y_bot(k, l) = Y_bot(k - 1, l) + sp * ny;
                    }
                }
            }

            // ------ March from top wall (k=end) downward ------
            Eigen::MatrixXd X_top = X;
            Eigen::MatrixXd Y_top = Y;
            if (ortho_outer) {
                for (int k = end - 1; k >= 1; --k) {
                    for (int l = 1; l < end; ++l) {
                        // Tangent along the previous row (k+1)
                        double tx = (X_top(k + 1, l + 1) - X_top(k + 1, l - 1)) / (2.0 * dphi);
                        double ty = (Y_top(k + 1, l + 1) - Y_top(k + 1, l - 1)) / (2.0 * dphi);
                        double tmag = std::sqrt(tx * tx + ty * ty);
                        if (tmag < 1e-14) continue;
                        tx /= tmag;
                        ty /= tmag;

                        double nx = -ty;
                        double ny =  tx;

                        // Ensure normal points from k+1 toward k
                        double dx = X(k, l) - X_top(k + 1, l);
                        double dy = Y(k, l) - Y_top(k + 1, l);
                        if (nx * dx + ny * dy < 0.0) { nx = -nx; ny = -ny; }

                        // Spacing between row k and row k+1
                        double sp = spacing_field(k, l);
                        X_top(k, l) = X_top(k + 1, l) + sp * nx;
                        Y_top(k, l) = Y_top(k + 1, l) + sp * ny;
                    }
                }
            }

            // ------ Blend the two marched fields, apply relaxed correction ------
            for (int k = 1; k < end; ++k) {
                for (int l = 1; l < end; ++l) {
                    double x_target, y_target;
                    if (ortho_walls && ortho_outer) {
                        x_target = w_bot(k) * X_bot(k, l) + (1.0 - w_bot(k)) * X_top(k, l);
                        y_target = w_bot(k) * Y_bot(k, l) + (1.0 - w_bot(k)) * Y_top(k, l);
                    } else if (ortho_walls) {
                        x_target = X_bot(k, l);
                        y_target = Y_bot(k, l);
                    } else if (ortho_outer) {
                        x_target = X_top(k, l);
                        y_target = Y_top(k, l);
                    } else {
                        continue;
                    }

                    double corr_x = ortho_relax * (x_target - X(k, l));
                    double corr_y = ortho_relax * (y_target - Y(k, l));

                    X(k, l) += corr_x;
                    Y(k, l) += corr_y;

                    max_correction = std::max(max_correction,
                                              std::sqrt(corr_x * corr_x + corr_y * corr_y));
                }
            }

            if (sweep % 50 == 0 || sweep == 1) {
                // Wall skewness for monitoring
                double skew_bot = 0.0, skew_top = 0.0;
                for (int l = 1; l < end; ++l) {
                    {
                        double xp = (X(0, l + 1) - X(0, l - 1)) / (2.0 * dphi);
                        double yp = (Y(0, l + 1) - Y(0, l - 1)) / (2.0 * dphi);
                        double xs = (X(1, l) - X(0, l)) / dpsi;
                        double ys = (Y(1, l) - Y(0, l)) / dpsi;
                        double mp = std::sqrt(xp * xp + yp * yp);
                        double ms = std::sqrt(xs * xs + ys * ys);
                        if (mp > 1e-14 && ms > 1e-14) {
                            double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                            skew_bot = std::max(skew_bot,
                                                std::abs(90.0 - std::acos(ca) * kRad2Deg));
                        }
                    }
                    {
                        double xp = (X(end, l + 1) - X(end, l - 1)) / (2.0 * dphi);
                        double yp = (Y(end, l + 1) - Y(end, l - 1)) / (2.0 * dphi);
                        double xs = (X(end, l) - X(end - 1, l)) / dpsi;
                        double ys = (Y(end, l) - Y(end - 1, l)) / dpsi;
                        double mp = std::sqrt(xp * xp + yp * yp);
                        double ms = std::sqrt(xs * xs + ys * ys);
                        if (mp > 1e-14 && ms > 1e-14) {
                            double ca = std::clamp((xp * xs + yp * ys) / (mp * ms), -1.0, 1.0);
                            skew_top = std::max(skew_top,
                                                std::abs(90.0 - std::acos(ca) * kRad2Deg));
                        }
                    }
                }
                std::printf("    Ortho sweep %d: max_corr=%.4e, wall_skew=[%.2f, %.2f] deg\n",
                            sweep, max_correction, skew_bot, skew_top);
            }

            if (max_correction < ortho_tol) {
                std::printf("    Ortho converged at sweep %d (max_corr=%.4e)\n",
                            sweep, max_correction);
                break;
            }
        }
    }

    // =====================================================================
    //  Final diagnostics
    // =====================================================================
    Diagnostics diag;
    diag.skewness = compute_skewness(X, Y, dphi, dpsi, N);
    diag.jacobian = compute_jacobian_field(X, Y, dphi, dpsi, N);

    const auto& skew = diag.skewness;
    const auto& Jf   = diag.jacobian;

    double bot_max = 0.0, top_max = 0.0, int_max = 0.0;
    for (int l = 1; l < end; ++l) {
        bot_max = std::max(bot_max, std::abs(skew(0,   l)));
        top_max = std::max(top_max, std::abs(skew(end, l)));
    }
    for (int k = 1; k < end; ++k) {
        for (int l = 1; l < end; ++l) {
            int_max = std::max(int_max, std::abs(skew(k, l)));
        }
    }

    std::printf("  --- Final diagnostics ---\n");
    std::printf("  Skewness: bot wall max = %.2f, top wall max = %.2f\n", bot_max, top_max);
    std::printf("  Skewness interior max = %.2f deg\n", int_max);
    std::printf("  Jacobian: min = %.4e, max = %.4e\n", Jf.minCoeff(), Jf.maxCoeff());

    return EllipticResult{X, Y, diag};
}