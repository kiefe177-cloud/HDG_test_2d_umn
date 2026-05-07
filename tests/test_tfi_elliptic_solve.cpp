// Tests for tfi_elliptic_solve.
//
// tfi_elliptic_solve is a plain Winslow/TTM elliptic smoother — no
// orthogonality features, no wall sliding. It takes a TFI initial grid
// and relaxes the interior toward the harmonic solution implied by the
// boundary curves.
//
// Test A: Pure smoothing on a noisy unit square. The solver should
//   recover the clean unit square from a noisy interior.
//
// Test B: Smoothing on a clean curved-wall grid. Bottom wall is a gentle
//   Gaussian bump, top wall is straight. Verifies that the Jacobian stays
//   positive (no folded cells) and the solver converges. Wall skewness is
//   reported but NOT checked — this solver doesn't enforce orthogonality,
//   so we don't expect it to drop dramatically.

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <string>

#include "hdg/tfi_elliptic_solve.h"

namespace {

void save_csv(const std::string& name, const Eigen::MatrixXd& mat) {
    std::ofstream file(name);
    const static Eigen::IOFormat CSVFormat(
        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    file << mat.format(CSVFormat);
}

// Total variation of a grid field: sum of |X(i,j+1)-X(i,j)| + |X(i+1,j)-X(i,j)|
// in both directions. Lower = smoother.
double total_variation(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    int N = static_cast<int>(X.rows());
    double tv = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            tv += std::abs(X(i, j + 1) - X(i, j));
            tv += std::abs(Y(i, j + 1) - Y(i, j));
        }
    }
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N; ++j) {
            tv += std::abs(X(i + 1, j) - X(i, j));
            tv += std::abs(Y(i + 1, j) - Y(i, j));
        }
    }
    return tv;
}

// Max wall skewness in degrees (deviation from 90 deg) at a given wall row.
double wall_skewness_deg(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                         int k_wall, int k_int, double dphi, double dpsi) {
    int N = static_cast<int>(X.cols());
    double s_max = 0.0;
    int sign = (k_int > k_wall) ? +1 : -1;
    for (int l = 1; l < N - 1; ++l) {
        double xp = (X(k_wall, l + 1) - X(k_wall, l - 1)) / (2.0 * dphi);
        double yp = (Y(k_wall, l + 1) - Y(k_wall, l - 1)) / (2.0 * dphi);
        double xs = sign * (X(k_int, l) - X(k_wall, l)) / dpsi;
        double ys = sign * (Y(k_int, l) - Y(k_wall, l)) / dpsi;
        double mp = std::sqrt(xp * xp + yp * yp);
        double ms = std::sqrt(xs * xs + ys * ys);
        if (mp < 1e-14 || ms < 1e-14) continue;
        double ca = std::max(-1.0, std::min(1.0, (xp * xs + yp * ys) / (mp * ms)));
        double dev = std::abs(90.0 - std::acos(ca) * 180.0 / M_PI);
        s_max = std::max(s_max, dev);
    }
    return s_max;
}

// Verify the four boundary curves are unchanged within tol.
bool boundaries_preserved(const Eigen::MatrixXd& X_ref, const Eigen::MatrixXd& Y_ref,
                          const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                          double tol) {
    int N = static_cast<int>(X.rows());
    for (int j = 0; j < N; ++j) {
        if (std::abs(X(0, j)     - X_ref(0, j))     > tol) return false;
        if (std::abs(Y(0, j)     - Y_ref(0, j))     > tol) return false;
        if (std::abs(X(N - 1, j) - X_ref(N - 1, j)) > tol) return false;
        if (std::abs(Y(N - 1, j) - Y_ref(N - 1, j)) > tol) return false;
    }
    for (int i = 0; i < N; ++i) {
        if (std::abs(X(i, 0)     - X_ref(i, 0))     > tol) return false;
        if (std::abs(Y(i, 0)     - Y_ref(i, 0))     > tol) return false;
        if (std::abs(X(i, N - 1) - X_ref(i, N - 1)) > tol) return false;
        if (std::abs(Y(i, N - 1) - Y_ref(i, N - 1)) > tol) return false;
    }
    return true;
}

// =====================================================================
//  Test A: Pure smoothing on a noisy unit square.
//  Boundaries form a clean unit square, interior is noisy. The harmonic
//  solution is the clean square, so the smoothed grid should converge to it.
// =====================================================================
int test_pure_smoothing() {
    std::cout << "\n========== Test A: Pure smoothing on noisy unit square ==========\n";

    const int N = 30;

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, N);
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);
    Eigen::VectorXd psi = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);

    // Clean unit square
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            X(i, j) = phi(j);
            Y(i, j) = psi(i);
        }
    }
    Eigen::MatrixXd X_ref = X, Y_ref = Y;

    // Add noise to interior only (boundaries untouched). Distinct phases
    // for X and Y so noise isn't purely diagonal.
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            X(i, j) += 0.03 * std::sin(2.7 * i) * std::cos(3.1 * j);
            Y(i, j) += 0.03 * std::cos(3.3 * i) * std::sin(2.9 * j);
        }
    }

    save_csv(std::string(OUTPUT_DIR_TEST) + "/testA_solve_noisy_x.csv", X);
    save_csv(std::string(OUTPUT_DIR_TEST)_TAGLC_ID_DEFINED + "/testA_solve_noisy_y.csv", Y);

    auto result = tfi_elliptic_solve(
        X, Y, phi, psi, N,
        /*n_elliptic_iter=*/20000, /*sor_omega=*/1.6, /*tol=*/1e-10);

    save_csv(std::string(OUTPUT_DIR_TEST) + "/testA_solve_smooth_x.csv", result.X_ellip_grid);
    save_csv(std::string(OUTPUT_DIR_TEST) + "/testA_solve_smooth_y.csv", result.Y_ellip_grid);

    bool bnd_ok = boundaries_preserved(
        X_ref, Y_ref, result.X_ellip_grid, result.Y_ellip_grid, 1e-12);
    double tv_noisy  = total_variation(X, Y);
    double tv_smooth = total_variation(result.X_ellip_grid, result.Y_ellip_grid);

    std::cout << "  Boundaries preserved: " << (bnd_ok ? "yes" : "NO") << "\n";
    std::cout << "  Total variation: noisy=" << tv_noisy
              << ", smooth=" << tv_smooth << "\n";

    if (!bnd_ok) {
        std::cerr << "  [FAIL] boundary moved\n";
        return 1;
    }
    if (tv_smooth >= tv_noisy) {
        std::cerr << "  [FAIL] grid did not get smoother\n";
        return 1;
    }

    // Winslow's solution on a clean square boundary IS the clean square,
    // so the smoothed interior should converge to it.
    double max_err = 0.0;
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            max_err = std::max(max_err, std::abs(result.X_ellip_grid(i, j) - X_ref(i, j)));
            max_err = std::max(max_err, std::abs(result.Y_ellip_grid(i, j) - Y_ref(i, j)));
        }
    }
    std::cout << "  Max deviation from clean square interior: " << max_err << "\n";
    if (max_err > 1e-3) {
        std::cerr << "  [FAIL] interior didn't recover the clean square\n";
        return 1;
    }
    std::cout << "  [PASS]\n";
    return 0;
}

// =====================================================================
//  Test B: Smoothing on a clean curved-wall grid.
//  Bottom wall is a gentle Gaussian bump. Top wall is straight at y=1.
//  Interior initialized as a vertical algebraic blend.
//
//  This solver does NOT enforce orthogonality, so the wall skewness is
//  reported for visibility but not checked. The pass criteria are that
//  the boundaries are preserved and no cell folds.
// =====================================================================
int test_curved_wall() {
    std::cout << "\n========== Test B: Smoothing on curved bottom wall ==========\n";

    const int N = 40;
    Eigen::MatrixXd X(N, N), Y(N, N);
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);
    Eigen::VectorXd psi = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);

    auto bump_y = [](double x) {
        return 0.05 * std::exp(-30.0 * (x - 0.5) * (x - 0.5));
    };

    for (int j = 0; j < N; ++j) {
        double x = phi(j);
        double y_bot = bump_y(x);
        double y_top = 1.0;
        for (int i = 0; i < N; ++i) {
            double s = psi(i);  // 0 at bottom, 1 at top
            X(i, j) = x;
            Y(i, j) = (1.0 - s) * y_bot + s * y_top;
        }
    }
    Eigen::MatrixXd X_ref = X, Y_ref = Y;

    save_csv(std::string(OUTPUT_DIR_TEST) + "/testB_solve_initial_x.csv", X);
    save_csv(std::string(OUTPUT_DIR_TEST) + "/testB_solve_initial_y.csv", Y);

    double dphi = phi(1) - phi(0);
    double dpsi = psi(1) - psi(0);

    double skew_bot_before = wall_skewness_deg(X, Y, 0,     1,     dphi, dpsi);
    double skew_top_before = wall_skewness_deg(X, Y, N - 1, N - 2, dphi, dpsi);

    auto result = tfi_elliptic_solve(
        X, Y, phi, psi, N,
        /*n_elliptic_iter=*/20000, /*sor_omega=*/1.5, /*tol=*/1e-10);

    save_csv(std::string(OUTPUT_DIR_TEST) + "/testB_solve_smooth_x.csv", result.X_ellip_grid);
    save_csv(std::string(OUTPUT_DIR_TEST) + "/testB_solve_smooth_y.csv", result.Y_ellip_grid);

    // Boundaries must remain fixed.
    bool bnd_ok = boundaries_preserved(
        X_ref, Y_ref, result.X_ellip_grid, result.Y_ellip_grid, 1e-12);
    std::cout << "  Boundaries preserved: " << (bnd_ok ? "yes" : "NO") << "\n";
    if (!bnd_ok) {
        std::cerr << "  [FAIL] boundary moved\n";
        return 1;
    }

    double skew_bot_after = wall_skewness_deg(
        result.X_ellip_grid, result.Y_ellip_grid, 0, 1, dphi, dpsi);
    double skew_top_after = wall_skewness_deg(
        result.X_ellip_grid, result.Y_ellip_grid, N - 1, N - 2, dphi, dpsi);

    std::cout << "  Bottom wall skewness: " << skew_bot_before
              << " deg -> " << skew_bot_after << " deg (informational)\n";
    std::cout << "  Top wall skewness:    " << skew_top_before
              << " deg -> " << skew_top_after << " deg (informational)\n";

    // Jacobian must stay positive in the interior (no folded cells).
    // Boundary entries are not computed and remain zero, so check only
    // the interior block.
    const auto& J = result.diagnost.jacobian;
    double jmin = J.block(1, 1, N - 2, N - 2).minCoeff();
    double jmax = J.block(1, 1, N - 2, N - 2).maxCoeff();
    std::cout << "  Interior Jacobian: min=" << jmin << ", max=" << jmax << "\n";
    if (jmin <= 0.0) {
        std::cerr << "  [FAIL] grid folded (non-positive interior Jacobian)\n";
        return 1;
    }

    std::cout << "  [PASS]\n";
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= test_pure_smoothing();
    rc |= test_curved_wall();
    if (rc == 0) {
        std::cout << "\nAll tests passed.\n";
    } else {
        std::cerr << "\nOne or more tests failed.\n";
    }
    return rc;
}