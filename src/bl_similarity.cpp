#include "hdg/bl_similarity.h"
#include "hdg/sd.h"
#include "hdg/fd_coeffs_uniform.h"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <vector>

Eigen::SparseMatrix<double> bl_jacobian(const Eigen::VectorXd& q, int N,
    const SimulationParams& params, const Eigen::SparseMatrix<double>& D1,
    const Eigen::SparseMatrix<double>& D2, const Eigen::SparseMatrix<double>& D3) {

    double gam = params.gam;
    double M   = params.M;
    double Pr  = params.Pr;
    double Ts  = params.Ts;
    double Tw  = params.Tw / params.Te;

    Eigen::VectorXd f = q.head(N);
    Eigen::VectorXd g = q.tail(N);

    Eigen::VectorXd f2 = D2 * f;
    Eigen::VectorXd f3 = D3 * f;
    Eigen::VectorXd g1 = D1 * g;
    Eigen::VectorXd g2 = D2 * g;
    Eigen::VectorXd g3 = D3 * g;

    // df = [I Z], dg = [Z I]
    Eigen::SparseMatrix<double> df(N, 2 * N); // empty Nx2N sparse matrix
    Eigen::SparseMatrix<double> dg(N, 2 * N);
    {
        std::vector<Eigen::Triplet<double>> tdf, tdg;  // two empty lists
        tdf.reserve(N);  // pre-allocate to avoid resizing
        tdg.reserve(N);
        for (int i = 0; i < N; ++i) {
            tdf.emplace_back(i, i, 1.0);        // df gets a 1 at row i, col i
            tdg.emplace_back(i, i + N, 1.0);    // dg gets a 1 at row i, col i+N
        }
        df.setFromTriplets(tdf.begin(), tdf.end()); // Produces actual matrix
        dg.setFromTriplets(tdg.begin(), tdg.end());
    }

    Eigen::ArrayXd a  = (1.0 + Ts) / (g1.array() + Ts) * g1.array().sqrt();
    Eigen::ArrayXd a1 = (1.0 / (2.0 * g1.array()) - 1.0 / (g1.array() + Ts)) * a * g2.array();

    Eigen::SparseMatrix<double> da =
          sd((1.0 / (2.0 * g1.array()) - 1.0 / (g1.array() + Ts)) * a) * D1 * dg;

    Eigen::SparseMatrix<double> da1 =
          sd((-1.0 / (2.0 * g1.array().square()) + 1.0 / (g1.array() + Ts).square()) * a * g2.array()) * D1 * dg
        + sd((1.0 / (2.0 * g1.array()) - 1.0 / (g1.array() + Ts)) * g2.array()) * da
        + sd((1.0 / (2.0 * g1.array()) - 1.0 / (g1.array() + Ts)) * a) * D2 * dg;

    Eigen::SparseMatrix<double> term1 =
          sd(a)  * D3 * df
        + sd(f3) * da
        + sd(a1) * D2 * df
        + sd(f2) * da1
        + sd(f)  * D2 * df
        + sd(f2) * df;

    Eigen::SparseMatrix<double> term2 =
          (1.0 / Pr) * ( sd(a)  * D3 * dg
                       + sd(g3) * da
                       + sd(a1) * D2 * dg
                       + sd(g2) * da1 )
        + sd(f)  * D2 * dg
        + sd(g2) * df
        + (gam - 1.0) * M * M * ( sd((2.0 * a * f2.array()).matrix()) * D2 * df
                                + sd((f2.array() * f2.array()).matrix()) * da );

    // Assemble A from triplets, skipping BC rows in the governing-equation contributions.
    // BC rows (0-based): 0, 1, N-1 (f-equations) and N, N+1, 2N-1 (g-equations).
    // term1 occupies A rows 0..N-1   -> skip term1 rows {0, 1, N-1}
    // term2 occupies A rows N..2N-1  -> skip term2 rows {0, 1, N-1} (mapped to {N, N+1, 2N-1})

    // Allocate Sparse Matrix
    Eigen::SparseMatrix<double> A(2 * N, 2 * N);
    std::vector<Eigen::Triplet<double>> trips; // Basically a list sparse (row, col, value)
    trips.reserve(term1.nonZeros() + term2.nonZeros() + 40); // Pre-allocate memory

    // Anonymous function to determine BC row
    auto isBcRow = [N](int i) { return i == 0 || i == 1 || i == N - 1; };

    // Copies term1 into A matrix
    for (int k = 0; k < term1.outerSize(); ++k) // iterate over columns
        for (Eigen::SparseMatrix<double>::InnerIterator it(term1, k); it; ++it) // InnerIterator grabs every non-zero in column k. Returns col row value
            if (!isBcRow(it.row()))                                             
                trips.emplace_back(it.row(), it.col(), it.value()); // places this column into trips list   

    // Copies term 2 in A matrix at N + row.
    for (int k = 0; k < term2.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(term2, k); it; ++it)
            if (!isBcRow(it.row()))
                trips.emplace_back(it.row() + N, it.col(), it.value()); // add a new triplet to the list

    // BC triplets
    Eigen::VectorXd c;

    // Row 0: f(0) = 0
    trips.emplace_back(0, 0, 1.0);

    // Row 1: f'(0) = 0  (5th-order forward stencil 0..5)
    c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, 0.0, 5.0), 1);
    for (int k = 0; k < 6; ++k) trips.emplace_back(1, k, c[k] / c[1]);

    // Row N-1: f'(N) = 1  (5th-order backward stencil -5..0)
    c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -5.0, 0.0), 1);
    for (int k = 0; k < 6; ++k) trips.emplace_back(N - 1, N - 6 + k, c[k] / c[5]);

    // Row N: g(0) = 0
    trips.emplace_back(N, N, 1.0);

    // Row N+1: wall BC
    if (Tw > 0.0) {
        c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, 0.0, 5.0), 1);
        for (int k = 0; k < 6; ++k) trips.emplace_back(N + 1, N + k, c[k] / c[1]);
    } else {
        c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, 0.0, 6.0), 2);
        for (int k = 0; k < 7; ++k) trips.emplace_back(N + 1, N + k, c[k] / c[1]);
    }

    // Row 2N-1: g'(N) = 1
    c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -5.0, 0.0), 1);
    for (int k = 0; k < 6; ++k) trips.emplace_back(2 * N - 1, 2 * N - 6 + k, c[k] / c[5]);

    A.setFromTriplets(trips.begin(), trips.end()); // always use begin() and end()
    return A;
}


Eigen::VectorXd bl_resid(const Eigen::VectorXd& q, int N, double deta,
    const SimulationParams& params, const Eigen::SparseMatrix<double>& D1,
    const Eigen::SparseMatrix<double>& D2, const Eigen::SparseMatrix<double>& D3) {

    double gam = params.gam;
    double M   = params.M;
    double Pr  = params.Pr;
    double Ts  = params.Ts;
    double Tw  = params.Tw / params.Te;

    Eigen::VectorXd f = q.head(N);
    Eigen::VectorXd g = q.tail(N);

    Eigen::VectorXd f2 = D2 * f;
    Eigen::VectorXd f3 = D3 * f;
    Eigen::VectorXd g1 = D1 * g;
    Eigen::VectorXd g2 = D2 * g;
    Eigen::VectorXd g3 = D3 * g;

    Eigen::ArrayXd a  = (1.0 + Ts) / (g1.array() + Ts) * g1.array().sqrt();
    Eigen::ArrayXd a1 = (1.0 / (2.0 * g1.array()) - 1.0 / (g1.array() + Ts)) * a * g2.array();

    Eigen::ArrayXd f_res = a * f3.array() + a1 * f2.array() + f.array() * f2.array();
    Eigen::ArrayXd g_res = 1.0 / Pr * (a * g3.array() + a1 * g2.array())
                         + f.array() * g2.array()
                         + (gam - 1.0) * (M * M) * a * f2.array() * f2.array();

    // Boundary conditions
    Eigen::VectorXd c;

    f_res[0] = f[0]; // f(0) = 0
    c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, 0.0, 5.0), 1);
    f_res[1] = c.dot(f.head(6)) / c[1];

    c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -5.0, 0.0), 1);
    f_res[N - 1] = (c.dot(f.tail(6)) - deta) / c[5];

    g_res[0] = g[0];
    if (Tw > 0.0) {
        c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, 0.0, 5.0), 1);
        g_res[1] = (c.dot(g.head(6)) - deta * Tw) / c[1];
    } else {
        c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, 0.0, 6.0), 2);
        g_res[1] = c.dot(g.head(7)) / c[1];
    }
    c = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -5.0, 0.0), 1);
    g_res[N - 1] = (c.dot(g.tail(6)) - deta) / c[5];

    Eigen::VectorXd q_res(2 * N);
    q_res << f_res.matrix(), g_res.matrix();
    return q_res;
}


BlProfiles bl_similarity(const SimulationParams& params, int N) {

    double Tw = params.Tw / params.Te;
    double Re = params.Re_L;

    double deta  = 8.0 / (N - 1);
    double deta2 = deta * deta;
    double deta3 = deta * deta * deta;

    Eigen::VectorXd eta = Eigen::VectorXd::LinSpaced(N, 0.0, 8.0);

    // ---------- D1 (6th-order, 5th-order at boundaries) ----------
    Eigen::VectorXd s_cen = Eigen::VectorXd::LinSpaced(7, -3.0, 3.0);
    Eigen::VectorXd c_cen = fd_coeffs_uniform(s_cen, 1) / deta;

    Eigen::VectorXd c_r0  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, 0.0, 5.0), 1) / deta;
    Eigen::VectorXd c_r1  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -1.0, 4.0), 1) / deta;
    Eigen::VectorXd c_r2  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -2.0, 3.0), 1) / deta;
    Eigen::VectorXd c_rn3 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -3.0, 2.0), 1) / deta;
    Eigen::VectorXd c_rn2 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -4.0, 1.0), 1) / deta;
    Eigen::VectorXd c_rn1 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(6, -5.0, 0.0), 1) / deta;

    Eigen::SparseMatrix<double> D1(N, N);
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(N * 7);
        for (int i = 0; i < N; ++i) {
            if      (i == 0)     for (int k = 0; k < 6; ++k) trips.emplace_back(i, k, c_r0(k));
            else if (i == 1)     for (int k = 0; k < 6; ++k) trips.emplace_back(i, k, c_r1(k));
            else if (i == 2)     for (int k = 0; k < 6; ++k) trips.emplace_back(i, k, c_r2(k));
            else if (i == N - 3) for (int k = 0; k < 6; ++k) trips.emplace_back(i, N - 6 + k, c_rn3(k));
            else if (i == N - 2) for (int k = 0; k < 6; ++k) trips.emplace_back(i, N - 6 + k, c_rn2(k));
            else if (i == N - 1) for (int k = 0; k < 6; ++k) trips.emplace_back(i, N - 6 + k, c_rn1(k));
            else                 for (int k = 0; k < 7; ++k) trips.emplace_back(i, i - 3 + k, c_cen(k));
        }
        D1.setFromTriplets(trips.begin(), trips.end());
    }

    // ---------- D2 (6th-order central, 5th-order biased at boundaries) ----------
    Eigen::VectorXd c2_cen = fd_coeffs_uniform(s_cen, 2) / deta2;

    Eigen::VectorXd c2_r0  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, 0.0, 6.0),  2) / deta2;
    Eigen::VectorXd c2_r1  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, -1.0, 5.0), 2) / deta2;
    Eigen::VectorXd c2_r2  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, -2.0, 4.0), 2) / deta2;
    Eigen::VectorXd c2_rn3 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, -4.0, 2.0), 2) / deta2;
    Eigen::VectorXd c2_rn2 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, -5.0, 1.0), 2) / deta2;
    Eigen::VectorXd c2_rn1 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(7, -6.0, 0.0), 2) / deta2;

    Eigen::SparseMatrix<double> D2(N, N);
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(N * 7);
        for (int i = 0; i < N; ++i) {
            if      (i == 0)     for (int k = 0; k < 7; ++k) trips.emplace_back(i, k, c2_r0(k));
            else if (i == 1)     for (int k = 0; k < 7; ++k) trips.emplace_back(i, k, c2_r1(k));
            else if (i == 2)     for (int k = 0; k < 7; ++k) trips.emplace_back(i, k, c2_r2(k));
            else if (i == N - 3) for (int k = 0; k < 7; ++k) trips.emplace_back(i, N - 7 + k, c2_rn3(k));
            else if (i == N - 2) for (int k = 0; k < 7; ++k) trips.emplace_back(i, N - 7 + k, c2_rn2(k));
            else if (i == N - 1) for (int k = 0; k < 7; ++k) trips.emplace_back(i, N - 7 + k, c2_rn1(k));
            else                 for (int k = 0; k < 7; ++k) trips.emplace_back(i, i - 3 + k, c2_cen(k));
        }
        D2.setFromTriplets(trips.begin(), trips.end());
    }

    // ---------- D3 (6th-order wall-biased, 5th-order biased at boundaries) ----------
    Eigen::VectorXd s3_cen = Eigen::VectorXd::LinSpaced(9, -5.0, 3.0);
    Eigen::VectorXd c3_cen = fd_coeffs_uniform(s3_cen, 3) / deta3;

    Eigen::VectorXd c3_r0  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, 0.0, 7.0),  3) / deta3;
    Eigen::VectorXd c3_r1  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -1.0, 6.0), 3) / deta3;
    Eigen::VectorXd c3_r2  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -2.0, 5.0), 3) / deta3;
    Eigen::VectorXd c3_r3  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -3.0, 4.0), 3) / deta3;
    Eigen::VectorXd c3_r4  = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -4.0, 3.0), 3) / deta3;
    Eigen::VectorXd c3_rn3 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -5.0, 2.0), 3) / deta3;
    Eigen::VectorXd c3_rn2 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -6.0, 1.0), 3) / deta3;
    Eigen::VectorXd c3_rn1 = fd_coeffs_uniform(Eigen::VectorXd::LinSpaced(8, -7.0, 0.0), 3) / deta3;

    Eigen::SparseMatrix<double> D3(N, N);
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(N * 9);
        for (int i = 0; i < N; ++i) {
            if      (i == 0)     for (int k = 0; k < 8; ++k) trips.emplace_back(i, k, c3_r0(k));
            else if (i == 1)     for (int k = 0; k < 8; ++k) trips.emplace_back(i, k, c3_r1(k));
            else if (i == 2)     for (int k = 0; k < 8; ++k) trips.emplace_back(i, k, c3_r2(k));
            else if (i == 3)     for (int k = 0; k < 8; ++k) trips.emplace_back(i, k, c3_r3(k));
            else if (i == 4)     for (int k = 0; k < 8; ++k) trips.emplace_back(i, k, c3_r4(k));
            else if (i == N - 3) for (int k = 0; k < 8; ++k) trips.emplace_back(i, N - 8 + k, c3_rn3(k));
            else if (i == N - 2) for (int k = 0; k < 8; ++k) trips.emplace_back(i, N - 8 + k, c3_rn2(k));
            else if (i == N - 1) for (int k = 0; k < 8; ++k) trips.emplace_back(i, N - 8 + k, c3_rn1(k));
            else                 for (int k = 0; k < 9; ++k) trips.emplace_back(i, i - 5 + k, c3_cen(k));
        }
        D3.setFromTriplets(trips.begin(), trips.end());
    }

    // ---------- Initial guess: f = g = eta ----------
    Eigen::VectorXd f = eta;
    Eigen::VectorXd g = eta;

    Eigen::VectorXd q(2 * N);
    q << f, g;

    // ---------- Newton iteration ----------
    Eigen::VectorXd q_res = bl_resid(q, N, deta, params, D1, D2, D3);
    double rnorm = q_res.norm();
    double rnorm_old;

    int Maxiter = 20;
    int i1;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    for (i1 = 0; i1 < Maxiter; ++i1) {
        Eigen::SparseMatrix<double> A = bl_jacobian(q, N, params, D1, D2, D3);

        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Sparse LU factorization failed at iter " << i1 << "\n";
            break;
        }
        Eigen::VectorXd dq = solver.solve(q_res);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Sparse LU solve failed at iter " << i1 << "\n";
            break;
        }
        q -= dq;

        q_res = bl_resid(q, N, deta, params, D1, D2, D3);

        rnorm_old = rnorm;
        rnorm = q_res.norm();

        if ((rnorm < 1e-4) && (rnorm > rnorm_old / 2)) break;
    }

    if (i1 == Maxiter) {
        std::cerr << "Warning: max iterations reached in bl_similarity; "
                  << "solution may not be converged.\n";
    }

    // ---------- Build output profiles ----------
    f = q.head(N);
    g = q.tail(N);

    BlProfiles bl_profiles;
    bl_profiles.y = std::sqrt(2.0 / Re) * g;
    bl_profiles.u = D1 * f;
    bl_profiles.T = D1 * g;
    bl_profiles.v = 1.0 / std::sqrt(2.0 * Re)
                  * (bl_profiles.u.array() * g.array() - f.array() * bl_profiles.T.array()).matrix();

    return bl_profiles;
}