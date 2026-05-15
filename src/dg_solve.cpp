#include "hdg/dg_solve.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <complex>

namespace {
using Complex   = std::complex<double>;
using SparseCD  = Eigen::SparseMatrix<Complex>;
using SparseD   = Eigen::SparseMatrix<double>;
using VectorCD  = Eigen::VectorXcd;
using MatrixCD  = Eigen::MatrixXcd;

// Apply QiD = dg.D_factors[e]^{-1} to a complex vector by splitting real/imag.
// (D_factors holds a real SparseLU; if it ever becomes complex, this collapses
// to a single .solve(col) call.)
VectorCD apply_QiD(const dg& dg_, int e, const VectorCD& col)
{
    Eigen::VectorXd sr = dg_.D_factors[e]->solve(col.real().eval());
    Eigen::VectorXd si = dg_.D_factors[e]->solve(col.imag().eval());
    VectorCD out(col.size());
    out.real() = sr;
    out.imag() = si;
    return out;
}

// ---------------------------------------------------------------------------
// solveA (sparse RHS): applies A^{-1} to a sparse multi-column RHS.
// Used during K-assembly where the RHS is dg.B and most columns are zero.
//
// Per-element layout: each element owns 3*Me consecutive rows, split as
//   [j1, j1+Me)        -> "upper" block (Me rows)         e.g. volume DOFs
//   [j1+Me, j1+3*Me)   -> "lower" block (2*Me rows)       e.g. element-face DOFs
// ---------------------------------------------------------------------------
SparseCD solveA(const SparseCD& b, const Msh& msh, const dg& dg_, const int nvar)
{
    const int p     = msh.elem[0].p;                  // "fix later" per the MATLAB
    const int Ne    = static_cast<int>(msh.elem.size());
    const int Me    = nvar * (p + 1) * (p + 1);
    const int n_rhs = static_cast<int>(b.cols());

    auto t0 = std::chrono::high_resolution_clock::now();

    // Result holders, two blocks per element (upper Me rows, lower 2*Me rows)
    std::vector<MatrixCD> xx(2 * Ne);

    for (int e = 0; e < Ne; ++e) {
        const int j1 = 3 * Me * e;
        const int j2 = j1 + Me;            // [j1, j2)  -> upper Me rows
        const int j3 = j2;                 // [j3, j3 + 2*Me) -> lower 2*Me rows

        // Extract b(j1:j2,:) and b(j3:j4,:) as dense blocks.
        // b is sparse; iterate nonzeros rather than densifying the whole thing.
        MatrixCD b_top = MatrixCD::Zero(Me,     n_rhs);
        MatrixCD b_bot = MatrixCD::Zero(2 * Me, n_rhs);

        for (int k = 0; k < n_rhs; ++k) {
            for (SparseCD::InnerIterator it(b, k); it; ++it) {
                const int row = static_cast<int>(it.row());
                if      (row >= j1 && row < j2)            b_top(row - j1, k) = it.value();
                else if (row >= j3 && row < j3 + 2 * Me)   b_bot(row - j3, k) = it.value();
            }
        }

        // QiD \ b_bot  (the inner "D" solve), column by column
        MatrixCD QiD_b_bot(2 * Me, n_rhs);
        for (int k = 0; k < n_rhs; ++k)
            QiD_b_bot.col(k) = apply_QiD(dg_, e, b_bot.col(k));

        // r = b_top - QB[e] * QiD_b_bot
        MatrixCD r = b_top - MatrixCD(dg_.QB[e]) * QiD_b_bot;

        // Sparse-RHS optimization: solve iK \ r only on nonzero columns
        std::vector<int> nz_cols;
        nz_cols.reserve(n_rhs);
        for (int k = 0; k < n_rhs; ++k) {
            if (r.col(k).cwiseAbs().sum() != 0.0) nz_cols.push_back(k);
        }

        MatrixCD upper = MatrixCD::Zero(Me, n_rhs);

        if (!nz_cols.empty()) {
            MatrixCD r_sub(Me, static_cast<int>(nz_cols.size()));
            for (size_t kk = 0; kk < nz_cols.size(); ++kk)
                r_sub.col(static_cast<int>(kk)) = r.col(nz_cols[kk]);

            // TODO: cache these factorizations alongside dg.Ik to avoid
            // refactoring on every solveA call.
            Eigen::SparseLU<SparseCD> iK_solver;
            iK_solver.compute(dg_.Ik[e]);
            MatrixCD tt = iK_solver.solve(r_sub);

            for (size_t kk = 0; kk < nz_cols.size(); ++kk)
                upper.col(nz_cols[kk]) = tt.col(static_cast<int>(kk));
        }

        xx[2 * e] = upper;

        // lower = QiD \ (b_bot - QC * upper)
        MatrixCD rhs_lower = b_bot - MatrixCD(dg_.QC[e]) * upper;
        MatrixCD lower(2 * Me, n_rhs);
        for (int k = 0; k < n_rhs; ++k)
            lower.col(k) = apply_QiD(dg_, e, rhs_lower.col(k));

        xx[2 * e + 1] = lower;
    }

    // Stitch the per-element blocks into one global sparse matrix (cell2mat)
    const int rows_total = 3 * Me * Ne;
    SparseCD x(rows_total, n_rhs);

    std::vector<Eigen::Triplet<Complex>> trips;
    trips.reserve(static_cast<size_t>(rows_total) * n_rhs / 8);  // rough guess

    for (int e = 0; e < Ne; ++e) {
        const int row0_top = 3 * Me * e;
        const int row0_bot = row0_top + Me;
        const MatrixCD& top = xx[2 * e];
        const MatrixCD& bot = xx[2 * e + 1];

        for (int k = 0; k < n_rhs; ++k) {
            for (int i = 0; i < Me; ++i) {
                const Complex v = top(i, k);
                if (v != Complex(0.0)) trips.emplace_back(row0_top + i, k, v);
            }
            for (int i = 0; i < 2 * Me; ++i) {
                const Complex v = bot(i, k);
                if (v != Complex(0.0)) trips.emplace_back(row0_bot + i, k, v);
            }
        }
    }
    x.setFromTriplets(trips.begin(), trips.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Solve A done, "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    return x;
}

// ---------------------------------------------------------------------------
// solveA (dense RHS): applies A^{-1} to a single dense vector.
// Used for src_e and the back-substitution RHS. No sparse-column trickery
// since a dense vector has no all-zero "columns" to exploit.
// ---------------------------------------------------------------------------
VectorCD solveA(const VectorCD& b, const Msh& msh, const dg& dg_, int nvar)
{
    const int p  = msh.elem[0].p;
    const int Ne = static_cast<int>(msh.elem.size());
    const int Me = nvar * (p + 1) * (p + 1);

    auto t0 = std::chrono::high_resolution_clock::now();

    VectorCD x(3 * Me * Ne);

    for (int e = 0; e < Ne; ++e) {
        const int j1 = 3 * Me * e;
        const int j3 = j1 + Me;

        // Slice b into top (Me) and bottom (2*Me) pieces
        VectorCD b_top = b.segment(j1, Me);
        VectorCD b_bot = b.segment(j3, 2 * Me);

        // QiD \ b_bot  (the inner "D" solve)
        VectorCD QiD_b_bot = apply_QiD(dg_, e, b_bot);

        // r = b_top - QB[e] * QiD_b_bot
        VectorCD r = b_top - MatrixCD(dg_.QB[e]) * QiD_b_bot;

        // upper = iK[e] \ r
        // TODO: cache these factorizations alongside dg.Ik.
        Eigen::SparseLU<SparseCD> iK_solver;
        iK_solver.compute(dg_.Ik[e]);
        VectorCD upper = iK_solver.solve(r);

        // lower = QiD \ (b_bot - QC[e] * upper)
        VectorCD lower = apply_QiD(dg_, e,
                                   b_bot - MatrixCD(dg_.QC[e]) * upper);

        x.segment(j1, Me)     = upper;
        x.segment(j3, 2 * Me) = lower;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Solve A (dense) done, "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    return x;
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// dg_solve: HDG Schur complement solve
//
//   [ A  B ] [ U   ]   [ E ]
//   [ C  D ] [ U^ ] = [ F ]
//
//   K  = D - C A^{-1} B
//   R  = F - C A^{-1} E
//   U^ = K^{-1} R                  (trace solve)
//   U  = A^{-1} ( E - B U^ )       (back-substitution)
// ---------------------------------------------------------------------------
HDG_Result dg_solve(const Eigen::VectorXcd& src_e,
                    const Eigen::VectorXcd& src_f,
                    const Msh& msh, const dg& dg_, int nvar)
{
    HDG_Result result;

    // K = D - C * (A^{-1} B) — sparse overload because dg.B is sparse
    SparseCD AinvB = solveA(dg_.B, msh, dg_, nvar);
    SparseCD K     = dg_.D - dg_.C * AinvB;

    // R = src_f - C * (A^{-1} src_e) — dense overload; sparse * dense = dense
    VectorCD AinvE = solveA(src_e, msh, dg_, nvar);
    VectorCD R     = src_f - dg_.C * AinvE;

    // Trace solve
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::SparseLU<SparseCD> K_solver;
    K_solver.analyzePattern(K);
    K_solver.factorize(K);
    if (K_solver.info() != Eigen::Success) {
        std::cerr << "Schur complement factorization failed\n";
        return result;
    }
    result.u_f = K_solver.solve(R);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Schur complement solve, "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // u_e = A^{-1} ( src_e - B * u_f )
    VectorCD rhs_e = src_e - dg_.B * result.u_f;
    result.u_e = solveA(rhs_e, msh, dg_, nvar);

    return result;
}