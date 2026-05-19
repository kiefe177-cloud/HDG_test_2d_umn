#include "hdg/dg_solve.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <complex>
// #include <Eigen/PardisoSupport>

namespace {
using Complex   = std::complex<double>;
using SparseCD  = Eigen::SparseMatrix<Complex>;
using SparseD   = Eigen::SparseMatrix<double>;
using VectorCD  = Eigen::VectorXcd;
using MatrixCD  = Eigen::MatrixXcd;

SparseCD solveA(const SparseCD& b, const Msh& msh, const dg& dg_, const int nvar)
{
    int p  = msh.elem[0].p;
    int Ne = msh.elem.size();
    int Me = nvar * (p + 1) * (p + 1);

    // Pre-extract each element's rows from b (middleRows is not thread-safe on sparse)
    std::vector<SparseCD> b_tops(Ne), b_bots(Ne);
    for (int i1 = 0; i1 < Ne; i1++) {
        int j1 = 3 * Me * i1;
        int j3 = j1 + Me;
        b_tops[i1] = b.middleRows(j1, Me);
        b_bots[i1] = b.middleRows(j3, 2 * Me);
    }

    int nthreads = omp_get_max_threads();
    std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_trips = thread_triplets[tid];
        local_trips.reserve(b.nonZeros() * 4 / nthreads);

        if (tid == 0)
            std::cout << "solveA running on " << omp_get_num_threads() << " threads" << std::endl;

        #pragma omp for schedule(dynamic)
        for (int i1 = 0; i1 < Ne; i1++) {
            // auto ta = std::chrono::high_resolution_clock::now();

            int j1 = 3 * Me * i1;
            int j3 = j1 + Me;

            const SparseCD& b_top = b_tops[i1];
            const SparseCD& b_bot = b_bots[i1];

            // r = b_top - QB * (QiD * b_bot)
            SparseCD r = b_top - dg_.QB[i1] * (dg_.QiD[i1] * b_bot);
            // auto tb = std::chrono::high_resolution_clock::now();

            // Find columns of r that have at least one nonzero
            std::vector<int> ind;
            ind.reserve(r.outerSize());
            for (int col = 0; col < r.outerSize(); ++col) {
                SparseCD::InnerIterator it(r, col);
                if (it) ind.push_back(col);
            }

            // ------------------------------------------------------------------
            // Always emit bottom contribution from QiD * b_bot (over b_bot's columns)
            // ------------------------------------------------------------------
            SparseCD QiD_bbot = dg_.QiD[i1] * b_bot;
            for (int k = 0; k < QiD_bbot.outerSize(); ++k)
                for (SparseCD::InnerIterator it(QiD_bbot, k); it; ++it)
                    if (std::abs(it.value()) > 1e-14)
                        local_trips.emplace_back(j3 + it.row(), it.col(), it.value());

            // If r is empty there's no top block and no QC correction
            if (ind.empty())
                continue;

            int nc = static_cast<int>(ind.size());

            // Pack the nc active columns of r into a dense (Me x nc) matrix
            Eigen::MatrixXcd r_dense = Eigen::MatrixXcd::Zero(Me, nc);
            for (int jj = 0; jj < nc; ++jj)
                for (SparseCD::InnerIterator it(r, ind[jj]); it; ++it)
                    r_dense(it.row(), jj) = it.value();

            // Solve K * xx_top = r for the active columns
            Eigen::MatrixXcd xx_top_dense = dg_.iK_factors[i1]->solve(r_dense);
            // auto tc = std::chrono::high_resolution_clock::now();

            // Scatter top block directly into local_trips
            for (int jj = 0; jj < nc; ++jj) {
                int col = ind[jj];
                for (int row = 0; row < Me; ++row) {
                    Complex v = xx_top_dense(row, jj);
                    if (std::abs(v) > 1e-14)
                        local_trips.emplace_back(j1 + row, col, v);
                }
            }
            // auto td = std::chrono::high_resolution_clock::now();

            // Bottom correction: -QiD * QC * xx_top, dense, over ind columns
            Eigen::MatrixXcd corr = dg_.QiD[i1] * (dg_.QC[i1] * xx_top_dense);
            for (int jj = 0; jj < nc; ++jj) {
                int col = ind[jj];
                for (int row = 0; row < 2 * Me; ++row) {
                    Complex v = -corr(row, jj);
                    if (std::abs(v) > 1e-14)
                        local_trips.emplace_back(j3 + row, col, v);
                }
            }
            // auto te = std::chrono::high_resolution_clock::now();

            // #pragma omp critical
            // {
            //     std::cout << "Elem " << i1
            //               << "  nc=" << nc
            //               << "  r="     << std::chrono::duration<double>(tb - ta).count()
            //               << "  solve=" << std::chrono::duration<double>(tc - tb).count()
            //               << "  top="   << std::chrono::duration<double>(td - tc).count()
            //               << "  bot="   << std::chrono::duration<double>(te - td).count()
            //               << " s\n";
            // }
        }
    }

    // Merge all thread-local triplets
    size_t total = 0;
    for (auto& v : thread_triplets) total += v.size();
    std::vector<Eigen::Triplet<Complex>> triplets;
    triplets.reserve(total);
    for (auto& v : thread_triplets)
        triplets.insert(triplets.end(), v.begin(), v.end());

    SparseCD x(b.rows(), b.cols());
    x.setFromTriplets(triplets.begin(), triplets.end());
    return x;
}

VectorCD solveA_vec(const VectorCD& b, const Msh& msh, const dg& dg_, const int nvar)
{
    int p = msh.elem[0].p;
    int Ne = msh.elem.size();
    int Me = nvar * (p+1) * (p+1);

    VectorCD x = VectorCD::Zero(b.rows());

    #pragma omp parallel for schedule(dynamic)
    for (int i1 = 0; i1 < Ne; i1++) {
        int j1 = 3 * Me * i1;
        int j3 = j1 + Me;

        VectorCD b_top = b.segment(j1, Me);
        VectorCD b_bot = b.segment(j3, 2 * Me);

        VectorCD r = b_top - dg_.QB[i1] * (dg_.QiD[i1] * b_bot);

        // Eigen::PartialPivLU<Eigen::MatrixXcd> iK_lu(Eigen::MatrixXcd(dg_.iK[i1]));
        // VectorCD xx_top = iK_lu.solve(r);

        VectorCD xx_top = dg_.iK_factors[i1]->solve(r);

        VectorCD xx_bot = dg_.QiD[i1] * (b_bot - dg_.QC[i1] * xx_top);

        x.segment(j1, Me)     = xx_top;
        x.segment(j3, 2 * Me) = xx_bot;
    }
    return x;
}

}  // anonymous namespace

HDG_Result dg_solve(const VectorCD& src_e, const VectorCD& src_f,
                    const Msh& msh, const dg& dg_, int nvar)
{
    HDG_Result result;

    auto t0 = std::chrono::high_resolution_clock::now();

    // K = D - C * solveA(B)
    SparseCD K;
    {
        SparseCD AinvB = solveA(dg_.B, msh, dg_, nvar);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "solveA(B): " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

        K = dg_.D - dg_.C * AinvB;
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "D - C*AinvB: " << std::chrono::duration<double>(t2 - t1).count() << " s\n";
    }  // AinvB freed here

    VectorCD R;{
         // R = src_f - C * solveA(src_e)
        auto t3 = std::chrono::high_resolution_clock::now();
        
        VectorCD AinvE = solveA_vec(src_e, msh, dg_, nvar);
        R = src_f - dg_.C * AinvE;
        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "solveA(E) + R: " << std::chrono::duration<double>(t4 - t3).count() << " s\n";
    }

    auto t3b = std::chrono::high_resolution_clock::now();

    Eigen::SparseLU<SparseCD> K_solver;
    K_solver.analyzePattern(K);
    K_solver.factorize(K);
    if (K_solver.info() != Eigen::Success) {
        std::cerr << "Pardiso factorization failed on global K\n";
        return result;
    }

    result.u_f = K_solver.solve(R);
    if (K_solver.info() != Eigen::Success) {
        std::cerr << "Pardiso solve failed on global K\n";
        return result;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "K\\R: " << std::chrono::duration<double>(t4 - t3b).count() << " s\n";

    VectorCD rhs_back = src_e - dg_.B * result.u_f;
    result.u_e = solveA_vec(rhs_back, msh, dg_, nvar);
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Back-sub: " << std::chrono::duration<double>(t5 - t4).count() << " s\n";

    std::cout << "Total dg_solve: " << std::chrono::duration<double>(t5 - t0).count() << " s\n";
    
    result.K = K;
    result.R = R;
    return result;
}