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

SparseCD solveA(const SparseCD& b, const Msh& msh, const dg& dg_, const int nvar)
{
    int p = msh.elem[0].p;
    int Ne = msh.elem.size();
    int Me = nvar * (p+1) * (p+1);

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

        #pragma omp for schedule(dynamic)
        for (int i1 = 0; i1 < Ne; i1++) {
            int j1 = 3 * Me * i1;
            int j3 = j1 + Me;

            const SparseCD& b_top = b_tops[i1];
            const SparseCD& b_bot = b_bots[i1];

            SparseCD r = b_top - dg_.QB[i1] * (dg_.QiD[i1] * b_bot);

            std::vector<int> ind;
            for (int col = 0; col < r.outerSize(); ++col)
                if (SparseCD::InnerIterator(r, col))
                    ind.push_back(col);

            if (ind.empty()) {
                SparseCD xx_bot = dg_.QiD[i1] * b_bot;
                for (int k = 0; k < xx_bot.outerSize(); ++k)
                    for (SparseCD::InnerIterator it(xx_bot, k); it; ++it)
                        if (std::abs(it.value()) > 1e-14)
                            local_trips.emplace_back(j3 + it.row(), it.col(), it.value());
                continue;
            }

            int nc = static_cast<int>(ind.size());

            Eigen::MatrixXcd r_dense(Me, nc);
            for (int jj = 0; jj < nc; ++jj) {
                r_dense.col(jj).setZero();
                for (SparseCD::InnerIterator it(r, ind[jj]); it; ++it)
                    r_dense(it.row(), jj) = it.value();
            }

            Eigen::PartialPivLU<Eigen::MatrixXcd> iK_lu(Eigen::MatrixXcd(dg_.iK[i1]));
            Eigen::MatrixXcd xx_top_dense = iK_lu.solve(r_dense);

            std::vector<Eigen::Triplet<Complex>> top_trips;
            for (int jj = 0; jj < nc; ++jj) {
                int col = ind[jj];
                for (int row = 0; row < Me; ++row)
                    if (std::abs(xx_top_dense(row, jj)) > 1e-14)
                        top_trips.emplace_back(row, col, xx_top_dense(row, jj));
            }
            SparseCD xx_top_sparse(Me, b.cols());
            xx_top_sparse.setFromTriplets(top_trips.begin(), top_trips.end());

            for (auto& t : top_trips)
                local_trips.emplace_back(j1 + t.row(), t.col(), t.value());

            SparseCD xx_bot = dg_.QiD[i1] * (b_bot - dg_.QC[i1] * xx_top_sparse);

            for (int k = 0; k < xx_bot.outerSize(); ++k)
                for (SparseCD::InnerIterator it(xx_bot, k); it; ++it)
                    if (std::abs(it.value()) > 1e-14)
                        local_trips.emplace_back(j3 + it.row(), it.col(), it.value());
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

        Eigen::PartialPivLU<Eigen::MatrixXcd> iK_lu(Eigen::MatrixXcd(dg_.iK[i1]));
        VectorCD xx_top = iK_lu.solve(r);

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

    SparseCD R;{
         // R = src_f - C * solveA(src_e)
        auto t3 = std::chrono::high_resolution_clock::now();
        
        VectorCD AinvE = solveA_vec(src_e, msh, dg_, nvar);
        VectorCD R = src_f - dg_.C * AinvE;
        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "solveA(E) + R: " << std::chrono::duration<double>(t4 - t3).count() << " s\n";
    }

    // // Trace solve: u_f = K \ R
    // Eigen::SparseLU<SparseCD> K_solver;
    // K_solver.compute(K);
    // if (K_solver.info() != Eigen::Success)
    //     std::cerr << "SparseLU factorization failed on global K\n";
    // result.u_f = K_solver.solve(R);
    // auto t4 = std::chrono::high_resolution_clock::now();
    // std::cout << "K\\R: " << std::chrono::duration<double>(t4 - t3).count() << " s\n";

    // // Back-sub: u_e = solveA(src_e - B * u_f)
    // VectorCD rhs_back = src_e - dg_.B * result.u_f;
    // result.u_e = solveA_vec(rhs_back, msh, dg_, nvar);
    // auto t5 = std::chrono::high_resolution_clock::now();
    // std::cout << "Back-sub: " << std::chrono::duration<double>(t5 - t4).count() << " s\n";

    // std::cout << "Total dg_solve: " << std::chrono::duration<double>(t5 - t0).count() << " s\n";

    return result;
}