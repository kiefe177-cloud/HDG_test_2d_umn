#include "hdg/flux_lin_euler.h"
#include "hdg/sd.h"
#include <vector>

namespace{
Eigen::SparseMatrix<double> selector(int var_idx, int n, int N){
    Eigen::SparseMatrix<double> S(n,N);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(n);
    for (int i = 0; i < n; ++i){
        trips.emplace_back(i, var_idx * n + i, 1.0);
    }
    S.setFromTriplets(trips.begin(), trips.end());
    return S;
}

Eigen::SparseMatrix<double> vstack5(const Eigen::SparseMatrix<double>& A0,
                                    const Eigen::SparseMatrix<double>& A1,
                                    const Eigen::SparseMatrix<double>& A2,
                                    const Eigen::SparseMatrix<double>& A3,
                                    const Eigen::SparseMatrix<double>& A4)
{    
    const int n = static_cast<int>(A0.rows());
    const int N = static_cast<int>(A0.cols());

    Eigen::SparseMatrix<double> M(5 * n,N);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(A0.nonZeros() + A1.nonZeros() + A2.nonZeros() + A3.nonZeros() + A4.nonZeros());

    auto append = [&](const Eigen::SparseMatrix<double>& A, int row_offset){
        for (int k = 0; k < A.outerSize(); ++k){
            for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it){
                trips.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()),it.value());
            }
        }
    };

    append(A0, 0 * n);
    append(A1, 1 * n);
    append(A2, 2 * n);
    append(A3, 3 * n);
    append(A4, 4 * n);

    M.setFromTriplets(trips.begin(), trips.end());
    return M;
}
}
inviscid_fluxes flux_lin_euler(const Eigen::VectorXd& bU, const SimulationParams& params, const int nvar){
    
    const double gam = params.gam;
    const int N = static_cast<int>(bU.size());
    const int n = N / nvar;
    
    const Eigen::SparseMatrix<double> rho  = selector(0, n, N);
    const Eigen::SparseMatrix<double> rhou = selector(1, n, N);
    const Eigen::SparseMatrix<double> rhov = selector(2, n, N);
    const Eigen::SparseMatrix<double> rhow = selector(3, n, N);
    const Eigen::SparseMatrix<double> E    = selector(4, n, N);

    const Eigen::VectorXd brho_v    = rho * bU;
    const Eigen::VectorXd brhou_v   = rhou * bU;
    const Eigen::VectorXd brhov_v   = rhov * bU;
    const Eigen::VectorXd brhow_v   = rhow * bU;
    const Eigen::VectorXd bE_v      = E * bU;

    const Eigen::VectorXd birho_v   = brho_v.cwiseInverse();
    const Eigen::VectorXd bu_v      = brhou_v.cwiseQuotient(brho_v);
    const Eigen::VectorXd bv_v      = brhov_v.cwiseQuotient(brho_v);
    const Eigen::VectorXd bw_v      = brhow_v.cwiseQuotient(brho_v);
    
    const Eigen::VectorXd kinetic = 0.5*(brhou_v.array().square() 
                                    + brhov_v.array().square()
                                    + brhow_v.array().square()).matrix().cwiseQuotient(brho_v);

    const Eigen::VectorXd bp_v = (gam - 1.0) * (bE_v - kinetic);

    const Eigen::SparseMatrix<double> birho = sd(birho_v);
    const Eigen::SparseMatrix<double> bu    = sd(bu_v);
    const Eigen::SparseMatrix<double> bv    = sd(bv_v);
    const Eigen::SparseMatrix<double> bw    = sd(bw_v);
    const Eigen::SparseMatrix<double> bp    = sd(bp_v);
    const Eigen::SparseMatrix<double> brho  = sd(brho_v);
    const Eigen::SparseMatrix<double> brhou = sd(brhou_v);
    const Eigen::SparseMatrix<double> brhov = sd(brhov_v);
    const Eigen::SparseMatrix<double> brhow = sd(brhow_v);
    const Eigen::SparseMatrix<double> bE    = sd(bE_v);

    Eigen::SparseMatrix<double> u = birho * (rhou - bu*rho);
    Eigen::SparseMatrix<double> v = birho * (rhov - bv*rho);
    Eigen::SparseMatrix<double> w = birho * (rhow - bw*rho);
    Eigen::SparseMatrix<double> p = (gam - 1.0) *
            (E - bu * rhou - bv*rhov - bw*rhow 
            + 0.5*(bu*bu+bv*bv+bw*bw)*rho);
    
    inviscid_fluxes out;

    out.F = vstack5(
        rhou,
        brhou*u + bu*rhou + p,
        brhov*u + bu*rhov,
        brhow*u + bu*rhow,
        (bE + bp)*u + bu*(E+p));

        
    out.G = vstack5(
        rhov,
        brhou*v + bv*rhou,
        brhov*v + bv*rhov + p,
        brhow*v + bv*rhow,
        (bE + bp)*v + bv*(E+p));

        
    out.H = vstack5(
        rhow,
        brhou*w + bw*rhou,
        brhov*w + bw*rhov,
        brhow*w + bw*rhow + p,
        (bE + bp)*w + bw*(E+p));

    return out;
}
