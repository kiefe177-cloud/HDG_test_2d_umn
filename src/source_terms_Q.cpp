#include "hdg/source_terms_Q.h"
#include "hdg/sd.h"
#include "hdg/vec.h"
#include <vector>
#include <complex>

namespace {

using cdouble = std::complex<double>;
using SpMatC  = Eigen::SparseMatrix<cdouble>;
using SpMatD  = Eigen::SparseMatrix<double>;
using VecD    = Eigen::VectorXd;

SpMatD selector(int var_idx, int n, int N) {
    SpMatD S(n, N);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(n);
    for (int i = 0; i < n; ++i) {
        trips.emplace_back(i, var_idx * n + i, 1.0);
    }
    S.setFromTriplets(trips.begin(), trips.end());
    return S;
}

// Promote a real sparse matrix to complex (zero imaginary part).
SpMatC to_complex(const SpMatD& A) {
    SpMatC B(A.rows(), A.cols());
    std::vector<Eigen::Triplet<cdouble>> trips;
    trips.reserve(A.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SpMatD::InnerIterator it(A, k); it; ++it) {
            trips.emplace_back(static_cast<int>(it.row()),
                               static_cast<int>(it.col()),
                               cdouble(it.value(), 0.0));
        }
    }
    B.setFromTriplets(trips.begin(), trips.end());
    return B;
}

// Vertically stack 5 complex sparse matrices of equal column count.
SpMatC vstack5(const SpMatC& A0, const SpMatC& A1, const SpMatC& A2,
               const SpMatC& A3, const SpMatC& A4)
{
    const int n = static_cast<int>(A0.rows());
    const int N = static_cast<int>(A0.cols());
    SpMatC Mout(5 * n, N);

    std::vector<Eigen::Triplet<cdouble>> trips;
    trips.reserve(A0.nonZeros() + A1.nonZeros() + A2.nonZeros()
                + A3.nonZeros() + A4.nonZeros());

    auto append = [&](const SpMatC& A, int row_offset) {
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SpMatC::InnerIterator it(A, k); it; ++it) {
                trips.emplace_back(static_cast<int>(it.row()) + row_offset,
                                   static_cast<int>(it.col()),
                                   it.value());
            }
        }
    };
    append(A0, 0 * n);
    append(A1, 1 * n);
    append(A2, 2 * n);
    append(A3, 3 * n);
    append(A4, 4 * n);

    Mout.setFromTriplets(trips.begin(), trips.end());
    return Mout;
}

} // anonymous namespace


source_terms source_terms_Q(const Eigen::VectorXd& bU,
                            const Eigen::VectorXd& bUx,
                            const Eigen::VectorXd& bUy,
                            const Eigen::VectorXd& bUz,
                            const SimulationParams& params,
                            const int nvar,
                            const int m,
                            const Eigen::MatrixXd& invr_in)
{
    // ===== physical constants =====
    const double Mach   = params.M;
    const double gam    = params.gam;
    const double Ts     = params.Ts;
    const double Re     = params.Re_L;

    // ===== sizes =====
    const int N = static_cast<int>(bU.size());
    const int n = N / nvar;
    const int Ntot = 4 * N;

    // ===== narrow selectors and background values (real) =====
    const SpMatD sel_rho  = selector(0, n, N);
    const SpMatD sel_rhou = selector(1, n, N);
    const SpMatD sel_rhov = selector(2, n, N);
    const SpMatD sel_rhow = selector(3, n, N);
    const SpMatD sel_E    = selector(4, n, N);

    const VecD brho_v   = sel_rho * bU;
    const VecD brhou_v  = sel_rhou * bU;
    const VecD brhov_v  = sel_rhov * bU;
    const VecD brhow_v  = sel_rhow * bU;
    const VecD bE_v     = sel_E    * bU;

    const VecD brho_x_v  = sel_rho  * bUx;
    const VecD brhou_x_v = sel_rhou * bUx;

    const VecD brho_y_v  = sel_rho  * bUy;
    const VecD brhov_y_v = sel_rhov * bUy;
    const VecD brhow_y_v = sel_rhow * bUy;

    const VecD brho_z_v  = sel_rho  * bUz;
    const VecD brhov_z_v = sel_rhov * bUz;
    const VecD brhow_z_v = sel_rhow * bUz;

    const VecD binvr_v = vec(invr_in);

    // ===== wide Real selectors (nxNtot) =====
    const SpMatD rho    = selector(0,  n, Ntot);
    const SpMatD rhou   = selector(1,  n, Ntot);
    const SpMatD rhov   = selector(2,  n, Ntot);
    const SpMatD rhow   = selector(3,  n, Ntot);
    const SpMatD E      = selector(4,  n, Ntot);
    const SpMatD rho_x  = selector(5,  n, Ntot);
    const SpMatD rhou_x = selector(6,  n, Ntot);
    //const SpMatD rhov_x = selector(7,  n, Ntot);
    //const SpMatD rhow_x = selector(8,  n, Ntot);
    //const SpMatD E_x    = selector(9,  n, Ntot);
    const SpMatD rho_y  = selector(10, n, Ntot);
    //const SpMatD rhou_y = selector(11, n, Ntot);
    const SpMatD rhov_y = selector(12, n, Ntot);
    const SpMatD rhow_y = selector(13, n, Ntot);
    //const SpMatD E_y    = selector(14, n, Ntot);

    const VecD brho_sq_v = brho_v.array().square().matrix();

    const VecD birho_v = brho_v.cwiseInverse();
    const VecD birho_x_v = -brho_x_v.cwiseQuotient(brho_sq_v);
    const VecD birho_y_v = -brho_y_v.cwiseQuotient(brho_sq_v);

    const VecD bu_v = brhou_v.cwiseQuotient(brho_v);
    const VecD bv_v = brhov_v.cwiseQuotient(brho_v);
    const VecD bw_v = brhow_v.cwiseQuotient(brho_v);

    auto vel_grad = [&](const VecD& brhoq_d, const VecD& bq, const VecD& brho_d) {
        return ((brhoq_d - bq.cwiseProduct(brho_d)).cwiseQuotient(brho_v)).eval();
    };

    const VecD bu_x_v = vel_grad(brhou_x_v, bu_v, brho_x_v);
    const VecD bv_y_v = vel_grad(brhov_y_v, bv_v, brho_y_v);
    const VecD bv_z_v = vel_grad(brhov_z_v, bv_v, brho_z_v);
    const VecD bw_y_v = vel_grad(brhow_y_v, bw_v, brho_y_v);
    const VecD bw_z_v = vel_grad(brhow_z_v, bw_v, brho_z_v);

    const VecD bdivu_v = bu_x_v + bv_y_v
                       + bv_v.cwiseProduct(binvr_v)
                       + bw_z_v.cwiseProduct(binvr_v);

    const VecD kinetic_v = 0.5 * (brhou_v.array().square()
                                + brhov_v.array().square()
                                + brhow_v.array().square()).matrix()
                                .cwiseQuotient(brho_v);

    const VecD bp_v = (gam - 1.0) * (bE_v - kinetic_v);
    const VecD bT_v = gam * Mach * Mach * bp_v.cwiseQuotient(brho_v);

    const VecD bmu_v =
        ((1.0 + Ts) / (bT_v.array() + Ts) * bT_v.array().pow(1.5)).matrix();     
        
    const VecD dmu_dT_v =
        ((1.0 + Ts) * bT_v.array().sqrt()
         * (1.5 * (bT_v.array() + Ts) - bT_v.array())
         / (bT_v.array() + Ts).square()).matrix();

    // ===== lift to REAL diagonals =====
    const SpMatD bmu     = sd(bmu_v);
    const SpMatD birho   = sd(birho_v);
    const SpMatD birho_x = sd(birho_x_v);
    const SpMatD birho_y = sd(birho_y_v);
    const SpMatD bu      = sd(bu_v);
    const SpMatD bv      = sd(bv_v);
    const SpMatD bw      = sd(bw_v);
    const SpMatD brhow   = sd(brhow_v);
    const SpMatD brhov   = sd(brhov_v);
    const SpMatD bu_x    = sd(bu_x_v);
    const SpMatD bv_y    = sd(bv_y_v);
    const SpMatD bv_z    = sd(bv_z_v);
    const SpMatD bw_y    = sd(bw_y_v);
    const SpMatD bw_z    = sd(bw_z_v);
    const SpMatD binvr   = sd(binvr_v);
    const SpMatD bdivu   = sd(bdivu_v);
    const SpMatD bE      = sd(bE_v);
    const SpMatD dmu_dT  = sd(dmu_dT_v);

    // ===== background strain rates (REAL — m doesn't enter here) =====
    const double two_thirds = 2.0 / 3.0;
    const SpMatD bSzz = 2.0 * (bw_z * binvr + bv * binvr) - two_thirds * bdivu;
    const SpMatD bSyz = bv_z * binvr + bw_y - bw * binvr;

    // ===== linearized REAL operators (no m) =====
    const SpMatD u = birho * (rhou - bu * rho);
    const SpMatD v = birho * (rhov - bv * rho);
    const SpMatD w = birho * (rhow - bw * rho);

    
    const SpMatD u_x = birho * (rhou_x - bu * rho_x - bu_x * rho)
                     + birho_x * (rhou - bu * rho);
    const SpMatD v_y = birho * (rhov_y - bv * rho_y - bv_y * rho)
                     + birho_y * (rhov - bv * rho);
    const SpMatD w_y = birho * (rhow_y - bw * rho_y - bw_y * rho)
                     + birho_y * (rhow - bw * rho);

    // ===== temperature operator and its gradients (REAL) =====
    const double gM2 = gam * (gam - 1.0) * Mach * Mach;

    const SpMatD T = gM2 * (birho * (E - birho * bE * rho) - bu * u - bv * v - bw * w);
    const SpMatD mu = dmu_dT * T;
    
    const SpMatD p = (gam - 1.0) *
            (E - bu * rhou - bv*rhov - bw*rhow 
            + 0.5*(bu*bu+bv*bv+bw*bw)*rho);
    
            // Operands reused in complex strain-rate / flux expressions.
    const SpMatC binvr_c = to_complex(binvr);
    const SpMatC u_c     = to_complex(u);
    const SpMatC v_c     = to_complex(v);
    const SpMatC w_c     = to_complex(w);
    const SpMatC u_x_c   = to_complex(u_x);
    const SpMatC v_y_c   = to_complex(v_y);
    const SpMatC w_y_c   = to_complex(w_y);
    const SpMatC T_c     = to_complex(T);
    const SpMatC mu_c    = to_complex(mu);

    
    // 1i * m
    const cdouble im(0.0, static_cast<double>(m));

    // ===== linearized strain-rate operators (now reading like the MATLAB) =====
    // divu = u_x + v_y + invr*v + 1i*m*invr*w
    const SpMatC divu = u_x_c + v_y_c + binvr_c * v_c + im * binvr_c * w_c;
    
    // Szz = 2*(1i*m*invr*w + invr*v) - 2/3*divu
    const SpMatC Szz = 2.0 * (im * binvr_c * w_c + binvr_c * v_c) - two_thirds * divu;
    
    // Syz = 1i*m*invr*v + w_y - invr*w
    const SpMatC Syz = im * binvr_c * v_c + w_y_c - binvr_c * w_c;

    // ===== assemble F, G, H (5n × Ntot) =====
    const SpMatC Z_row(n, Ntot);  // zero block

    const cdouble inv_Re   (1.0 / Re,  0.0);

    // Background quantities used in F/G/H assembly.
    const SpMatC bmu_c   = to_complex(bmu);
    const SpMatC bu_c    = to_complex(bu);
    const SpMatC bv_c    = to_complex(bv);
    const SpMatC bw_c    = to_complex(bw);
    const SpMatC bSzz_c  = to_complex(bSzz);
    const SpMatC bSyz_c  = to_complex(bSyz);
    const SpMatC p_c     = to_complex(p);
    const SpMatC rhow_c  = to_complex(rhow);
    const SpMatC brhow_c = to_complex(brhow);
    const SpMatC rhov_c  = to_complex(rhov);
    const SpMatC brhov_c = to_complex(brhov);
    

    SpMatC Q = vstack5(
        Z_row,
        Z_row,
        (p_c + brhow_c*w_c + bw_c*rhow_c) - inv_Re*(bmu_c*Szz + bSzz_c*mu_c),
        -brhov_c*w_c - bw_c*rhov_c + inv_Re*(bmu_c*Syz + bSyz_c*mu_c),
        Z_row
    );

    source_terms out;

    auto take_real = [](const auto& A_expr) {
        SpMatC A = A_expr;   // evaluate block to concrete matrix
        SpMatD R(A.rows(), A.cols());
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(A.nonZeros());
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SpMatC::InnerIterator it(A, k); it; ++it) {
                trips.emplace_back(static_cast<int>(it.row()),
                                static_cast<int>(it.col()),
                                it.value().real());
            }
        }
        R.setFromTriplets(trips.begin(), trips.end());
        return R;
    };

    
    out.Q  = Q.middleCols(0 * N, N);
    out.Qx = take_real(Q.middleCols(1 * N, N));
    out.Qy = take_real(Q.middleCols(2 * N, N));
    out.Qz = take_real(Q.middleCols(3 * N, N));

    return out;

}
