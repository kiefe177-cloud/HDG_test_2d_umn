#include "hdg/flux_lin_visc.h"
#include "hdg/sd.h"
#include "hdg/vec.h"
#include <Eigen/Sparse>
#include <complex>
#include <vector>

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

viscous_fluxes flux_lin_visc(const Eigen::VectorXd& bU,
                             const Eigen::VectorXd& bUx,
                             const Eigen::VectorXd& bUy,
                             const Eigen::VectorXd& bUz,
                             const SimulationParams& params,
                             const int nvar,
                             const int m,
                             const Eigen::MatrixXd& invr_in)
{
    // ===== physical constants =====
    const double Mach = params.M;
    const double gam  = params.gam;
    const double Pr   = params.Pr;
    const double Ts   = params.Ts;
    const double Re   = params.Re_L;

    // ===== sizes =====
    const int N    = static_cast<int>(bU.size());
    const int n    = N / nvar;
    const int Ntot = 4 * N;       // 20 * n

    // ===== narrow selectors and background values (real) =====
    const SpMatD sel_rho  = selector(0, n, N);
    const SpMatD sel_rhou = selector(1, n, N);
    const SpMatD sel_rhov = selector(2, n, N);
    const SpMatD sel_rhow = selector(3, n, N);
    const SpMatD sel_E    = selector(4, n, N);

    const VecD brho_v   = sel_rho  * bU;
    const VecD brhou_v  = sel_rhou * bU;
    const VecD brhov_v  = sel_rhov * bU;
    const VecD brhow_v  = sel_rhow * bU;
    const VecD bE_v     = sel_E    * bU;

    const VecD brho_x_v  = sel_rho  * bUx;
    const VecD brhou_x_v = sel_rhou * bUx;
    const VecD brhov_x_v = sel_rhov * bUx;
    const VecD brhow_x_v = sel_rhow * bUx;
    const VecD bE_x_v    = sel_E    * bUx;

    const VecD brho_y_v  = sel_rho  * bUy;
    const VecD brhou_y_v = sel_rhou * bUy;
    const VecD brhov_y_v = sel_rhov * bUy;
    const VecD brhow_y_v = sel_rhow * bUy;
    const VecD bE_y_v    = sel_E    * bUy;

    const VecD brho_z_v  = sel_rho  * bUz;
    const VecD brhou_z_v = sel_rhou * bUz;
    const VecD brhov_z_v = sel_rhov * bUz;
    const VecD brhow_z_v = sel_rhow * bUz;
    const VecD bE_z_v    = sel_E    * bUz;

    const VecD binvr_v = vec(invr_in);

    // ===== wide REAL selectors (n × Ntot) =====
    const SpMatD rho    = selector(0,  n, Ntot);
    const SpMatD rhou   = selector(1,  n, Ntot);
    const SpMatD rhov   = selector(2,  n, Ntot);
    const SpMatD rhow   = selector(3,  n, Ntot);
    const SpMatD E      = selector(4,  n, Ntot);
    const SpMatD rho_x  = selector(5,  n, Ntot);
    const SpMatD rhou_x = selector(6,  n, Ntot);
    const SpMatD rhov_x = selector(7,  n, Ntot);
    const SpMatD rhow_x = selector(8,  n, Ntot);
    const SpMatD E_x    = selector(9,  n, Ntot);
    const SpMatD rho_y  = selector(10, n, Ntot);
    const SpMatD rhou_y = selector(11, n, Ntot);
    const SpMatD rhov_y = selector(12, n, Ntot);
    const SpMatD rhow_y = selector(13, n, Ntot);
    const SpMatD E_y    = selector(14, n, Ntot);
    // rho_z..E_z are unused since T_z is not assembled.

    // ===== background scalar fields (real vectors) =====
    const VecD brho_sq_v = brho_v.array().square().matrix();

    const VecD birho_v   = brho_v.cwiseInverse();
    const VecD birho_x_v = -brho_x_v.cwiseQuotient(brho_sq_v);
    const VecD birho_y_v = -brho_y_v.cwiseQuotient(brho_sq_v);
    const VecD birho_z_v = -brho_z_v.cwiseQuotient(brho_sq_v);

    const VecD bu_v = brhou_v.cwiseQuotient(brho_v);
    const VecD bv_v = brhov_v.cwiseQuotient(brho_v);
    const VecD bw_v = brhow_v.cwiseQuotient(brho_v);

    auto vel_grad = [&](const VecD& brhoq_d, const VecD& bq, const VecD& brho_d) {
        return ((brhoq_d - bq.cwiseProduct(brho_d)).cwiseQuotient(brho_v)).eval();
    };

    const VecD bu_x_v = vel_grad(brhou_x_v, bu_v, brho_x_v);
    const VecD bu_y_v = vel_grad(brhou_y_v, bu_v, brho_y_v);
    const VecD bu_z_v = vel_grad(brhou_z_v, bu_v, brho_z_v);
    const VecD bv_x_v = vel_grad(brhov_x_v, bv_v, brho_x_v);
    const VecD bv_y_v = vel_grad(brhov_y_v, bv_v, brho_y_v);
    const VecD bv_z_v = vel_grad(brhov_z_v, bv_v, brho_z_v);
    const VecD bw_x_v = vel_grad(brhow_x_v, bw_v, brho_x_v);
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
    const SpMatD bu_x    = sd(bu_x_v);
    const SpMatD bu_y    = sd(bu_y_v);
    const SpMatD bu_z    = sd(bu_z_v);
    const SpMatD bv_x    = sd(bv_x_v);
    const SpMatD bv_y    = sd(bv_y_v);
    const SpMatD bv_z    = sd(bv_z_v);
    const SpMatD bw_x    = sd(bw_x_v);
    const SpMatD bw_y    = sd(bw_y_v);
    const SpMatD bw_z    = sd(bw_z_v);
    const SpMatD binvr   = sd(binvr_v);
    const SpMatD bdivu   = sd(bdivu_v);
    const SpMatD bE      = sd(bE_v);
    const SpMatD bE_x    = sd(bE_x_v);
    const SpMatD bE_y    = sd(bE_y_v);
    const SpMatD dmu_dT  = sd(dmu_dT_v);

    // ===== background strain rates (REAL — m doesn't enter here) =====
    const double two_thirds = 2.0 / 3.0;

    const SpMatD bSxx = 2.0 * bu_x - two_thirds * bdivu;
    const SpMatD bSyy = 2.0 * bv_y - two_thirds * bdivu;
    const SpMatD bSzz = 2.0 * (bw_z * binvr + bv * binvr) - two_thirds * bdivu;
    const SpMatD bSxy = bv_x + bu_y;
    const SpMatD bSyx = bSxy;
    const SpMatD bSxz = bw_x + bu_z * binvr;
    const SpMatD bSzx = bSxz;
    const SpMatD bSyz = bv_z * binvr + bw_y - bw * binvr;
    const SpMatD bSzy = bSyz;

    // ===== linearized REAL operators (no m) =====
    const SpMatD u = birho * (rhou - bu * rho);
    const SpMatD v = birho * (rhov - bv * rho);
    const SpMatD w = birho * (rhow - bw * rho);

    const SpMatD u_x = birho * (rhou_x - bu * rho_x - bu_x * rho)
                     + birho_x * (rhou - bu * rho);
    const SpMatD u_y = birho * (rhou_y - bu * rho_y - bu_y * rho)
                     + birho_y * (rhou - bu * rho);
    const SpMatD v_x = birho * (rhov_x - bv * rho_x - bv_x * rho)
                     + birho_x * (rhov - bv * rho);
    const SpMatD v_y = birho * (rhov_y - bv * rho_y - bv_y * rho)
                     + birho_y * (rhov - bv * rho);
    const SpMatD w_x = birho * (rhow_x - bw * rho_x - bw_x * rho)
                     + birho_x * (rhow - bw * rho);
    const SpMatD w_y = birho * (rhow_y - bw * rho_y - bw_y * rho)
                     + birho_y * (rhow - bw * rho);

    // ===== temperature operator and its gradients (REAL) =====
    const double gM2 = gam * (gam - 1.0) * Mach * Mach;

    const SpMatD T = gM2 * (birho * (E - birho * bE * rho) - bu * u - bv * v - bw * w);
    const SpMatD mu = dmu_dT * T;

    const SpMatD T_x = gM2 * (
          birho * (E_x - birho * bE * rho_x - birho_x * bE * rho - birho * bE_x * rho)
        + birho_x * (E - birho * bE * rho)
        - bu * u_x - bu_x * u - bv * v_x - bv_x * v - bw * w_x - bw_x * w
    );

    const SpMatD T_y = gM2 * (
          birho * (E_y - birho * bE * rho_y - birho_y * bE * rho - birho * bE_y * rho)
        + birho_y * (E - birho * bE * rho)
        - bu * u_y - bu_y * u - bv * v_y - bv_y * v - bw * w_y - bw_y * w
    );

    // =========================================================
    // Promote to COMPLEX. The 1i*m terms enter from here.
    // =========================================================

    // Operands reused in complex strain-rate / flux expressions.
    const SpMatC binvr_c =  binvr.cast<cdouble>();
    const SpMatC u_c     =  u.cast<cdouble>();
    const SpMatC v_c     =  v.cast<cdouble>();
    const SpMatC w_c     =  w.cast<cdouble>();
    const SpMatC u_x_c   =  u_x.cast<cdouble>();
    const SpMatC u_y_c   =  u_y.cast<cdouble>();
    const SpMatC v_x_c   =  v_x.cast<cdouble>();
    const SpMatC v_y_c   =  v_y.cast<cdouble>();
    const SpMatC w_x_c   =  w_x.cast<cdouble>();
    const SpMatC w_y_c   =  w_y.cast<cdouble>();
    const SpMatC T_c     =  T.cast<cdouble>();
    const SpMatC T_x_c   =  T_x.cast<cdouble>();
    const SpMatC T_y_c   =  T_y.cast<cdouble>();
    const SpMatC mu_c    =  mu.cast<cdouble>();

    // Background quantities used in F/G/H assembly.
    const SpMatC bmu_c   =  bmu.cast<cdouble>();
    const SpMatC bu_c    =  bu.cast<cdouble>();
    const SpMatC bv_c    =  bv.cast<cdouble>();
    const SpMatC bw_c    =  bw.cast<cdouble>();
    const SpMatC bSxx_c  =  bSxx.cast<cdouble>();
    const SpMatC bSyy_c  =  bSyy.cast<cdouble>();
    const SpMatC bSzz_c  =  bSzz.cast<cdouble>();
    const SpMatC bSxy_c  =  bSxy.cast<cdouble>();
    const SpMatC bSyx_c  = bSxy_c;
    const SpMatC bSxz_c  =  bSxz.cast<cdouble>();
    const SpMatC bSzx_c  = bSxz_c;
    const SpMatC bSyz_c  =  bSyz.cast<cdouble>();
    const SpMatC bSzy_c  = bSyz_c;

    // 1i * m
    const cdouble im(0.0, static_cast<double>(m));

    // ===== linearized strain-rate operators (now reading like the MATLAB) =====
    // divu = u_x + v_y + invr*v + 1i*m*invr*w
    const SpMatC divu = u_x_c + v_y_c + binvr_c * v_c + im * binvr_c * w_c;

    // Sxx = 2*u_x - 2/3*divu
    const SpMatC Sxx = 2.0 * u_x_c - two_thirds * divu;
    const SpMatC Syy = 2.0 * v_y_c - two_thirds * divu;

    // Szz = 2*(1i*m*invr*w + invr*v) - 2/3*divu
    const SpMatC Szz = 2.0 * (im * binvr_c * w_c + binvr_c * v_c) - two_thirds * divu;

    const SpMatC Sxy = v_x_c + u_y_c;
    const SpMatC Syx = Sxy;

    // Sxz = w_x + 1i*m*invr*u
    const SpMatC Sxz = w_x_c + im * binvr_c * u_c;
    const SpMatC Szx = Sxz;

    // Syz = 1i*m*invr*v + w_y - invr*w
    const SpMatC Syz = im * binvr_c * v_c + w_y_c - binvr_c * w_c;
    const SpMatC Szy = Syz;

    // ===== assemble F, G, H (5n × Ntot) =====
    const SpMatC Z_row(n, Ntot);  // zero block

    const cdouble inv_Re   (1.0 / Re,  0.0);
    const cdouble heat_coef(1.0 / ((gam - 1.0) * Mach * Mach * Re * Pr), 0.0);

    SpMatC F = vstack5(
        Z_row,
        -inv_Re * (bmu_c * Sxx + bSxx_c * mu_c),
        -inv_Re * (bmu_c * Sxy + bSxy_c * mu_c),
        -inv_Re * (bmu_c * Sxz + bSxz_c * mu_c),
        -(inv_Re * (  bmu_c*bu_c*Sxx + bmu_c*bSxx_c*u_c + bu_c*bSxx_c*mu_c
                    + bmu_c*bv_c*Sxy + bmu_c*bSxy_c*v_c + bv_c*bSxy_c*mu_c
                    + bmu_c*bw_c*Sxz + bmu_c*bSxz_c*w_c + bw_c*bSxz_c*mu_c)
          + heat_coef * T_x_c)
    );

    SpMatC G = vstack5(
        Z_row,
        -inv_Re * (bmu_c * Syx + bSyx_c * mu_c),
        -inv_Re * (bmu_c * Syy + bSyy_c * mu_c),
        -inv_Re * (bmu_c * Syz + bSyz_c * mu_c),
        -(inv_Re * (  bmu_c*bu_c*Syx + bmu_c*bSyx_c*u_c + bu_c*bSyx_c*mu_c
                    + bmu_c*bv_c*Syy + bmu_c*bSyy_c*v_c + bv_c*bSyy_c*mu_c
                    + bmu_c*bw_c*Syz + bmu_c*bSyz_c*w_c + bw_c*bSyz_c*mu_c)
          + heat_coef * T_y_c)
    );

    SpMatC H = vstack5(
        Z_row,
        -inv_Re * (bmu_c * Szx + bSzx_c * mu_c),
        -inv_Re * (bmu_c * Szy + bSzy_c * mu_c),
        -inv_Re * (bmu_c * Szz + bSzz_c * mu_c),
        -(inv_Re * (  bmu_c*bu_c*Szx + bmu_c*bSzx_c*u_c + bu_c*bSzx_c*mu_c
                    + bmu_c*bv_c*Szy + bmu_c*bSzy_c*v_c + bv_c*bSzy_c*mu_c
                    + bmu_c*bw_c*Szz + bmu_c*bSzz_c*w_c + bw_c*bSzz_c*mu_c)
          + heat_coef * im * binvr_c * T_c)
    );

    // ===== slice columns by derivative block =====
    viscous_fluxes out;

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

    out.F  = F.middleCols(0 * N, N);
    out.Fx = take_real(F.middleCols(1 * N, N));
    out.Fy = take_real(F.middleCols(2 * N, N));
    out.Fz = take_real(F.middleCols(3 * N, N));

    out.G  = G.middleCols(0 * N, N);
    out.Gx = take_real(G.middleCols(1 * N, N));
    out.Gy = take_real(G.middleCols(2 * N, N));
    out.Gz = take_real(G.middleCols(3 * N, N));

    out.H  = H.middleCols(0 * N, N);
    out.Hx = take_real(H.middleCols(1 * N, N));
    out.Hy = take_real(H.middleCols(2 * N, N));
    out.Hz = take_real(H.middleCols(3 * N, N));

    out.F.prune(cdouble(0.0, 0.0),1e-14);
    out.Fx.prune(0.0, 1e-14);
    out.Fy.prune(0.0, 1e-14);
    out.Fz.prune(0.0, 1e-14);

    out.G.prune(cdouble(0.0, 0.0),1e-14);
    out.Gx.prune(0.0, 1e-14);
    out.Gy.prune(0.0, 1e-14);
    out.Gz.prune(0.0, 1e-14);

    out.H.prune(cdouble(0.0, 0.0),1e-14);
    out.Hx.prune(0.0, 1e-14);
    out.Hy.prune(0.0, 1e-14);
    out.Hz.prune(0.0, 1e-14);

    return out;
}