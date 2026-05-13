#include "hdg/dg_mat2_Roe.h"
#include "hdg/multidof_ops.h"
#include "hdg/flux_lin_euler.h"
#include "hdg/flux_lin_visc.h"
#include "hdg/debug_print.h"
#include "hdg/source_terms_Q.h"
#include "hdg/vec.h"
#include "hdg/sd.h"
#include "hdg/flux_split.h"
#include "hdg/get_tau_visc.h"

#include <algorithm>
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>
#include <complex>
#include <chrono>

namespace{
using SparseCD  = Eigen::SparseMatrix<std::complex<double>>;
using SparseD  = Eigen::SparseMatrix<double>;
using cdouble  = std::complex<double>;

template <typename Scalar>
Eigen::SparseMatrix<Scalar> sparse_block_t(
    const std::vector<std::vector<Eigen::SparseMatrix<Scalar>>>& blocks)
{
    const int br = blocks.size();
    const int bc = blocks[0].size();
    std::vector<int> row_offsets(br + 1, 0), col_offsets(bc + 1, 0);
    for (int i = 0; i < br; ++i) row_offsets[i + 1] = row_offsets[i] + blocks[i][0].rows();
    for (int j = 0; j < bc; ++j) col_offsets[j + 1] = col_offsets[j] + blocks[0][j].cols();

    std::vector<Eigen::Triplet<Scalar>> trips;
    int nnz = 0;
    for (auto& r : blocks) for (auto& B : r) nnz += B.nonZeros();
    trips.reserve(nnz);

    for (int i = 0; i < br; ++i)
        for (int j = 0; j < bc; ++j) {
            const auto& B = blocks[i][j];
            const int r0 = row_offsets[i], c0 = col_offsets[j];
            for (int k = 0; k < B.outerSize(); ++k)
                for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(B, k); it; ++it)
                    trips.emplace_back(r0 + it.row(), c0 + it.col(), it.value());
        }

    Eigen::SparseMatrix<Scalar> out(row_offsets[br], col_offsets[bc]);
    out.setFromTriplets(trips.begin(), trips.end());
    return out;
}

double face_sgn(int f){
    return 2.0*((f) % 2) - 1.0;
}

static SparseD kron_sparse(const SparseD& A, const SparseD& B) {
    SparseD result = Eigen::kroneckerProduct(A, B);   // <-- explicit type
    result.prune(0.0, 1e-14);
    return result;
}
}

dg dg_mat_2Roe(const SimulationParams& params, const double omega,
    const Msh& msh, const int nvar, const Sem1Op& op1, const Sem2Op& op2,
    const Eigen::MatrixXd& bU, const Eigen::MatrixXd& bUx, const Eigen::MatrixXd& bUy, const Eigen::MatrixXd& bUz,
    const Eigen::MatrixXd& bU_f, const Eigen::MatrixXd& bU_fx, const Eigen::MatrixXd& bU_fy, const Eigen::MatrixXd& bU_fz,
    int m)
{

    using SparseCD  = Eigen::SparseMatrix<std::complex<double>>;
    using SparseD  = Eigen::SparseMatrix<double>;
    using cdouble  = std::complex<double>;

    dg result;

    const double gam = params.gam;
    
    const double c_v = 1.0;

    const int p = msh.elem[0].p;
    const int Ne = msh.elem.size();
    const int Nf = msh.face.size();
    const int npe = p+1;

    const int Me = nvar*(npe*npe);
    const int Mf = nvar*(npe);

    double xmax = msh.elem[0].x.maxCoeff();
    double ymax = msh.elem[0].y.maxCoeff();
    double xmin = msh.elem[0].x.minCoeff();
    double ymin = msh.elem[0].y.minCoeff();

    for (int i1 = 1;i1 < Ne; ++i1){
        xmax = std::max(xmax,msh.elem[i1].x.maxCoeff());
        ymax = std::max(ymax,msh.elem[i1].y.maxCoeff());
        xmin = std::min(xmin,msh.elem[i1].x.minCoeff());
        ymin = std::min(ymin,msh.elem[i1].y.minCoeff());
    }

    MultidofOps ops = multidof_ops(nvar, op1, op2);

    // std::cout << "sem1.M nnz: "    << op1.M.nonZeros()    << "\n";
    // std::cout << "sem1.S nnz: "    << op1.S.nonZeros()    << "\n";
    // std::cout << "sem2.M nnz: "    << op2.M.nonZeros()    << "\n";
    // std::cout << "sem2.S[0] nnz: " << op2.S[0].nonZeros() << "\n";
    // std::cout << "ops.S[0] nnz: "  << ops.S[0].nonZeros()      << "\n";

    SparseD I(nvar,nvar);
    I.setIdentity();

    SparseD Ip(npe,npe);
    Ip.setIdentity();

    SparseD I_nvar(nvar,nvar);
    I_nvar.setIdentity();

    SparseD diag5(nvar,nvar);
    diag5.insert(0,0) = 1.0;
    diag5.insert(4,4) = 1.0;
    diag5.makeCompressed();

    //std::cout <<"test 1" << std::endl;

    SparseD BC_adiabatic = kron_sparse(diag5,Ip);


    double Tw = params.Tw;
    double Mach = params.M;
    double E_bar_o_rho_bar = Tw/(gam*(gam-1)*Mach*Mach);
    double E = E_bar_o_rho_bar;

    SparseD diag5_iso(nvar,nvar);
    diag5_iso.insert(0,0) = 1.0;
    diag5_iso.insert(4,0) = E;

    SparseD BC_iso = kron_sparse(diag5_iso,Ip);
    
    SparseD BC = BC_iso;

    result.A.resize(3 * Ne * Me, 3 * Ne * Me);
    result.B.resize(3 * Ne * Me, Nf * Mf);
    result.C.resize(Nf * Mf,     3 * Ne * Me);
    result.D.resize(Nf * Mf,     Nf * Mf);

    result.Ik.resize(Ne);
    result.QB.resize(Ne);
    result.QC.resize(Ne);
    result.D_factors.resize(Ne);

    const int rows_per_var = p + 1;
    const int ncols = bU_f.cols();

    auto brho_face  = bU_f.middleRows(0 * rows_per_var, rows_per_var);
    auto brhou_face = bU_f.middleRows(1 * rows_per_var, rows_per_var);
    auto brhov_face = bU_f.middleRows(2 * rows_per_var, rows_per_var);
    auto brhow_face = bU_f.middleRows(3 * rows_per_var, rows_per_var);
    auto bE_face    = bU_f.middleRows(4 * rows_per_var, rows_per_var);

    // Invariants
    const SparseCD M_c = ops.M.cast<cdouble>();
    const SparseCD S_0_c = ops.S[0].cast<cdouble>();
    const SparseCD S_1_c = ops.S[1].cast<cdouble>();
    SparseD I2(2, 2); 
    I2.setIdentity();        // 1i * m
    const cdouble im(0.0, static_cast<double>(m));
    // 1i * omega
    const cdouble iomega(0.0, omega);

    double m_d = static_cast<double>(m);
    cdouble m_d_c(m_d,0.0);

    double x_sponge_layer = (xmax - xmin) * 0.1;
    double sponge_start_x = xmax - x_sponge_layer;
    
    // OUTSIDE the i1 loop:
    std::vector<Eigen::Triplet<cdouble>> A_triplets;
    // Rough: each elem_block has ~5 nnz per row × 3Me rows = 15*Me, times Ne elements
    A_triplets.reserve(static_cast<size_t>(15) * Me * Ne);


    for (int i1 = 0; i1 < Ne; ++i1){
        SparseD J = kron_sparse(I,msh.elem[i1].J);
        SparseD Ja11 = kron_sparse(I,msh.elem[i1].Ja11);
        SparseD Ja12 = kron_sparse(I,msh.elem[i1].Ja12);
        SparseD Ja21 = kron_sparse(I,msh.elem[i1].Ja21);
        SparseD Ja22 = kron_sparse(I,msh.elem[i1].Ja22);

        SparseD rV = kron_sparse(I,sd(vec(msh.elem[i1].r)));
        SparseD invrV = kron_sparse(I,sd(vec(msh.elem[i1].invr)));

        //std::cout << "bU (first 10): " << bU.col(i1).head(10) << std::endl;

        // Call flux functions
        auto t_loop = std::chrono::high_resolution_clock::now();
        inviscid_fluxes inv_flux = flux_lin_euler(bU.col(i1),params,nvar);
        auto t_loop_end = std::chrono::high_resolution_clock::now();
        std::cout << "flux_inv took "
             << std::chrono::duration<double>(t_loop_end - t_loop).count()
             << " s\n";
        SparseD F = inv_flux.F;
        SparseD G = inv_flux.G;
        SparseD H = inv_flux.H;
        // std::cout << "F nnz: " << F.nonZeros() << "\n";
        // std::cout << "F explicit zeros: ";
        // int z = 0;
        // for (int k = 0; k < F.outerSize(); ++k)
        //     for (SparseD::InnerIterator it(F, k); it; ++it)
        //         if (it.value() == 0.0) ++z;
        // std::cout << z << "\n";

        t_loop = std::chrono::high_resolution_clock::now();
        viscous_fluxes visc_flux = flux_lin_visc(bU.col(i1),bUx.col(i1),bUy.col(i1),bUz .col(i1),
                                                params,nvar,m,msh.elem[i1].invr);
        t_loop_end = std::chrono::high_resolution_clock::now();
        std::cout << "flux_visc took "
             << std::chrono::duration<double>(t_loop_end - t_loop).count()
             << " s\n";

        SparseCD Fv = visc_flux.F;
        SparseD Fvx = visc_flux.Fx;
        SparseD Fvy = visc_flux.Fy;
        SparseD Fvz = visc_flux.Fz;
        
        SparseCD Gv = visc_flux.G;
        SparseD Gvx = visc_flux.Gx;
        SparseD Gvy = visc_flux.Gy;
        SparseD Gvz = visc_flux.Gz;
        
        SparseCD Hv = visc_flux.H;
        SparseD Hvx = visc_flux.Hx;
        SparseD Hvy = visc_flux.Hy;
        SparseD Hvz = visc_flux.Hz;

        t_loop = std::chrono::high_resolution_clock::now();
        source_terms source_terms = source_terms_Q(bU.col(i1),bUx.col(i1),bUy.col(i1),bUz.col(i1),params, nvar, m, msh.elem[i1].invr);
        t_loop_end = std::chrono::high_resolution_clock::now();
        std::cout << "flux_source took "
             << std::chrono::duration<double>(t_loop_end - t_loop).count()
             << " s\n";
        SparseCD Qr_t = source_terms.Q;
        SparseD Qr_tx = source_terms.Qx;
        SparseD Qr_ty = source_terms.Qy;
        SparseD Qr_tz = source_terms.Qz;

        SparseD Ft = Ja11*F + Ja12*G;
        SparseD Gt = Ja21*F + Ja22*G;

        SparseCD Fvt = Ja11*Fv + Ja12*Gv;
        SparseCD Gvt = Ja21*Fv + Ja22*Gv;
        SparseD Fvxt = Ja11*Fvx + Ja12*Gvx;
        SparseD Gvxt = Ja21*Fvx + Ja22*Gvx;
        SparseD Fvyt = Ja11*Fvy + Ja12*Gvy;
        SparseD Gvyt = Ja21*Fvy + Ja22*Gvy;
        

        // 1i * invr
        SparseCD i_invr = cdouble(0.0, 1.0) * invrV.cast<cdouble>();

        
        // std::cout << "F:   " << F.rows()   << " x " << F.cols()   << "\n";
        // std::cout << "Fv:  " << Fv.rows()  << " x " << Fv.cols()  << "\n";
        // std::cout << "H:   " << H.rows()   << " x " << H.cols()   << "\n";
        // std::cout << "Hv:  " << Hv.rows()  << " x " << Hv.cols()  << "\n";
        // std::cout << "Hvz: " << Hvz.rows() << " x " << Hvz.cols() << "\n";
        // std::cout << "Qr_t:   " << Qr_t.rows()   << " x " << Qr_t.cols()   << "\n";
        // std::cout << "Qr_tz:  " << Qr_tz_c.rows()<< " x " << Qr_tz_c.cols()<< "\n";
        // std::cout << "invrV: " << invrV.rows() << " x " << invrV.cols() << "\n";
        // std::cout << "rV:    " << rV.rows()    << " x " << rV.cols()    << "\n";
        // std::cout << "J:     " << J.rows()     << " x " << J.cols()     << "\n";
        // std::cout << "M:     " << ops.M.rows() << " x " << ops.M.cols() << "\n";
        // std::cout << "S[0]:  " << ops.S[0].rows() << " x " << ops.S[0].cols() << "\n";

        t_loop = std::chrono::high_resolution_clock::now();
        // Cast everything to complex once, reuse from here on.
        // (S_0_c, S_1_c, M_c, im, iomega, m_d_c were hoisted outside the i1 loop.)
        const SparseCD rV_c    = rV.cast<cdouble>();
        const SparseCD invrV_c = invrV.cast<cdouble>();
        const SparseCD J_c     = J.cast<cdouble>();
        const SparseCD H_c     = H.cast<cdouble>();
        const SparseCD Hvx_c   = Hvx.cast<cdouble>();
        const SparseCD Hvy_c   = Hvy.cast<cdouble>();
        const SparseCD Hvz_c   = Hvz.cast<cdouble>();
        const SparseCD Ft_c    = Ft.cast<cdouble>();
        const SparseCD Gt_c    = Gt.cast<cdouble>();
        const SparseCD Fvxt_c  = Fvxt.cast<cdouble>();
        const SparseCD Fvyt_c  = Fvyt.cast<cdouble>();
        const SparseCD Gvxt_c  = Gvxt.cast<cdouble>();
        const SparseCD Gvyt_c  = Gvyt.cast<cdouble>();
        const SparseCD Qr_tx_c = Qr_tx.cast<cdouble>();
        const SparseCD Qr_ty_c = Qr_ty.cast<cdouble>();
        const SparseCD Qr_tz_c = Qr_tz.cast<cdouble>();
        // Qr_t, Fvt, Gvt, Hv, Gv, Fv are already SparseCD — no cast needed.

        // J*M shows up in A, Bx, and By — compute once.
        const SparseCD JM_c = J_c * M_c;

        // S * rV shows up twice in A's correction and once each in Bx, By corrections.
        const SparseCD S0_rV = S_0_c * rV_c;
        const SparseCD S1_rV = S_1_c * rV_c;

        // ---- A ----
        // (-iω·rV + im·(H + Hv) - Qr_t - m²·invrV·Hvz - im·Qr_tz) · J·M
        SparseCD A = (-iomega * rV_c
                    + im * (H_c + Hv)
                    - Qr_t
                    - (m_d_c * m_d_c) * invrV_c * Hvz_c
                    - im * Qr_tz_c) * JM_c;

        // - (S0·rV·(Ft + Fvt) + S1·rV·(Gt + Gvt))
        A -= S0_rV * (Ft_c + Fvt) + S1_rV * (Gt_c + Gvt);

        // ---- Bx, By ----
        SparseCD Bx = (im * Hvx_c - Qr_tx_c) * JM_c;
        SparseCD By = (im * Hvy_c - Qr_ty_c) * JM_c;

        Bx -= S0_rV * Fvxt_c + S1_rV * Gvxt_c;
        By -= S0_rV * Fvyt_c + S1_rV * Gvyt_c;

        t_loop_end = std::chrono::high_resolution_clock::now();
        std::cout << "part 1 took "
             << std::chrono::duration<double>(t_loop_end - t_loop).count()
             << " s\n";

        t_loop = std::chrono::high_resolution_clock::now();
        // Stabilization Matix
        for (int f = 0; f < 12; ++f) {
            int i2 = msh.EtoF(f, i1);
            if (i2 >= 0) {   // or > 0 if your sentinel is 0; check this
                SparseD nn = msh.face[i2].nn;
                SparseD nx_t = face_sgn(f) * msh.face[i2].nx;
                SparseD ny_t = face_sgn(f) * msh.face[i2].ny;
                SparseD nn_kron = kron_sparse(I, msh.face[i2].nn);   // .nn, not .r
                SparseD r_face  = kron_sparse(I, sd(vec(msh.face[i2].r)));

                flux_split_jacobians flux_Jac = flux_split(bU_f.col(i2), nx_t, ny_t, nn, params, nvar);
                SparseD Tau_matrix_visc = get_tau_visc(msh, brho_face, brhou_face, brhov_face,
                                                    brhow_face, bE_face, params, c_v, i2, i1);

                SparseD term_visc_stab = ops.L[f] *
                    ((flux_Jac.Ap - flux_Jac.Am + Tau_matrix_visc * nn_kron) * r_face * ops.T[f]);

                A += term_visc_stab.cast<cdouble>();

                //SparseD r_face = kron_sparse(I,sd(vec(msh.face[i2].r)));
                SparseD nx = face_sgn(f)* kron_sparse(I,msh.face[i2].nx);
                SparseD ny = face_sgn(f)* kron_sparse(I,msh.face[i2].ny);

                SparseD boundaryTerm_x = ops.L[f]*(r_face * (nx*ops.T[f]*Fvx + ny*ops.T[f]*Gvx));
                SparseD boundaryTerm_y = ops.L[f]*(r_face * (nx*ops.T[f]*Fvy + ny*ops.T[f]*Gvy));

                Bx += boundaryTerm_x.cast<cdouble>();
                By += boundaryTerm_y.cast<cdouble>();
            }
        }
        t_loop_end = std::chrono::high_resolution_clock::now();
        std::cout << "face part took "
             << std::chrono::duration<double>(t_loop_end - t_loop).count()
             << " s\n";
        // Sponge layer — once per element, OUTSIDE the face loop
        Eigen::VectorXd x_flat = msh.elem[i1].x.reshaped();
        //Eigen::VectorXd y_flat = msh.elem[i1].y.reshaped();

        // Skip elements entirely outside the sponge zone
        if (msh.elem[i1].x.maxCoeff() < sponge_start_x) {
            // Entirely outside — no sponge contribution
            // Do nothing; just skip the sponge code
        } else {
            Eigen::VectorXd x_flat = msh.elem[i1].x.reshaped();
            Eigen::VectorXd sponge_vals = 10.0 *
                ((x_flat.array() - xmax + x_sponge_layer).max(0.0) / x_sponge_layer).square();

            SparseD sponge_term = J * ops.M * kron_sparse(I_nvar, sd(sponge_vals));
            A += sponge_term.cast<cdouble>();
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        SparseD Cx_real = ops.S[0]*(rV*Ja11) + ops.S[1]*(rV*Ja21);
        SparseD Cy_real = ops.S[0]*(rV*Ja12) + ops.S[1]*(rV*Ja22) + J*ops.M;
        SparseD D_real  = rV * J * ops.M;

        auto t2 = std::chrono::high_resolution_clock::now();

        // Complex casts for block assembly
        SparseCD Cx = Cx_real.cast<cdouble>();
        SparseCD Cy = Cy_real.cast<cdouble>();
        SparseCD D  = D_real.cast<cdouble>();

        // Stash blocks into global result.A (uncondensed form)
        const int j1 = 3 * Me * i1;
        auto stash = [&](const SparseCD& B, int row_off, int col_off) {
            for (int k = 0; k < B.outerSize(); ++k)
                for (SparseCD::InnerIterator it(B, k); it; ++it)
                    A_triplets.emplace_back(j1 + row_off + it.row(),
                                            j1 + col_off + it.col(),
                                            it.value());
        };
        stash(A,  0,    0   );
        stash(Bx, 0,    Me  );
        stash(By, 0,    2*Me);
        stash(Cx, Me,   0   );
        stash(D,  Me,   Me  );
        stash(Cy, 2*Me, 0   );
        stash(D,  2*Me, 2*Me);

        // Build QB, QC for the condensed solve
        SparseCD QB = sparse_block_t<cdouble>({{ Bx, By }});
        SparseCD QC = sparse_block_t<cdouble>({{ Cx }, { Cy }});

        auto t3 = std::chrono::high_resolution_clock::now();

        // Factor D once — store for later use in dg_solve
        auto solver = std::make_unique<Eigen::SparseLU<SparseD>>();
        solver->compute(D_real);
        if (solver->info() != Eigen::Success) {
            std::cerr << "SparseLU factorization failed on element " << i1 << "\n";
        }

        auto t4 = std::chrono::high_resolution_clock::now();

        // Form inv(D) LOCALLY for the iK computation (this is the fast path for matmul)
        Eigen::MatrixXd I_dense = Eigen::MatrixXd::Identity(D_real.rows(), D_real.cols());
        SparseD Dinv_real = solver->solve(I_dense).sparseView(0.0, 1e-14);

        auto t5 = std::chrono::high_resolution_clock::now();

        // Build iK = A - QB * QiD * QC where QiD = kron(I_2, inv(D))
        SparseCD QiD = kron_sparse(I2, Dinv_real).cast<cdouble>();
        SparseCD iK = A - QB * QiD * QC;

        // Store factorization (NOT inv(D) or QiD) for downstream use in dg_solve
        result.QB[i1]  = QB;
        result.QC[i1]  = QC;
        result.D_factors[i1] = std::move(solver);
        result.Ik[i1]  = iK;

        std::cout << "  Cx/Cy/D build:     " << std::chrono::duration<double>(t2-t1).count() << " s\n";
        std::cout << "  cast+stash+QB/QC:  " << std::chrono::duration<double>(t3-t2).count() << " s\n";
        std::cout << "  SparseLU compute:  " << std::chrono::duration<double>(t4-t3).count() << " s\n";
        std::cout << "  form inv(D):       " << std::chrono::duration<double>(t5-t4).count() << " s\n";
    }
    result.A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    std::cout << "===== dg.A complete =====" << std::endl;
}