#include "hdg/dg_mat2_Roe.h"
#include "hdg/multidof_ops.h"
#include "hdg/flux_lin_euler.h"
#include "hdg/flux_lin_visc.h"
#include "hdg/debug_print.h"
#include "hdg/source_terms_Q.h"
#include "hdg/vec.h"
#include "hdg/sd.h"
#include <algorithm>
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>
#include <complex>

namespace{
using SparseCD  = Eigen::SparseMatrix<std::complex<double>>;
using SparseD  = Eigen::SparseMatrix<double>;
using cdouble  = std::complex<double>;

double face_sgn(int f){
    return 2.0*((f + 1) % 2) - 1.0;
}
// Promote a real sparse matrix to complex (zero imaginary part).
SparseCD to_complex(const SparseD& A) {
    SparseCD B(A.rows(), A.cols());
    std::vector<Eigen::Triplet<cdouble>> trips;
    trips.reserve(A.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseD::InnerIterator it(A, k); it; ++it) {
            trips.emplace_back(static_cast<int>(it.row()),
                               static_cast<int>(it.col()),
                               cdouble(it.value(), 0.0));
        }
    }
    B.setFromTriplets(trips.begin(), trips.end());
    return B;
}
}

dg dg_mat_2Roe(const SimulationParams& params, const double omega,
    const Msh& msh, const int nvar, const Sem1Op& op1, const Sem2Op& op2,
    const Eigen::MatrixXd& bU, const Eigen::MatrixXd& bUx, const Eigen::MatrixXd& bUy, const Eigen::MatrixXd& bUz,
    const Eigen::MatrixXd& bU_f, const Eigen::MatrixXd& bU_fx, const Eigen::MatrixXd& bU_fy, const Eigen::MatrixXd& bU_fz,
    int m){

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

        SparseD I(nvar,nvar);
        I.setIdentity();

        SparseD Ip(npe,npe);
        Ip.setIdentity();

        SparseD diag5(nvar,nvar);
        diag5.insert(0,0) = 1.0;
        diag5.insert(4,4) = 1.0;
        diag5.makeCompressed();

        SparseD BC_adiabatic = Eigen::kroneckerProduct(diag5,Ip);

        double Tw = params.Tw;
        double Mach = params.M;
        double E_bar_o_rho_bar = Tw/(gam*(gam-1)*Mach*Mach);
        double E = E_bar_o_rho_bar;

        SparseD diag5_iso(nvar,nvar);
        diag5_iso.insert(0,0) = 1.0;
        diag5_iso.insert(4,0) = E;

        SparseD BC_iso = Eigen::kroneckerProduct(diag5_iso,Ip);
        
        SparseD BC = BC_iso;

        result.A.resize(3 * Ne * Me, 3 * Ne * Me);
        result.B.resize(3 * Ne * Me, Nf * Mf);
        result.C.resize(Nf * Mf,     3 * Ne * Me);
        result.D.resize(Nf * Mf,     Nf * Mf);

        result.Ik.resize(Ne);
        result.QB.resize(Ne);
        result.QC.resize(Ne);
        result.QiD.resize(Ne);

        const int rows_per_var = p + 1;
        const int ncols = bU_f.cols();

        auto brho_face  = bU_f.middleRows(0 * rows_per_var, rows_per_var);
        auto brhou_face = bU_f.middleRows(1 * rows_per_var, rows_per_var);
        auto brhov_face = bU_f.middleRows(2 * rows_per_var, rows_per_var);
        auto brho2_face = bU_f.middleRows(3 * rows_per_var, rows_per_var);
        auto bE_face    = bU_f.middleRows(4 * rows_per_var, rows_per_var);

        for (int i1 = 0; i1 < Ne; ++i1){
            SparseD J = Eigen::kroneckerProduct(I,msh.elem[i1].J);
            SparseD Ja11 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja11);
            SparseD Ja12 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja12);
            SparseD Ja21 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja21);
            SparseD Ja22 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja22);

            SparseD rV = Eigen::kroneckerProduct(I,sd(vec(msh.elem[i1].r)));
            SparseD invrV = Eigen::kroneckerProduct(I,sd(vec(msh.elem[i1].invr)));

            //std::cout << "bU (first 10): " << bU.col(i1).head(10) << std::endl;

            // Call flux functions
            inviscid_fluxes inv_flux = flux_lin_euler(bU.col(i1),params,nvar);
            SparseD F = inv_flux.F;
            SparseD G = inv_flux.G;
            SparseD H = inv_flux.H;

            viscous_fluxes visc_flux = flux_lin_visc(bU.col(i1),bUx.col(i1),bUy.col(i1),bUz .col(i1),
                                                    params,nvar,m,msh.elem[i1].invr);
            
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

            source_terms source_terms = source_terms_Q(bU.col(i1),bUx.col(i1),bUy.col(i1),bUz.col(i1),params, nvar, m, msh.elem[i1].invr);

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
            
            // 1i * m
            const cdouble im(0.0, static_cast<double>(m));
            // 1i * omega
            const cdouble iomega(0.0, omega);
            // 1i * invr
            SparseCD i_invr = cdouble(0.0, 1.0) * to_complex(invrV);

            
            SparseCD invrV_c = to_complex(invrV);
            SparseCD Ft_c = to_complex(Ft);
            SparseCD Gt_c = to_complex(Gt);
            SparseCD rV_c = to_complex(rV);
            SparseCD Fvxt_c = to_complex(Fvxt);
            SparseCD Fvyt_c = to_complex(Fvyt);
            SparseCD Gvxt_c = to_complex(Gvxt);
            SparseCD Gvyt_c = to_complex(Gvyt);
            SparseCD H_c = to_complex(H);
            SparseCD Hvx_c = to_complex(Hvx);
            SparseCD Hvy_c = to_complex(Hvy);
            SparseCD Hvz_c = to_complex(Hvz);
            SparseCD Qr_tx_c = to_complex(Qr_tx);
            SparseCD Qr_ty_c = to_complex(Qr_ty);
            SparseCD Qr_tz_c = to_complex(Qr_tz);
            SparseCD J_c = to_complex(J);
            SparseCD M_c = to_complex(ops.M);
            SparseCD S_0_c = to_complex(ops.S[0]);
            SparseCD S_1_c = to_complex(ops.S[1]);

            double m_d = static_cast<double>(m);
            cdouble m_d_c(m_d,0.0);
            
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

            SparseCD A = (-iomega*rV_c + im*(H_c+Hv) - Qr_t - m_d_c*m_d_c*invrV_c*Hvz_c - im*Qr_tz_c)*J_c*M_c;
            A -= (S_0_c*rV_c*(Ft_c + Fvt) + S_1_c*rV_c*(Gt_c + Gvt));

            SparseCD Bx = (im*Hvx_c - Qr_tx_c)*J_c*M_c;
            SparseCD By = (im*Hvy_c - Qr_ty_c)*J_c*M_c;

            Bx += (-(S_0_c*(rV_c*Fvxt_c) + S_1_c*(rV_c*Gvxt_c)));
            By += (-(S_0_c*(rV_c*Fvyt_c) + S_1_c*(rV_c*Gvyt_c)));

            // Stabilization Matix
            for (int f = 0; f < 12; ++f)
            {
                int i2 = msh.EtoF(f,i1);
                if (i2 > 0)
                {
                    Eigen::MatrixXd nn = msh.face[i2].nn;
                    SparseD nx_t = face_sgn(f)*msh.face[i2].nx;
                    SparseD ny_t = face_sgn(f)*msh.face[i2].ny;
                    SparseD nn_kron = Eigen::kroneckerProduct(I,msh.face[i2].r);
                    SparseD r_face = Eigen::kroneckerProduct(I, sd(vec(msh.face[i2].r)));

                    

                }
                
            }
            

        }

    }