#include "hdg/dg_mat2_Roe.h"
#include "hdg/multidof_ops.h"
#include "hdg/flux_lin_euler.h"
#include "hdg/debug_print.h"
#include <algorithm>
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>


dg dg_mat_2Roe(const SimulationParams& params, const double omega,
    const Msh& msh, const int nvar, const Sem1Op& op1, const Sem2Op& op2,
    const Eigen::MatrixXd& bU, const Eigen::MatrixXd& bUx, const Eigen::MatrixXd& bUy, const Eigen::MatrixXd& bUz,
    const Eigen::MatrixXd& bU_f, const Eigen::MatrixXd& bU_fx, const Eigen::MatrixXd& bU_fy, const Eigen::MatrixXd& bU_fz,
    int m){
        
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

        Eigen::SparseMatrix<double> I(nvar,nvar);
        I.setIdentity();

        Eigen::SparseMatrix<double> Ip(npe,npe);
        Ip.setIdentity();

        Eigen::SparseMatrix<double> diag5(nvar,nvar);
        diag5.insert(0,0) = 1.0;
        diag5.insert(4,4) = 1.0;
        diag5.makeCompressed();

        Eigen::SparseMatrix<double> BC_adiabatic = Eigen::kroneckerProduct(diag5,Ip);

        double Tw = params.Tw;
        double Mach = params.M;
        double E_bar_o_rho_bar = Tw/(gam*(gam-1)*Mach*Mach);
        double E = E_bar_o_rho_bar;

        Eigen::SparseMatrix<double> diag5_iso(nvar,nvar);
        diag5_iso.insert(0,0) = 1.0;
        diag5_iso.insert(4,0) = E;

        Eigen::SparseMatrix<double> BC_iso = Eigen::kroneckerProduct(diag5_iso,Ip);
        
        Eigen::SparseMatrix<double> BC = BC_iso;

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
            Eigen::SparseMatrix<double> J = Eigen::kroneckerProduct(I,msh.elem[i1].J);
            Eigen::SparseMatrix<double> Ja11 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja11);
            Eigen::SparseMatrix<double> Ja12 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja12);
            Eigen::SparseMatrix<double> Ja21 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja21);
            Eigen::SparseMatrix<double> Ja22 = Eigen::kroneckerProduct(I,msh.elem[i1].Ja22);

            Eigen::SparseMatrix<double> rV = Eigen::kroneckerProduct(I,msh.elem[i1].r);
            Eigen::SparseMatrix<double> invrV = Eigen::kroneckerProduct(I,msh.elem[i1].invr);

            // Call flux functions
            inviscid_fluxes inv_flux = flux_lin_euler(bU.col(i1),params,nvar);
            Eigen::SparseMatrix<double> F = inv_flux.F;
            Eigen::SparseMatrix<double> G = inv_flux.G;
            Eigen::SparseMatrix<double> H = inv_flux.H;

            
            std::cout << "bU (first 10): " << bU.col(i1).head(10) << std::endl;


            // dprint(F);
            // dprint(G);
            // dprint(H);
            
        }

    }