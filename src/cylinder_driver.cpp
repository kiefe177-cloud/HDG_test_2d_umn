/*
0. Scenario
*/

/*
1. Grid Parameters

ndgrid

generate tfi

elliptic solve
*/

// TODO 
// Implement reader

#include <Eigen/Dense>
#include <iostream>
#include <highfive/H5File.hpp>
#include <string>
#include <cmath>

#include "hdg/all.h" // Main driver files
#include "hdg/mesh_utils.h"
#include "hdg/io_utils.h" // For save_csv
#include "hdg/save_grid_diagnostics.h" // For save_grid_diagnostics
#include "parameters/MalikSpallParams.h"

using namespace Eigen;

int main(){

    SimulationParams params = get_malik_spall_params();

    double gam = params.gam;
    int nvar = 5;
    int p = 12;
    int Ne_x = 8;

    int m = 0;

    double F_final = 1.17e-4;

    double omega = F_final * params.Re_L;
    double omega_dim = omega * params.ue / params.L;
    double F_dim = omega_dim / (2* M_PI);

    // Test print statements
    std::cout << "Re_L: " << params.Re_L << std::endl;
    std::cout << "Omega_dim (e5): " << omega_dim/1e5 << std::endl;
    std::cout << "F_dim (kHz): " << F_dim/1000 << std::endl;
    
    double R_inner = 50.0;

    double ymin = 0.0 + R_inner;
    double ymax = 100 + ymin;
    double xmin = 0.0;
    double xmax = xmin + 24*2*M_PI/omega;
    int n = 400;

    // ndgrid here
    VectorXd x_vec = VectorXd::LinSpaced(n, xmin, xmax);
    VectorXd y_vec = VectorXd::LinSpaced(n, ymin, ymax);
    
    auto [X, Y] = ndgrid(x_vec, y_vec);

    X = X.transpose().eval(); // .eval(): Forces to be evaluated first
    Y = Y.transpose().eval();

    TFI_Mesh tfi_grid = generate_tfi_mesh(X,Y, Ne_x);

    // Access results like:
    std::cout << "TFI grid: " << tfi_grid.X_tfi.rows() << " x " << tfi_grid.X_tfi.cols() << std::endl;
    std::cout << "Test point: (" << tfi_grid.x_p << ", " << tfi_grid.y_p << ")" << std::endl;

    // Four Elliptic Options: 
    //                      solve (TTM),
    //                      ortho (Wall Ortho only),
    //                      full Ortho,
    //                      Steger-Sorenson

    EllipticMethod method = EllipticMethod::Ortho;

    int n_elliptic_iter = 100000;
    double sor_omega = 1.5;
    double convergence_tol = 1e-12;

    OrthoOpts ortho_opts;
    ortho_opts.ortho_tol = convergence_tol;
    ortho_opts.march_tol = convergence_tol;

    EllipticResult result = tfi_elliptic_orthogonal(
        tfi_grid.X_tfi, tfi_grid.Y_tfi,
        tfi_grid.phi_comp_vec, tfi_grid.psi_comp_vec,
        tfi_grid.n_grid_pts, n_elliptic_iter, sor_omega, convergence_tol,
        ortho_opts);

    // Sample boundary curves at high resolution for plotting.
    const int n_fine = 5000;
    Eigen::VectorXd phi_fine = Eigen::VectorXd::LinSpaced(n_fine, -1.0, 1.0);
    Eigen::VectorXd psi_fine = Eigen::VectorXd::LinSpaced(n_fine, -1.0, 1.0);

    Eigen::VectorXd x_left(n_fine),  y_left(n_fine);
    Eigen::VectorXd x_right(n_fine), y_right(n_fine);
    Eigen::VectorXd x_bot(n_fine),   y_bot(n_fine);
    Eigen::VectorXd x_top(n_fine),   y_top(n_fine);

    for (int i = 0; i < n_fine; ++i) {
        std::tie(x_left(i),  y_left(i))  = tfi_grid.f.left_boundary  (psi_fine(i));
        std::tie(x_right(i), y_right(i)) = tfi_grid.f.right_boundary (psi_fine(i));
        std::tie(x_bot(i),   y_bot(i))   = tfi_grid.f.bottom_boundary(phi_fine(i));
        std::tie(x_top(i),   y_top(i))   = tfi_grid.f.top_boundary   (phi_fine(i));
    }

    // Plot_comp_tfi_grid python script will plot this
    save_grid_diagnostics(
        std::string(OUTPUT_DIR) + "/cyl_grid.h5",
        tfi_grid.Phi_tfi, tfi_grid.Psi_tfi,
        result.X_ellip_grid, result.Y_ellip_grid,
        x_left, y_left, x_right, y_right,
        x_bot, y_bot, x_top, y_top,
        tfi_grid.x_p, tfi_grid.y_p, tfi_grid.phi_val, tfi_grid.psi_val,
        result.diagnost.skewness, result.diagnost.jacobian);


    PhysMesh phys_msh;
    phys_msh.geom = comp_space_metrics(
        result.X_ellip_grid, result.Y_ellip_grid,
        p, tfi_grid.phi_vec, tfi_grid.psi_vec, tfi_grid.f);
    phys_msh.base_grid = base_grid_metrics(
        result.X_ellip_grid, result.Y_ellip_grid,
        tfi_grid.phi_comp_vec, tfi_grid.psi_comp_vec);
    
    
    // msh_test_grid

    Msh msh = msh_test_grad(p, phys_msh,
                        std::string(OUTPUT_DIR) + "/amr_leaves.csv");

    
    // std::cout << "msh.elem size: " << msh.elem.size() << " || msh.face size: " << msh.face.size() << std::endl;
    
    // std::cout << "msh.FtoE rows: " << msh.FtoE.rows() << "|| msh.FtoE cols: " << msh.FtoE.cols() << std::endl;

    // std::cout << "msh.EtoF rows: " << msh.EtoF.rows() << "  || msh.EtoF cols: " << msh.EtoF.cols() << std::endl;

    // std::cout << "msh.elem.J rows: " << msh.elem[0].J.rows() << " || msh.elem.J cols: " << msh.elem[0].J.cols() << std::endl;
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // std::cout << "msh.elem.Ja11 rows: " << msh.elem[0].Ja11.rows() << " || msh.elem.Ja11 cols: " << msh.elem[0].Ja11.cols() << std::endl;
    
    // std::cout << "msh.elem.Ja12 rows: " << msh.elem[0].Ja12.rows() << " || msh.elem.Ja12 cols: " << msh.elem[0].Ja12.cols() << std::endl;
    
    // std::cout << "msh.elem.Ja21 rows: " << msh.elem[0].Ja21.rows() << " || msh.elem.Ja21 cols: " << msh.elem[0].Ja21.cols() << std::endl;
    
    // std::cout << "msh.elem.Ja22 rows: " << msh.elem[0].Ja22.rows() << " || msh.elem.Ja22 cols: " << msh.elem[0].Ja22.cols() << std::endl;

    // std::cout << "msh.elem.w1d size: "  << msh.elem[0].w1d.size()  << std::endl;
    
    // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // std::cout << "msh.face.x size: " << msh.face[0].x.size() << " || msh.face.Ja11 cols: " << msh.face[0].y.size() << std::endl;
    
    // std::cout << "msh.face.nx rows: " << msh.face[0].nx.rows() << " || msh.face.nx cols: " << msh.face[0].nx.cols() << std::endl;
    
    // std::cout << "msh.face.ny rows: " << msh.face[0].ny.rows() << " || msh.face.ny cols: " << msh.face[0].ny.cols() << std::endl;
    
    // std::cout << "msh.face.nn rows: " << msh.face[0].nn.rows() << " || msh.face.nn cols: " << msh.face[0].nn.cols() << std::endl;

    // std::cout << "msh.face.w1d size: " << msh.face[0].w1d.size() << " || msh.face.Jf size: " << msh.face[0].Jf.size() << std::endl;

    // for (int i = 0; i < msh.face[0].w1d.size(); ++i)
    // {
    //     std::cout << "msh.face.Jf[" << i << "] value: " << msh.face[9].Jf(i) << std::endl;
    // }
    
    Sem1Op op1 = sem1(p);
    Sem2Op op2 = sem2(p);

    const int Ne = msh.elem.size();
    const int Nf = msh.FtoE.cols();
    
    std::cout << "Ne = " << Ne << std::endl;
    std::cout << "Nf = " << Nf << std::endl;

    const int Me = nvar*(p+1)*(p+1);
    const int Mf = nvar*(p+1);
    const int npe    = (p + 1) * (p + 1);
    const int stride = std::max(1, (Ne + 7) / 8);

    int Ny = 350;
    double x0 = 0.0;

    // baseflow_copy = baseflow_flatplate_1d(params, Ny, i1)

    double w_tr = 2.0;
    double n0 = 0.0;
    double w_n = 0.0;
    std::cout <<"============ Creating Element Baseflow ============"<< std::endl;
    
    Baseflow baseflow_copy = baseflow_flatplate_1d(params, Ny, 1);

    Eigen::VectorXd y_bf           = baseflow_copy.x[1];
    Eigen::VectorXd rho_bf         = baseflow_copy.rho;
    Eigen::VectorXd rhoE_bf        = baseflow_copy.E;
    Eigen::VectorXd rhou_stream_bf = baseflow_copy.rhou[0];
    Eigen::VectorXd rhou_normal_bf = baseflow_copy.rhou[1];

    // Primitives
    Eigen::VectorXd ut_bf = (rhou_stream_bf.array() / rho_bf.array()).matrix();
    Eigen::VectorXd un_bf = (rhou_normal_bf.array() / rho_bf.array()).matrix();

    Eigen::VectorXd p_bf = ((gam - 1.0)
        * ( rhoE_bf.array()
        - 0.5 * rho_bf.array() * (ut_bf.array().square() + un_bf.array().square())
        )
        ).matrix();


    Eigen::MatrixXd brho    = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd brhou   = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd brhov   = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd brhow   = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd bE      = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd bu      = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd bv      = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd bw      = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);
    Eigen::MatrixXd bp      = Eigen::MatrixXd::Zero((p+1)*(p+1),Ne);

    for (int i1 = 0; i1 < Ne; ++i1)
    {
        Eigen::MatrixXd xN = msh.elem[i1].x;
        Eigen::MatrixXd yN = msh.elem[i1].y;
        Eigen::MatrixXd y_flat = yN.array() - R_inner;   // (p+1) x (p+1)

        // Flatten to 1D (column-major, matching MATLAB's `(:)`)
        const int nn = y_flat.size();   // (p+1)*(p+1)
        Eigen::Map<const Eigen::VectorXd> y_flat_1d(y_flat.data(), nn);

        // std::cout << "y_bf:    " << y_bf.rows()    << " x " << y_bf.cols()    << "\n";
        // std::cout << "rho_bf:  " << rho_bf.rows()  << " x " << rho_bf.cols()  << "\n";
        // std::cout << "y_flat:  " << y_flat.rows()  << " x " << y_flat.cols()  << "\n";

        // Interpolate
        Eigen::VectorXd rho_flat_1d = interp1(y_bf, rho_bf, y_flat_1d,
                                            InterpMethod::Spline, kExtrapNatural);
        Eigen::VectorXd ut_flat_1d = interp1(y_bf, ut_bf, y_flat_1d,
                                            InterpMethod::Spline, kExtrapNatural);
        Eigen::VectorXd un_flat_1d = interp1(y_bf, un_bf, y_flat_1d,
                                            InterpMethod::Spline, kExtrapNatural);                                     
        Eigen::VectorXd p_flat_1d = interp1(y_bf, p_bf, y_flat_1d,
                                            InterpMethod::Spline, kExtrapNatural);

        // Reshape back to (p+1) x (p+1)
        Eigen::Map<Eigen::MatrixXd> rho_flat(rho_flat_1d.data(), p+1, p+1);
        Eigen::Map<Eigen::MatrixXd> ut_flat(ut_flat_1d.data(), p+1, p+1);
        Eigen::Map<Eigen::MatrixXd> un_flat(un_flat_1d.data(), p+1, p+1);
        Eigen::Map<Eigen::MatrixXd> p_flat(p_flat_1d.data(), p+1, p+1);

        Eigen::MatrixXd u_flat  = ut_flat;
        Eigen::MatrixXd v_flat  = un_flat;

        Eigen::MatrixXd r       = (xN.array().square() + yN.array().square()).sqrt().matrix();
        Eigen::MatrixXd theta   = yN.array().binaryExpr(xN.array(),[](double y, double x){
            return std::atan2(y,x);
        });
        
        Eigen::MatrixXd n_ann   = r.array() - R_inner;

        Eigen::Map<const Eigen::VectorXd> n_ann_1d(n_ann.data(), nn);


        //std::cout << r.array() << std::endl;
        // Interpolate
        Eigen::VectorXd rho_ann_1d = interp1(y_bf, rho_bf, n_ann_1d,
                                            InterpMethod::Spline, kExtrapNatural);
        Eigen::VectorXd ut_ann_1d = interp1(y_bf, ut_bf, n_ann_1d,
                                            InterpMethod::Spline, kExtrapNatural);
        Eigen::VectorXd un_ann_1d = interp1(y_bf, un_bf, n_ann_1d,
                                            InterpMethod::Spline, kExtrapNatural);                                     
        Eigen::VectorXd p_ann_1d = interp1(y_bf, p_bf, n_ann_1d,
                                            InterpMethod::Spline, kExtrapNatural);

        
        // Reshape back to (p+1) x (p+1)
        Eigen::Map<Eigen::MatrixXd> rho_ann(rho_ann_1d.data(), p+1, p+1);
        Eigen::Map<Eigen::MatrixXd> ut_ann(ut_ann_1d.data(), p+1, p+1);
        Eigen::Map<Eigen::MatrixXd> un_ann(un_ann_1d.data(), p+1, p+1);
        Eigen::Map<Eigen::MatrixXd> p_ann(p_ann_1d.data(), p+1, p+1);

        double s_gate = 1.0;
        ut_ann = (ut_ann.array()*s_gate);

        Eigen::MatrixXd ct = theta.array().cos();
        Eigen::MatrixXd st = theta.array().sin();

        Eigen::MatrixXd u_ann = -(-ut_ann.array()*st.array() + un_ann.array()*ct.array());
        Eigen::MatrixXd v_ann = -(ut_ann.array()*ct.array() + un_ann.array()*st.array());
        
        Eigen::ArrayXXd a = blendx(xN.array(),x0,w_tr);

        Eigen::MatrixXd rho = a*rho_flat.array() + (1-a)*rho_ann.array();
        Eigen::MatrixXd pres = a*p_flat.array() + (1-a)*p_ann.array();
        Eigen::MatrixXd u = a*u_flat.array() + (1-a)*u_ann.array();
        Eigen::MatrixXd v = a*v_flat.array() + (1-a)*v_ann.array();  
        Eigen::MatrixXd w = Eigen::MatrixXd::Zero(p+1,p+1);
     
        Eigen::VectorXd rhoV = rho.reshaped();
        Eigen::VectorXd uV = u.reshaped();
        Eigen::VectorXd vV = v.reshaped();
        Eigen::VectorXd wV = w.reshaped();
        Eigen::VectorXd pV = pres.reshaped();

        brho.col(i1) = rhoV;
        brhou.col(i1) = rhoV.array() * uV.array();
        brhov.col(i1) = rhoV.array() * vV.array();
        brhow.col(i1) = rhoV.array() * wV.array();
        bE.col(i1) = pV.array()/(gam-1.0) + 0.5*rhoV.array()*(uV.array().square() + vV.array().square() + wV.array().square());
        bu.col(i1) = uV;
        bv.col(i1) = vV;
        bw.col(i1) = wV;
        bp.col(i1) = pV;

        double r_tol = 1e-10 * std::max(1.0, yN.mean());
        msh.elem[i1].r    = yN.cwiseMax(r_tol);
        msh.elem[i1].invr = (1.0 / msh.elem[i1].r.array()).matrix();   // adjust per type

        // Optional pressure check
        if (i1 % stride == 0) {
            Eigen::ArrayXd rho_col  = brho.col(i1).array();
            Eigen::ArrayXd rhou_col = brhou.col(i1).array();
            Eigen::ArrayXd rhov_col = brhov.col(i1).array();
            Eigen::ArrayXd rhow_col = brhow.col(i1).array();
            Eigen::ArrayXd E_col    = bE.col(i1).array();

            Eigen::ArrayXd kE = 0.5 * rho_col
                * ( (rhou_col / rho_col).square()
                + (rhov_col / rho_col).square()
                + (rhow_col / rho_col).square() );

            Eigen::ArrayXd p_chk = (gam - 1.0) * (E_col - kE);

            std::printf("elem %d: min rho=%.3e, min p=%.3e, max|p-p_chk|=%.3e\n",
                        i1,
                        rho_col.minCoeff(),
                        pV.minCoeff(),
                        (pV.array() - p_chk).abs().maxCoeff());
        }
    }

    // Stack the block matrices vertically
    Eigen::MatrixXd bU(5 * npe, Ne);
    bU << brho,
        brhou,
        brhov,
        brhow,
        bE;
    
    save_csv(std::string(OUTPUT_DIR) + "/bU.csv",bU);
    Eigen::MatrixXd elem_x(npe, Ne);
    Eigen::MatrixXd elem_y(npe, Ne);
    for (int i1 = 0; i1 < Ne; ++i1) {
        elem_x.col(i1) = msh.elem[i1].x.reshaped();
        elem_y.col(i1) = msh.elem[i1].y.reshaped();
    }
    
    save_csv(std::string(OUTPUT_DIR) + "/elem_x.csv",elem_x);
    save_csv(std::string(OUTPUT_DIR) + "/elem_y.csv",elem_y);


    // Faces interpolation
    std::cout <<"============ Creating Face Baseflow from elements ============"<< std::endl;

    const int nf  = p + 1;
    Eigen::MatrixXd brho_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd brhou_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd brhov_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd brhow_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd bE_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd bu_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd bv_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd bw_f = Eigen::MatrixXd::Zero((nf),Nf);
    Eigen::MatrixXd bp_f = Eigen::MatrixXd::Zero((nf),Nf);

    const int faces_per_elem = msh.EtoF.rows();

    for (int i1 = 0; i1 < Nf; ++i1) {
        int e1 = msh.FtoE(0, i1);
        int e2 = msh.FtoE(1, i1);

        if (e2 < 0) {
            int f1 = find_local_face(msh.EtoF, e1, i1, faces_per_elem);
            assert(f1 >= 0);
            const auto& T1 = op2.T[f1];   // already nf x npe; no slicing needed

            brho_f.col(i1) = T1 * brho.col(e1);
            bu_f.col(i1)   = T1 * bu.col(e1);
            bv_f.col(i1)   = T1 * bv.col(e1);
            bw_f.col(i1)   = T1 * bw.col(e1);
            bp_f.col(i1)   = T1 * bp.col(e1);
        }
        else if (e1 < 0) {
            int f2 = find_local_face(msh.EtoF, e2, i1, faces_per_elem);
            assert(f2 >= 0);
            const auto& T1 = op2.T[f2];

            brho_f.col(i1) = T1 * brho.col(e2);
            bu_f.col(i1)   = T1 * bu.col(e2);
            bv_f.col(i1)   = T1 * bv.col(e2);
            bw_f.col(i1)   = T1 * bw.col(e2);
            bp_f.col(i1)   = T1 * bp.col(e2);
        }
        else {
            int f1 = find_local_face(msh.EtoF, e1, i1, faces_per_elem);
            int f2 = find_local_face(msh.EtoF, e2, i1, faces_per_elem);
            assert(f1 >= 0 && f2 >= 0);

            const auto& T_e1 = op2.T[f1];
            const auto& T_e2 = op2.T[f2];

            brho_f.col(i1) = 0.5 * (T_e1 * brho.col(e1) + T_e2 * brho.col(e2));
            bu_f.col(i1)   = 0.5 * (T_e1 * bu.col(e1)   + T_e2 * bu.col(e2));
            bv_f.col(i1)   = 0.5 * (T_e1 * bv.col(e1)   + T_e2 * bv.col(e2));
            bw_f.col(i1)   = 0.5 * (T_e1 * bw.col(e1)   + T_e2 * bw.col(e2));
            bp_f.col(i1)   = 0.5 * (T_e1 * bp.col(e1)   + T_e2 * bp.col(e2));
        }

        brhou_f.col(i1) = brho_f.col(i1).array() * bu_f.col(i1).array();
        brhov_f.col(i1) = brho_f.col(i1).array() * bv_f.col(i1).array();
        brhow_f.col(i1) = brho_f.col(i1).array() * bw_f.col(i1).array();

        bE_f.col(i1) = bp_f.col(i1).array() / (gam - 1.0)
                    + 0.5 * brho_f.col(i1).array()
                        * ( bu_f.col(i1).array().square()
                            + bv_f.col(i1).array().square()
                            + bw_f.col(i1).array().square() );

        const double r_tol = 1e-10;
        msh.face[i1].r    = msh.face[i1].y.cwiseMax(r_tol);
        msh.face[i1].invr = (1.0 / msh.face[i1].r.array()).matrix();   // adjust per type

        if ((bp_f.col(i1).array() < 0.0).any()) {
            std::printf("Warning: Negative Pressure on Face %d\n", i1);
        }
    }
    
    Eigen::MatrixXd bU_f(5 * nf, Nf);
    bU_f << brho_f,
        brhou_f,
        brhov_f,
        brhow_f,
        bE_f;
    
    save_csv(std::string(OUTPUT_DIR) + "/bU_f.csv",bU_f);
    Eigen::MatrixXd face_x(nf, Nf);
    Eigen::MatrixXd face_y(nf, Nf);
    for (int i1 = 0; i1 < Nf; ++i1) {
        face_x.col(i1) = msh.face[i1].x.reshaped();
        face_y.col(i1) = msh.face[i1].y.reshaped();
    }
    
    save_csv(std::string(OUTPUT_DIR) + "/face_x.csv",face_x);
    save_csv(std::string(OUTPUT_DIR) + "/face_y.csv",face_y);

    Grad_Variables grad_variables =calc_grads_strong(bU,bU_f,msh,nvar,op1,op2);

    save_csv(std::string(OUTPUT_DIR) + "/bUx.csv",grad_variables.bUx);
    save_csv(std::string(OUTPUT_DIR) + "/bUy.csv",grad_variables.bUy);
    save_csv(std::string(OUTPUT_DIR) + "/bUz.csv",grad_variables.bUz);

    save_csv(std::string(OUTPUT_DIR) + "/bUx_f.csv",grad_variables.bUx_f);
    save_csv(std::string(OUTPUT_DIR) + "/bUy_f.csv",grad_variables.bUy_f);
    save_csv(std::string(OUTPUT_DIR) + "/bUz_f.csv",grad_variables.bUz_f);
}