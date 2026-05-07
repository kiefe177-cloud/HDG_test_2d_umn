#include "hdg/calc_geom_iso_tfi.h"
#include "hdg/generate_tfi.h"
#include <iostream>

calc_geom_output calc_geom_iso_tfi(const int p, const Eigen::MatrixXd& xi, const Eigen::MatrixXd& eta, const Eigen::RowVector4d& elem, const PhysMesh& phys_msh) {

    double phi_min = elem(0);
    double psi_min = elem(1);
    double phi_max = elem(2);
    double psi_max = elem(3);
    
    double phi_c = (phi_min + phi_max) / 2.0;
    double dphi  = (phi_max - phi_min) / 2.0;

    double psi_c = (psi_min + psi_max) / 2.0;
    double dpsi  = (psi_max - psi_min) / 2.0;

    Eigen::MatrixXd phi = (xi.array() * dphi + phi_c).matrix();
    Eigen::MatrixXd psi = (eta.array() * dpsi + psi_c).matrix();
    
    Eigen::MatrixXd Xloc;
    Eigen::MatrixXd Yloc;

    compute_tfi_matrices(phi, psi, phys_msh.geom.f.left_boundary, phys_msh.geom.f.right_boundary,
                                phys_msh.geom.f.bottom_boundary, phys_msh.geom.f.top_boundary,
                                phys_msh.geom.f.x_sw, phys_msh.geom.f.y_sw,
                                phys_msh.geom.f.x_se, phys_msh.geom.f.y_se,
                                phys_msh.geom.f.x_ne, phys_msh.geom.f.y_ne,
                                phys_msh.geom.f.x_nw, phys_msh.geom.f.y_nw,
                                Xloc, Yloc);

    Eigen::MatrixXd Dr1d = phys_msh.geom.Dr1d;
    
    Eigen::MatrixXd x_xi   = Dr1d * Xloc;
    Eigen::MatrixXd x_eta  = Xloc * (Dr1d.transpose());
    Eigen::MatrixXd y_xi   = Dr1d * Yloc;
    Eigen::MatrixXd y_eta  = Yloc * (Dr1d.transpose());

    Eigen::MatrixXd x_xi_y_eta = x_xi.cwiseProduct(y_eta);
    Eigen::MatrixXd y_xi_x_eta = x_eta.cwiseProduct(y_xi);

    Eigen::MatrixXd J = x_xi_y_eta - y_xi_x_eta;
    
    Eigen::MatrixXd Ja11 = y_eta;
    Eigen::MatrixXd Ja12 = -x_eta;
    Eigen::MatrixXd Ja21 = -y_xi;
    Eigen::MatrixXd Ja22 = x_xi;

    int end_row = x_eta.rows() - 1;
    int end_col = x_xi.cols() - 1;

    Eigen::VectorXd Jf_1 = (x_eta.row(0).array().square() + y_eta.row(0).array().square()).sqrt().transpose();
    Eigen::VectorXd Jf_2 = (x_eta.row(end_row).array().square() + y_eta.row(end_row).array().square()).sqrt().transpose();
    
    // CORRECTED: These now properly extract the columns instead of the rows!
    Eigen::VectorXd Jf_3 = (x_xi.col(0).array().square() + y_xi.col(0).array().square()).sqrt();
    Eigen::VectorXd Jf_4 = (x_xi.col(end_col).array().square() + y_xi.col(end_col).array().square()).sqrt();
    
    // ==========================================
    // POPULATE AND RETURN THE STRUCT
    // ==========================================
    calc_geom_output out;
    
    out.x1 = Xloc; 
    out.y1 = Yloc; 
    out.J = J;
    out.Ja11 = Ja11;
    out.Ja12 = Ja12;
    out.Ja21 = Ja21;
    out.Ja22 = Ja22;
    out.phi = phi;
    out.psi = psi;
    out.Jf_1 = Jf_1;
    out.Jf_2 = Jf_2;
    out.Jf_3 = Jf_3;
    out.Jf_4 = Jf_4;
    
    // std::cout << "xi:\n" << xi << "\n";
    // std::cout << "eta:\n" << eta << "\n";
    // std::cout << "phi:\n" << phi << "\n";
    // std::cout << "psi:\n" << psi << "\n";
    // std::cout << "Xloc:\n" << Xloc << "\n";
    // std::cout << "Yloc:\n" << Yloc << "\n";
    // std::cout << "x_xi:\n" << x_xi << "\n";
    // std::cout << "x_eta:\n" << x_eta << "\n";
    // std::cout << "Jf_1: " << Jf_1.transpose() << "\n";
    // std::cout << "Jf_2: " << Jf_2.transpose() << "\n";
    // std::cout << "Jf_3: " << Jf_3.transpose() << "\n";
    // std::cout << "Jf_4: " << Jf_4.transpose() << "\n";

    return out;
}