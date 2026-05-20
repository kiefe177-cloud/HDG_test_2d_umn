#include "hdg/save_outputs_h5.h"

#include <highfive/H5Easy.hpp>

using namespace H5Easy;

void save_outputs_h5(
    const std::string& path,
    const Eigen::MatrixXd& Phi_tfi,
    const Eigen::MatrixXd& Psi_tfi,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const Msh& msh,
    const Eigen::MatrixXd& brho,
    const Eigen::MatrixXd& brhou,
    const Eigen::MatrixXd& brhov,
    const Eigen::MatrixXd& brhow,
    const Eigen::MatrixXd& bE,
    const int Ne,
    const int Nf,
    const int Me,
    const int Mf,
    const int p,
    const Eigen::VectorXcd& u_e,
    const Eigen::VectorXcd& u_f)
    {
    File file(path, File::Overwrite);

    dump(file, "/grid/computational/Phi", Phi_tfi);
    dump(file, "/grid/computational/Psi", Psi_tfi);

    dump(file, "/grid/physical/X", X);
    dump(file, "/grid/physical/Y", Y); 
    dump(file, "/grid/physical/Ne", Ne);
    dump(file, "/grid/physical/Nf", Nf);  
    dump(file, "/grid/physical/Me", Me);
    dump(file, "/grid/physical/Mf", Mf);
    dump(file, "/grid/physical/p", p);

    const int Ne_local = static_cast<int>(msh.elem.size());

    if (Ne_local > 0) {
        const int p_e   = msh.elem[0].p;
        const int npts  = (p_e + 1) * (p_e + 1);

        Eigen::VectorXi p_all(Ne_local);
        for (int e = 0; e < Ne_local; ++e) p_all(e) = msh.elem[e].p;
        dump(file, "/mesh/elem/p", p_all);

        Eigen::MatrixXd x_all(npts, Ne_local);
        Eigen::MatrixXd y_all(npts, Ne_local);

        Eigen::MatrixXd J_diag   (npts, Ne_local);
        Eigen::MatrixXd Ja11_diag(npts, Ne_local);
        Eigen::MatrixXd Ja12_diag(npts, Ne_local);
        Eigen::MatrixXd Ja21_diag(npts, Ne_local);
        Eigen::MatrixXd Ja22_diag(npts, Ne_local);

        for (int e = 0; e < Ne_local; ++e) {
            const auto& el = msh.elem[e];

            x_all.col(e) = Eigen::Map<const Eigen::VectorXd>(el.x.data(), npts);
            y_all.col(e) = Eigen::Map<const Eigen::VectorXd>(el.y.data(), npts);

            J_diag.col(e)    = el.J.diagonal();
            Ja11_diag.col(e) = el.Ja11.diagonal();
            Ja12_diag.col(e) = el.Ja12.diagonal();
            Ja21_diag.col(e) = el.Ja21.diagonal();
            Ja22_diag.col(e) = el.Ja22.diagonal();
        }

        dump(file, "/mesh/elem/x",    x_all);
        dump(file, "/mesh/elem/y",    y_all);
        dump(file, "/mesh/elem/J",    J_diag);
        dump(file, "/mesh/elem/Ja11", Ja11_diag);
        dump(file, "/mesh/elem/Ja12", Ja12_diag);
        dump(file, "/mesh/elem/Ja21", Ja21_diag);
        dump(file, "/mesh/elem/Ja22", Ja22_diag);

    }
        dump(file, "/baseflow/element/rho", brho);
        dump(file, "/baseflow/element/rhou", brhou);
        dump(file, "/baseflow/element/rhov", brhov);
        dump(file, "/baseflow/element/rhow", brhow);
        dump(file, "/baseflow/element/E", bE);
        
        dump(file, "/outputs/u_e", u_e);   
        dump(file, "/outputs/u_f", u_f);   
    }