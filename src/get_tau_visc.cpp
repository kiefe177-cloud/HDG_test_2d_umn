#include "hdg/get_tau_visc.h"
#include "hdg/sd.h"

Eigen::SparseMatrix<double> get_tau_visc(const Msh& msh,
                                        const Eigen::MatrixXd& brho_face,
                                        const Eigen::MatrixXd& brhou_face,
                                        const Eigen::MatrixXd& brhov_face,
                                        const Eigen::MatrixXd& brhow_face,
                                        const Eigen::MatrixXd& bE_face,
                                        const SimulationParams& params,
                                        const double c_v,
                                        const int i2,
                                        const int i1)
{

    using SparseD = Eigen::SparseMatrix<double>;

    const double Pr = params.Pr;
    const double gam = params.gam;
    const double Minf = params.M;
    const double Re = params.Re_L;

    const int elem_idx = i1;
    const int face_idx = i2;
    const int p = msh.elem[elem_idx].p;
    const double d = 2.0;

    Eigen::VectorXd w1d = msh.elem[elem_idx].w1d;
    Eigen::MatrixXd W2d = w1d * w1d.transpose();                       // outer product

    Eigen::VectorXd Jdiag = msh.elem[elem_idx].J.diagonal();
    Eigen::Map<const Eigen::MatrixXd> Jdet(Jdiag.data(), p+1, p+1);    // reshape column-major

    double elem_meas = (W2d.array() * Jdet.array()).sum();

    double face_meas = (msh.face[face_idx].w1d.array() * msh.face[face_idx].Jf.array()).sum();

    double npe = static_cast<double>(p) + 1.0;
    double npd = static_cast<double>(p) + d;
    double eta_f = (npe)*(npd) / d * (face_meas/elem_meas);

    
    Eigen::VectorXd rho = brho_face.col(face_idx);
    Eigen::VectorXd rhou = brhou_face.col(face_idx);
    Eigen::VectorXd rhov = brhov_face.col(face_idx);
    Eigen::VectorXd rhow = brhow_face.col(face_idx);
    Eigen::VectorXd E = bE_face.col(face_idx);
    Eigen::VectorXd inv_rho = rho.cwiseInverse();

    Eigen::VectorXd pres = (gam - 1.0) * (E.array() - 0.5*(rhou.array().square() + rhov.array().square() + rhow.array().square())*inv_rho.array());
    Eigen::VectorXd T = (gam*Minf*Minf) * (pres.array()*inv_rho.array());
    double S_nd = params.Ts;
    Eigen::VectorXd mu = (T.array() * T.array().sqrt() * ((1.0 + S_nd) / (T.array() + S_nd))).matrix();

    
    Eigen::VectorXd mom_coeff = (4.0/3.0) * (mu.array() * inv_rho.array())/ Re;
    Eigen::VectorXd eng_coeff = (gam/Pr) * (mu.array() * inv_rho.array()) / Re;

    int Npts = rho.size();
    Eigen::VectorXd tau_rho = Eigen::VectorXd::Zero(Npts);
    Eigen::VectorXd tau_mom = c_v * eta_f * mom_coeff;
    Eigen::VectorXd tau_eng = c_v * eta_f * eng_coeff;

    Eigen::VectorXd diagvec(5 * Npts);
    diagvec << tau_rho, tau_mom, tau_mom, tau_mom, tau_eng;   // comma initializer

    return sd(diagvec);
}