#include "hdg/dg_mat2_Roe.h"
#include "hdg/msh_amr_curv.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>


struct HDG_Result{
    Eigen::VectorXcd u_e;
    Eigen::VectorXcd u_f;
    Eigen::SparseMatrix<std::complex<double>> K;
    Eigen::VectorXcd R;
};

HDG_Result dg_solve(const Eigen::VectorXcd& src_e,const Eigen::VectorXcd& src_f, const Msh& msh, const dg& dg_, int nvar);
