#include "hdg/dg_mat2_Roe.h"
#include "hdg/msh_amr_curv.h"

#include <Eigen/Dense>


struct HDG_Result{
    Eigen::VectorXcd u_e;
    Eigen::VectorXcd u_f;
};

HDG_Result dg_solve(const Eigen::VectorXcd& src_e,const Eigen::VectorXcd& src_f,const Msh& msh, const dg& dg_,int nvar);
