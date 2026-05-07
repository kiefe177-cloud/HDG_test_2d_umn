#pragma once
#include "hdg/msh_amr_curv.h"
#include "hdg/sem1.h"
#include "hdg/sem2.h"
#include <Eigen/Dense>


struct Grad_Variables
{
    Eigen::MatrixXd bUx;
    Eigen::MatrixXd bUy;
    Eigen::MatrixXd bUz;
    Eigen::MatrixXd bUx_f;
    Eigen::MatrixXd bUy_f;
    Eigen::MatrixXd bUz_f;
};

Grad_Variables calc_grads_strong(const Eigen::MatrixXd& bU, const Eigen::MatrixXd& bU_f,const Msh& msh, const int nvar,const Sem1Op& op1, const Sem2Op& op2);
