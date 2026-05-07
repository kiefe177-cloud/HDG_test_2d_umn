#pragma once

#include "hdg/amr_refine.h"
#include "hdg/amr_number.h"
#include <Eigen/Dense>

struct Connect {
    Eigen::MatrixXi FtoE;
    Eigen::MatrixXi EtoF;
};

Connect msh_connect(AmrNode& amr);