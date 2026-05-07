#pragma once
#include "parameters/params_struct.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// The struct we discussed earlier to return multiple vectors safely
struct BlProfiles {
    Eigen::VectorXd y;
    Eigen::VectorXd u;
    Eigen::VectorXd v;
    Eigen::VectorXd T;
};

BlProfiles bl_similarity(const SimulationParams& params, int N);