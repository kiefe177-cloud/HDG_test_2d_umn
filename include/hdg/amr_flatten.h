#pragma once

#include "hdg/amr_refine.h"
#include <Eigen/Dense>

// Takes the AMR tree and returns an N x 5 matrix of all leaf bounding boxes and IDs
Eigen::MatrixXd amr_flatten(const AmrNode& amr);