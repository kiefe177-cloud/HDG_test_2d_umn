#pragma once

#include "hdg/amr_refine.h"
#include <Eigen/Dense>
#include <string>

// Returns an Nleaves x 4 matrix where each row is [x0, y0, x2, y2].
Eigen::MatrixXd collect_amr_leaves(const AmrNode& amr);

// Convenience: walks the tree and writes one CSV row per leaf.
void dump_amr_leaves_csv(const std::string& path, const AmrNode& amr);