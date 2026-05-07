#pragma once

#include "hdg/calcgeom_struct.h"
#include "hdg/phys_mesh.h"
#include <Eigen/Dense>

// Notice we use Eigen::MatrixXd instead of just MatrixXd
calc_geom_output calc_geom_iso_tfi(const int p, const Eigen::MatrixXd& xi, const Eigen::MatrixXd& eta,
                                  const Eigen::RowVector4d& elem, const PhysMesh& phys_msh);