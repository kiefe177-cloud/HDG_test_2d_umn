#include <iostream>

#include "hdg/comp_space_metrics.h"
#include "hdg/GLDeriv.h"

GeomMetrics comp_space_metrics(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    int p,
    const Eigen::VectorXd& phi_s,
    const Eigen::VectorXd& psi_s,
    const TFI_Mesh::Boundaries& f)
{
    GeomMetrics geom;

    geom.phi_global_vec = phi_s;
    geom.psi_global_vec = psi_s;

    geom.Fx_interpolant.fit(geom.phi_global_vec, geom.psi_global_vec, X);
    geom.Fy_interpolant.fit(geom.phi_global_vec, geom.psi_global_vec, Y);

    geom.Dr1d = GLDeriv(p);
    geom.f = f;

    return geom;
}