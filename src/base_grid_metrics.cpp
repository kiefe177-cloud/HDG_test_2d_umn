#include "hdg/base_grid_metrics.h"

BaseGridMetrics base_grid_metrics(
    const Eigen::MatrixXd& X_ellip_grid,
    const Eigen::MatrixXd& Y_ellip_grid,
    const Eigen::VectorXd& phi_comp_vec,
    const Eigen::VectorXd& psi_comp_vec)
{
    BaseGridMetrics base_grid;

    base_grid.X       = X_ellip_grid;
    base_grid.Y       = Y_ellip_grid;
    base_grid.phi_vec = phi_comp_vec;
    base_grid.psi_vec = psi_comp_vec;
    base_grid.nx      = static_cast<int>(phi_comp_vec.size()) - 1;
    base_grid.ny      = static_cast<int>(psi_comp_vec.size()) - 1;
    base_grid.dphi    = phi_comp_vec(1) - phi_comp_vec(0);
    base_grid.dpsi    = psi_comp_vec(1) - psi_comp_vec(0);

    return base_grid;
}