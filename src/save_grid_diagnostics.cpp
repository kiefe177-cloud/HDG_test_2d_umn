#include "hdg/save_grid_diagnostics.h"

#include <highfive/H5Easy.hpp>

using namespace H5Easy;

void save_grid_diagnostics(
    const std::string& path,
    const Eigen::MatrixXd& Phi_tfi,
    const Eigen::MatrixXd& Psi_tfi,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    const Eigen::VectorXd& x_left,    const Eigen::VectorXd& y_left,
    const Eigen::VectorXd& x_right,   const Eigen::VectorXd& y_right,
    const Eigen::VectorXd& x_bottom,  const Eigen::VectorXd& y_bottom,
    const Eigen::VectorXd& x_top,     const Eigen::VectorXd& y_top,
    const double& x_p,                const double& y_p,
    const double& phi_p,              const double& psi_p,
    const Eigen::MatrixXd& skewness,
    const Eigen::MatrixXd& jacobian)
    {
        File file(path, File::Overwrite);

        dump(file, "/computational/Phi", Phi_tfi);
        dump(file, "/computational/Psi", Psi_tfi);
        dump(file, "/computational/phi_p", phi_p);
        dump(file, "/computational/psi_p", psi_p);

        dump(file, "/physical/X", X);
        dump(file, "/physical/Y", Y);
        dump(file, "/physical/x_p", x_p);
        dump(file, "/physical/y_p", y_p);

        dump(file, "/boundaries/left/x", x_left);
        dump(file, "/boundaries/left/y", y_left);

        dump(file, "/boundaries/right/x", x_right);
        dump(file, "/boundaries/right/y", y_right);

        dump(file, "/boundaries/bottom/x", x_bottom);
        dump(file, "/boundaries/bottom/y", y_bottom);
        
        dump(file, "/boundaries/top/x", x_top);
        dump(file, "/boundaries/top/y", y_top);

        dump(file, "/diagnostics/skewness", skewness);
        dump(file, "/diagnostics/jacobian", jacobian);    
    }