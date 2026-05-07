#include "hdg/baseflow_flatplate_1d.h"
#include "hdg/bl_similarity.h"
#include "hdg/interp1.h"
#include <array>
#include <cmath>
#include <iostream>

Eigen::VectorXd mesh1d_alg(int Ny, double yi, double ymax) {
    Eigen::VectorXd eta = Eigen::VectorXd::LinSpaced(Ny, 0.0, 1.0);

    double a = ymax * yi / (ymax - 2.0 * yi);
    double b = 1.0 + a / ymax;

    return ((a * eta.array()) / (b - eta.array())).matrix();
}

Baseflow baseflow_flatplate_1d(SimulationParams& params, int Ny, int /*i1*/) {
    const double Re = params.Re_L;
    const double x  = Re;

    // Similarity solution on a fine mesh
    BlProfiles bl = bl_similarity(params, 10 * Ny);
    bl.y[0] = 0.0;

    // std::cout << "Break 1" << std::endl;

    // Auto-detect mesh extents if y_mid was left at zero
    if (params.y_mid == 0.0) {
        int target_idx = 0;
        for (int i = static_cast<int>(bl.u.size()) - 1; i >= 0; --i) {
            if (bl.u[i] < 0.9) {
                target_idx = i;
                break;
            }
        }
        params.y_mid = 1.5 * bl.y[target_idx] * std::sqrt(x);
        params.y_end = 8.0 * bl.y.maxCoeff() * std::sqrt(x);
    }

    
    // std::cout << "Break 2" << std::endl;

    // Stretched wall-normal mesh for stability analysis
    Eigen::VectorXd y2 = mesh1d_alg(Ny, params.y_mid, params.y_end);

    // Interpolate similarity solution onto y2 (extrap to freestream value 1.0)
    Eigen::VectorXd y_scaled = bl.y * std::sqrt(x);

    // std::cout << "y_scaled[0] = " << y_scaled[0] << "\n";
    // std::cout << "y2[0]       = " << y2[0] << "\n";
    // std::cout << "comparison: y2[0] < y_scaled[0]? " 
    //       << (y2[0] < y_scaled[0]) << "\n";
    Eigen::VectorXd UU = interp1(y_scaled, bl.u, y2, InterpMethod::Spline, 1.0);
    Eigen::VectorXd TT = interp1(y_scaled, bl.T, y2, InterpMethod::Spline, 1.0);

    
    //std::cout << "Break 3" << std::endl;

    // Convert to conservative variables (locally parallel: v = w = 0)
    const double gam = params.gam;
    const double M   = params.M;

    std::array<Eigen::VectorXd, 3> x_coords = {
        Eigen::VectorXd::Zero(Ny),
        y2,
        Eigen::VectorXd::Zero(Ny)
    };

    Eigen::VectorXd rho = (1.0 / TT.array()).matrix();

    std::array<Eigen::VectorXd, 3> rhou = {
        (rho.array() * UU.array()).matrix(),
        Eigen::VectorXd::Zero(Ny),
        Eigen::VectorXd::Zero(Ny)
    };

    Eigen::VectorXd E = ( 1.0 / (gam * (gam - 1.0) * M * M)
                        + 0.5 * rho.array() * UU.array().square() ).matrix();

    
    // std::cout << "Break 4" << std::endl;

    return Baseflow(params, x_coords, rho, rhou, E);
}