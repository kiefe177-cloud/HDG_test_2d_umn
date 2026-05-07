#pragma once

#include "parameters/MalikSpallParams.h"
#include "parameters/params_struct.h"
#include <Eigen/Dense>
#include <array>

// 1D port of the MATLAB `baseflow` class.
//
// Stores conservative flow variables (rho, rhou, E) on a wall-normal grid,
// plus the (x, y, z) coordinate triple. Dependent quantities (p, T, mu) are
// computed on demand to mirror MATLAB's `Dependent` properties.
//
// 2D/3D, slicing, displacement-thickness, and plotting methods from the
// MATLAB original are intentionally omitted — add them back as downstream
// code requires.
class Baseflow {
public:
    // Grid shape. For the 1D wall-normal baseflow, Nx = Nz = 1 and Ny is
    // the number of wall-normal points.
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;

    // Coordinates: x[0] = x, x[1] = y, x[2] = z. Each entry is a column
    // vector of length Ny (for the 1D case).
    std::array<Eigen::VectorXd, 3> x;

    // Conservative variables.
    Eigen::VectorXd rho;                    // density
    std::array<Eigen::VectorXd, 3> rhou;    // momentum components (rho*u, rho*v, rho*w)
    Eigen::VectorXd E;                      // total energy per unit volume

    Baseflow() = default;

    // Mirrors the MATLAB constructor:
    //   obj = baseflow(params, x, rho, rhou, E)
    // Caller is responsible for ensuring all input vectors have matching length Ny.
    Baseflow(const SimulationParams& params,
             const std::array<Eigen::VectorXd, 3>& x_in,
             const Eigen::VectorXd& rho_in,
             const std::array<Eigen::VectorXd, 3>& rhou_in,
             const Eigen::VectorXd& E_in)
        : params_(params),
          x(x_in),
          rho(rho_in),
          rhou(rhou_in),
          E(E_in)
    {
        // 1D baseflow: shape is (1, Ny, 1).
        Nx = 1;
        Ny = static_cast<int>(rho.size());
        Nz = 1;
    }

    // ---- Dependent quantities (recomputed on each call, like MATLAB's get.* ) ----

    // Pressure: p = (gam - 1) * (E - 0.5 * |rhou|^2 / rho)
    Eigen::VectorXd p() const {
        const double gam = params_.gam;
        const auto rhou_sq = rhou[0].array().square()
                           + rhou[1].array().square()
                           + rhou[2].array().square();
        return (gam - 1.0) * (E.array() - 0.5 * rhou_sq / rho.array());
    }

    // Temperature: T = gam * M^2 * p / rho
    Eigen::VectorXd T() const {
        const double gam = params_.gam;
        const double M   = params_.M;
        return (gam * M * M) * (p().array() / rho.array());
    }

    // Sutherland viscosity: mu = (1 + Ts) / (T + Ts) * T^(3/2)
    Eigen::VectorXd mu() const {
        const double Ts = params_.Ts;
        const Eigen::ArrayXd Tarr = T().array();
        return ((1.0 + Ts) / (Tarr + Ts)) * Tarr.pow(1.5);
    }

    // Read-only access to parameters (some downstream code may want them).
    const SimulationParams& params() const { return params_; }

private:
    SimulationParams params_;
};