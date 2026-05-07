#pragma once
#include "parameters/params_struct.h"
#include <cmath>

// struct SimulationParams {
//     double M;
//     double Re_unit;
//     double Re_x;
//     double Re_L;
//     double gam;
//     double T0;
//     double Te;
//     double Tw;
//     double Pr;
//     double y_mid;
//     double y_end;
//     double Ts;
//     double R_air;
//     double L;
//     double ue;
//     double h_e;
//     double rho_e;
//     double a_e;
// };


inline SimulationParams get_M5p8_params(){

    SimulationParams p;

    p.M = 5.8;                             // Freestream Mach number
    p.Re_unit = 2.28e6;

    p.Te = 56.0;
    p.rho_e = 0.01;
    p.gam = 1.4;

    p.R_air = 287.05;
    p.Pr = 0.72;

    double S_dim = 110.3;
    double Tw_dim = 300.0;

    p.a_e = std::sqrt(p.gam * p.R_air * p.Te);
    p.ue = p.M * p.a_e;

    p.T0 = p.Te * (1.0 + (p.gam-1.0)/2.0 * std::pow(p.M,2.0));

    p.h_e = p.gam * p.R_air * p.Te / (p.gam-1);

    p.Ts = S_dim / p.Te;

    p.Tw = Tw_dim / p.Te;

    double x_station = 1.0;
    p.Re_x = p.Re_unit * x_station;

    p.Re_L = std::sqrt(p.Re_x);

    p.L = std::sqrt(p.Re_x) / p.Re_unit;

    return p;
}