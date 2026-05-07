#pragma once
#include <cmath>
#include "params_struct.h"

inline SimulationParams get_malik_spall_params(){

    SimulationParams p;

    p.M = 5;                             // Freestream Mach number
    // p.Re_unit = 10^6;
    p.Re_unit = 3281000;                   // Unit Reynolds number (per meter) (1e6/ft = 3.281e6/m)
    p.Re_x = 2000000;
    p.Re_L = std::sqrt(p.Re_x);
    // p.Re_L = 2000000;                      // Reynolds number based on x

    p.gam = 1.4;
    p.T0 = 300;                            // Stagnation temperature (K) 
    p.Te = p.T0 / (1.0 + (p.gam-1)/ 2* std::pow(p.M,2.0));       // Edge temperature (K)
    p.Tw = 0;                              // Adibatic wall at first


    p.Pr = 0.7;                        // Malik uses 0.7 for Prandtl number of air
    p.y_mid = 0;                       // auto-detect mesh stretching from similarity solution
    p.y_end = 0;
    p.Ts = 110/p.Te;
    // p.Tw = baseflow_flatplate_Ta(p); // Set isothermal wall temp equal to adiabatic wall temp (hot wall)...
    p.Tw = 258.5709;

    //Added
    p.R_air = 287.06;                      // Specific gas constant
    // p.L = sqrt(1/p.Re_x);
    p.L = std::sqrt(p.Re_x) / p.Re_unit; // Viscous reference length
    p.ue = p.M*std::sqrt(p.gam*p.R_air*p.Te); // Edge velocity
    p.h_e = p.gam*p.R_air*p.Te/(p.gam-1); // Edge enthalpy

    return p;
}