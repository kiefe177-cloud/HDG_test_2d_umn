// include/hdg/ortho_opts.h
#pragma once
#include <string>

enum class EllipticMethod {
    TTM,
    Ortho,
    FullOrtho,
    StegerSorenson
};

struct OrthoOpts {
    bool   enable         = true;
    bool   walls          = true;
    bool   outer          = true;
    double relax          = 1.0;
    int    n_ortho_sweeps = 400;
    double ortho_tol      = 1e-12;

    // Full Ortho extras
    double wall_relax     = 1.0;
    double march_relax    = 1.0;
    int    n_march_sweeps = 1000;
    double march_tol      = 1e-12;
    bool   wall_slide     = true;
    double slide_relax    = 1.0;
};