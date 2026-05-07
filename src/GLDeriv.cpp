#include <cmath>

#include "hdg/GLDeriv.h" 
#include "hdg/GLQuad.h"
#include "hdg/jacobiP.h"



Eigen::MatrixXd GLDeriv(int N){
    // Compute the Gauss-Lobatto-Legendre Derivative Matrix of order N

    // 1. Get the GL points
    Eigen::VectorXd r, w;
    GLQuad(N, r, w);

    // 2. Initialize Vandermonde Matrices V and Vr
    int size = N+1;
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(size, size);
    Eigen::MatrixXd Vr = Eigen::MatrixXd::Zero(size, size);

    // 3. Fill V and Vr
    // V_ij = sqrt( (2j+1)/2 ) * P_j^(0,0)(r_i)
    for (int j = 0; j <= N; ++j){
        V.col(j) = std::sqrt(j+0.5) * jacobiP(j, 0.0, 0.0, r);
    }

    // Vr_ij = (j+1) * sqrt( (2j+1)/8 ) * P_(j-1)^(1,1)(r_i)
    for (int j = 1; j <= N; ++j){
        Vr.col(j) = (j+1) * std::sqrt((2*j+1)/8.0) * jacobiP(j-1, 1.0, 1.0, r);
    }

    // 4. Compute the Derivative Matrix D
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(size, size);
    // D = Vr * V^{-1}
    D = Vr * V.inverse();

    return D;
}
