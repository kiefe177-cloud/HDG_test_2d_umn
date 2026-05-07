#include "hdg/GLQuad.h"
#include "hdg/jacobiZerosGW.h"
#include "hdg/legendreP.h"

void GLQuad(int N, Eigen::VectorXd &x, Eigen::VectorXd &w) {
    // Compute the Gauss-Lobatto-Legendre points and weights
    
    // 1. Compute the roots of the Jacobi Polynomial P_(N-1)^(1,1)
    Eigen::VectorXd roots = jacobiZerosGW(N-1,1,1);
    x.resize(roots.size() + 2);
    x << -1.0, roots, 1.0;

    // 2. Compute the weights using the Legendre polynomial P_N at the points x
    Eigen::VectorXd P = legendreP(N, x);
    w.resize(x.size());
    
    // w_i = 2 / (N(N+1) [P_N(x_i)]^2 )
    for (int i = 0; i < x.size(); ++i){
        w(i) = 2.0 / (N*(N+1)*std::pow(P(i),2));
    }

    // Returns the computed points and weights
    return;
}
