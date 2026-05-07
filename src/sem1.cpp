#include "hdg/sem1.h"
#include "hdg/sd.h"
#include "hdg/GLQuad.h"
#include "hdg/GLDeriv.h"

Sem1Op sem1(int p){
    // Computes 1D spectral Element Semi-Discrete Operators

    Sem1Op op;
    int N = p + 1;
    Eigen::VectorXd r,w;

    // 1. Compute 1D Gauss-Lobatto Quadrature points and weights
    GLQuad(p, r, w);

    // 2. Compute 1D Gauss-Lobatto Derivative matrix
    Eigen::MatrixXd Dr = GLDeriv(p);

    op.M = sd(w);
    op.S = (sd(w) * Dr).transpose();

    // 3. Compute Left and Right Boundary Operators
    op.Lm.resize(N,N);
    op.Lm.insert(0,0) = 1.0;

    op.Lp.resize(N,N);
    op.Lp.insert(p,p) = 1.0;

    return op;
}
