#include <cmath>
#include <vector>

#include "hdg/sem2.h"
#include "hdg/sem1.h"
#include "hdg/GLQuad.h"
#include "hdg/jacobiP.h"

#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;

// Helper Functions
// Vandermonde1D
MatrixXd Vandermonde1D(int N, const VectorXd& r) {
    MatrixXd V(N + 1, N + 1);
    // Fill Vandermonde Matrix
    // sqrt(j+0.5) * P_j(r)
    for (int j = 0; j <= N; ++j) {
        V.col(j) = std::sqrt(j + 0.5) * jacobiP(j, 0.0, 0.0, r);
    }
    return V;
}

// Projection Operators for subdivisions
void proj(int p, VectorXd& r, MatrixXd& P12, MatrixXd& P13, MatrixXd& P21, MatrixXd& P31){

    // 1. Compute Vandermonde Matrix
    MatrixXd V = Vandermonde1D(p, r);

    // 2. Compute inverse mass matrix
    // M = (V^T V)^-1
    // M^-1 = V V^T
    MatrixXd VVt = V * V.transpose();
    MatrixXd M = VVt.inverse();

    // 3. Compute projection matrices
    VectorXd r2 = 0.5 * (r.array() - 1.0);
    VectorXd r3 = 0.5 * (r.array() + 1.0);

    // Vandermonde matrices at r2 and r3
    MatrixXd V2 = Vandermonde1D(p, r2);
    MatrixXd V3 = Vandermonde1D(p, r3);

    P12 = V2 * V.inverse();
    P13 = V3 * V.inverse();

    // 4. Compute transposed projection matrices
    P21 = VVt * (P12.transpose()*M);
    P31 = VVt * (P13.transpose()*M);
}

Sem2Op sem2(int p){
    // Computes the 2D Spectral Element Semi-discrete operators

    // 1. Initialize output struct
    Sem2Op op;
    
    // 2. Call sem1 to get 1D operators
    Sem1Op op1 = sem1(p);

    // 3. Construct 2D operators via Kronecker products
    // Mass Matrix
    op.M = kroneckerProduct(op1.M, op1.M);

    // Stiffness Matrices
    op.S.resize(2);
    op.S[0] = kroneckerProduct(MatrixXd(op1.M), op1.S);
    op.S[1] = kroneckerProduct(op1.S, MatrixXd(op1.M)); 

    VectorXd r,w;
    GLQuad(p, r, w);

    MatrixXd P12, P13, P21, P31;
    proj(p, r, P12, P13, P21, P31);

    int N = p + 1;
    SparseMatrix<double> I(N,N);
    I.setIdentity();

    SparseMatrix<double> Sm(1,N);
    Sm.insert(0,0) = 1.0;

    SparseMatrix<double> Sp(1,N);
    Sp.insert(0,p) = 1.0;

    SparseMatrix<double> SmT = Sm.transpose();
    SparseMatrix<double> SpT = Sp.transpose();

    // Helper to compute kron(A,B) and push to L
    auto add_L = [&](const auto& A, const auto& B, const MatrixXd* P = nullptr) {
        SparseMatrix<double> K = kroneckerProduct(A, B);
        if (P) {
            // If P is provided, multiply: K * P
            MatrixXd denseRes = K * (*P);
            op.L.push_back(denseRes.sparseView()); // Convert to sparse and push
        } else {
            op.L.push_back(K);
        }
    };

    // Lifting Operators
    add_L(op1.M, SmT);          // L0
    add_L(op1.M, SpT);          // L1
    add_L(SmT, op1.M);          // L2
    add_L(SpT, op1.M);          // L3

    add_L(op1.M, SmT, &P21);    // L4
    add_L(op1.M, SpT, &P21);    // L5
    add_L(SmT, op1.M, &P21);    // L6
    add_L(SpT, op1.M, &P21);    // L7

    add_L(op1.M, SmT, &P31);    // L8
    add_L(op1.M, SpT, &P31);    // L9
    add_L(SmT, op1.M, &P31);    // L10
    add_L(SpT, op1.M, &P31);    // L11

    // Helper to compute kron(A,B) and push to T
    auto add_T = [&](const auto& A, const auto& B, const MatrixXd* P = nullptr) {
        SparseMatrix<double> K = kroneckerProduct(A, B);
        if (P) {
            // P * K
            MatrixXd denseRes = (*P) * K;
            op.T.push_back(denseRes.sparseView());
        } else {
            op.T.push_back(K);
        }
    };

    // Trace Operators
    add_T(I, Sm);          // T0
    add_T(I, Sp);          // T1
    add_T(Sm, I);          // T2
    add_T(Sp, I);          // T3

    add_T(I, Sm, &P12);    // T4
    add_T(I, Sp, &P12);    // T5
    add_T(Sm, I, &P12);    // T6
    add_T(Sp, I, &P12);    // T7

    add_T(I, Sm, &P13);    // T8
    add_T(I, Sp, &P13);    // T9
    add_T(Sm, I, &P13);    // T10
    add_T(Sp, I, &P13);    // T11

    return op;
}


