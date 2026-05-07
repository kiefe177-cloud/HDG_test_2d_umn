#include "hdg/jacobiP.h"

Eigen::VectorXd jacobiP(int N, double alpha, double beta, const Eigen::VectorXd &x) {
    // Computes the N-th order Jacobi Polynomial P_N^(alpha,beta)(x)
    
    // 1. Initialize variables 
    int size = x.size();
    Eigen::VectorXd P(size);

    // 2. Handle base cases
    if (N == 0) {
        P.setOnes();
        return P;
    }

    // 3. Recurrence relation
    Eigen::VectorXd P_nm1 = Eigen::VectorXd::Ones(size); // P_0^(alpha,beta)(x)
    Eigen::VectorXd P_n = 0.5 * (alpha - beta + (alpha + beta + 2)*x.array()).matrix(); // P_1^(alpha,beta)(x)

    // If N=1, return P_1
    if (N == 1) {
        return P_n;
    }   

    // 4. Iteratively compute P_n using the recurrence relation
    // a_1 P_(n+1)(x) = (a_2 + a_3 x) P_n(x) - a_4 P_(n-1)(x)
    for (int n = 1; n<N; ++n) {

        double a1 = 2.0 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta);
        double a2 = (2 * n + alpha + beta + 1) * (alpha*alpha - beta*beta);
        double a3 = (2 * n + alpha + beta) * (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2);
        double a4 = 2.0 * (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2);

        Eigen::VectorXd P_np1 = ((a2 + a3 * x.array()).matrix().cwiseProduct(P_n) - a4 * P_nm1) / a1;

        P_nm1 = P_n;
        P_n = P_np1;
    }

    return P_n;
}
