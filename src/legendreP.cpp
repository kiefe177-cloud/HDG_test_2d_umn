#include "hdg/legendreP.h"

Eigen::VectorXd legendreP(int n, const Eigen::VectorXd& x) {
    // Computes the n-th order Legendre polynomial P_n at points x using recurrence relation

    // 1. Handle base cases
    if (n == 0){
        return Eigen::VectorXd::Ones(x.size());
    } else if (n == 1){
        return x;
    } else {
        // 2. Use recurrence relation to compute P_n(x)
        Eigen::VectorXd p_prev = Eigen::VectorXd::Ones(x.size());

        Eigen::VectorXd p_curr = x;

        Eigen::VectorXd p_next(x.size());

        for (int k = 1; k < n; ++k){
            // Recurrence Formula:
            // (k+1) P_{k+1}(x) = (2k + 1) x P_k(x) - k P_{k-1}(x) / (k + 1)
            p_next = ( (2.0 * k + 1) *x.array() * p_curr.array() - k * p_prev.array() ) / (k + 1.0);
            
            p_prev = p_curr;
            p_curr = p_next;
        }
    
        return p_curr;
    }
}
