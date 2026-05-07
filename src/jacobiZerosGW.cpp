#include "hdg/jacobiZerosGW.h"
#include <limits> // Reuired for 'eps'
#include <cmath>  // Required for 'sqrt'

Eigen::VectorXd jacobiZerosGW(int n, double alpha, double beta) {
    // Computes the zeros of the n-th order Jacobi polynomial P_n^(alpha,beta) using the Golub-Welsch algorithm

    if (n == 0){
        // Return empty vector for n=0
        return Eigen::VectorXd(0);
    } else if (n == 1){
        Eigen::VectorXd x(1);
        x(0) = (beta - alpha)/(alpha + beta + 2.0);
        return x;
    } else {
        // Form symmetric matrix from recurrence (Golub-Welsch)

        int N = n-1;
        // Create index vector 0:N
        Eigen::VectorXd indices = Eigen::VectorXd::LinSpaced(n, 0, N);

        // Compute h1 = 2(0:N) + alpha + beta
        Eigen::VectorXd h1 = 2.0*indices.array() + alpha + beta;

        // Initialize Jacobi matrix
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n); 

        // Fill Main Diagonal
        J.diagonal() = (beta*beta - alpha*alpha)*(h1.array() + 2.0).inverse() * h1.array().inverse();

        // Fix first element if needed
        if (alpha + beta < 10.0 * std::numeric_limits<double>::epsilon()){
            J(0,0) = 0.0;
        }

        //  Fill off-diagonal
        for (int i = 0; i < n-1; ++i){

            double idx = i + 1;

            double h_j = h1(i)+2.0;

            double val = (2.0 / (h_j)) * std::sqrt((idx*(idx+alpha+beta)*(idx+alpha)*(idx+beta) / ((h_j - 1.0)*(h_j+1.0))));
            
            J(i, i+1) = val;
            J(i+1, i) = val;
        }
        
        // Compute eigenvalues
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);
        return solver.eigenvalues();
    } 
}
