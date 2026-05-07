#include <iostream>
#include <cassert>
#include "hdg/sd.h" // Includes our new function

int main() {
    // 1. Setup Input: Vector v = [2, 9, 4]
    Eigen::VectorXd v(3);
    v << 2.0, 9.0, 4.0;

    // 2. Run the function
    Eigen::SparseMatrix<double> A = sd(v);

    // 3. Verify Results
    // A should look like:
    // 1  0  0
    // 0  2  0
    // 0  0  3
    
    // Check Main Diagonal
    // .coeff(row, col) gets the value at that position
    if (A.coeff(0,0) != 2.0) { std::cout << "Fail: (0,0) is wrong\n"; return 1; }
    if (A.coeff(1,1) != 9.0) { std::cout << "Fail: (1,1) is wrong\n"; return 1; }
    if (A.coeff(2,2) != 4.0) { std::cout << "Fail: (2,2) is wrong\n"; return 1; }

    // Check Off-Diagonal (Should be zero)
    if (A.coeff(0,1) != 0.0) { std::cout << "Fail: (0,1) is not zero\n"; return 1; }

    // Check structure properties
    if (A.nonZeros() != 3) { std::cout << "Fail: Sparse matrix has wrong number of non-zeros\n"; return 1; }

    std::cout << "[PASS] Sparse Diagonal (sd) Test" << std::endl;
    // Optional: Print the matrix to see it visually
    // Eigen prints sparse matrices by listing non-zero indices: (row, col) value
    std::cout << "Matrix A:\n" << A << std::endl;

    return 0;
}
