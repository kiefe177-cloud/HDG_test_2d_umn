#include <iostream>
#include <cassert>
#include "hdg/vec.h" 

int main() {
    // 1. Setup a 2x2 Matrix
    Eigen::MatrixXd u(2, 2);
    // Fill it manually to be sure of order
    u(0,0) = 1.0; u(0,1) = 3.0;
    u(1,0) = 2.0; u(1,1) = 4.0;

    // 2. Run the function
    Eigen::VectorXd v = vec(u);

    // 3. Verify
    // Size check: Should be 4x1
    if (v.rows() != 4 || v.cols() != 1) {
        std::cout << "Fail: Wrong dimensions. Got " << v.rows() << "x" << v.cols() << "\n";
        return 1;
    }

    // Order check: MATLAB/Eigen are Column-Major.
    // It should read down Col 1 (1, 2), then down Col 2 (3, 4).
    // Expected: [1, 2, 3, 4]
    
    if (v(0) != 1.0) { std::cout << "Fail: Index 0 is wrong\n"; return 1; }
    if (v(1) != 2.0) { std::cout << "Fail: Index 1 is wrong\n"; return 1; }
    if (v(2) != 3.0) { std::cout << "Fail: Index 2 is wrong\n"; return 1; }
    if (v(3) != 4.0) { std::cout << "Fail: Index 3 is wrong\n"; return 1; }

    std::cout << "[PASS] vec() Test" << std::endl;
    std::cout << "Original:\n" << u << "\n\nFlattened:\n" << v << std::endl;

    return 0;
}
