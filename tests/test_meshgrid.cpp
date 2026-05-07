#include <iostream>
#include <Eigen/Dense>
#include <cassert>

#include "hdg/meshgrid.h"

int main(){
    // Test meshgrid
    // MATLAB: [X, Y] = meshgrid([1,2], [3,4,5])
    // X = [1 2;     Y = [3 3;
    //      1 2;          4 4;
    //      1 2]          5 5]

    Eigen::VectorXd x(2);
    x << 1, 2;

    Eigen::VectorXd y(3);
    y << 3, 4, 5;

    auto [X, Y] = meshgrid(x, y);

    std::cout << "--- X Grid ---\n" << X << "\n\n";
    std::cout << "--- Y Grid ---\n" << Y << "\n";

    // X: columns vary with x, rows are constant
    assert(X(0,0) == 1);
    assert(X(0,1) == 2);

    assert(X(1,0) == 1);
    assert(X(1,1) == 2);

    assert(X(2,0) == 1);
    assert(X(2,1) == 2);

    // Y: rows vary with y, columns are constant
    assert(Y(0,0) == 3);
    assert(Y(0,1) == 3);

    assert(Y(1,0) == 4);
    assert(Y(1,1) == 4);

    assert(Y(2,0) == 5);
    assert(Y(2,1) == 5);

    // Check dimensions: meshgrid(2-vec, 3-vec) -> 3x2 matrices
    assert(X.rows() == 3);
    assert(X.cols() == 2);
    assert(Y.rows() == 3);
    assert(Y.cols() == 2);

    std::cout << "All assertions passed!" << std::endl;
    return 0;
}