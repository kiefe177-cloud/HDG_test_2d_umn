#include <iostream>
#include <Eigen/Dense>
#include <cassert>

#include "hdg/ndgrid.h"

int main(){
    // Test ndgrid

    Eigen::VectorXd x(2);
    x << 1,2;

    Eigen::VectorXd y(3);
    y << 3,4,5;

    auto [X, Y] = ndgrid(x,y);

    std::cout << "--- X Grid ---\n" << X << "\n\n";
    std::cout << "--- Y Grid ---\n" << Y << "\n";

    assert(X(0,0) == 1);
    assert(X(1,0) == 2);
    
    assert(X(0,1) == 1);
    assert(X(1,1) == 2);
    
    assert(X(0,2) == 1);
    assert(X(1,2) == 2);

    assert(Y(0,0) == 3);
    assert(Y(1,0) == 3);

    assert(Y(0,1) == 4);
    assert(Y(1,1) == 4);
    
    assert(Y(0,2) == 5);
    assert(Y(1,2) == 5);

    std::cout << "All assertions passed!" << std::endl;
    return 0;
}
