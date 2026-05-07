#include <iostream>
#include <cmath>
#include <cassert>

#include "hdg/legendreP.h"
#include "hdg/jacobiZerosGW.h"
#include "hdg/GLQuad.h"

int main(){
    // 1. The Function Call
    int N = 4;
    Eigen::VectorXd x, w;
    GLQuad(N, x, w);

    // 2. The Expected Results
    Eigen::VectorXd expected_x(x.size());
    expected_x << -1.0, -0.6547, 0.0, 0.6547, 1.0;
    Eigen::VectorXd expected_w(w.size());
    expected_w << .1, 0.5444, 0.7111, 0.5444, 0.1;

    // 3. Verification
    double tol = 1e-4;
    for(int i = 0; i < x.size(); ++i){
        assert(std::abs(x(i) - expected_x(i)) < tol && "GL points do not match expected values.");
        assert(std::abs(w(i) - expected_w(i)) < tol && "GL weights do not match expected values.");
    }

    std::cout << "[PASS] GLQuad Test for N=4" << std::endl;

    // Second Test: N=8
    N = 8;  
    GLQuad(N, x, w);
    Eigen::VectorXd expected_x2(x.size());
    expected_x2 << -1.0, -0.8998, -0.6772, -0.3631, 0.0, 0.3631, 0.6772, 0.8998, 1.0;
    Eigen::VectorXd expected_w2(w.size());
    expected_w2 << 0.0278, 0.1655, 0.2745, 0.3464, 0.3715, 0.3464, 0.2745, 0.1655, 0.0278;

    for(int i = 0; i < x.size(); ++i){
        assert(std::abs(x(i) - expected_x2(i)) < tol && "GL points do not match expected values for N=8.");
        assert(std::abs(w(i) - expected_w2(i)) < tol && "GL weights do not match expected values for N=8.");
    }

    std::cout << "[PASS] GLQuad Test for N=8" << std::endl;
    return 0;
}
