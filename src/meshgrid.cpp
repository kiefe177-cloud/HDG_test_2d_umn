#include "hdg/meshgrid.h"
#include "hdg/ndgrid.h"

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // MATLAB: [X, Y] = meshgrid(x, y)
    // is equivalent to: [Y_nd, X_nd] = ndgrid(y, x), then X = X_nd, Y = Y_nd
    auto [Y_out, X_out] = ndgrid(y, x);
    return {X_out, Y_out};
}