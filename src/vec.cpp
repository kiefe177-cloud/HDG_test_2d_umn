#include "hdg/vec.h"

Eigen::VectorXd vec(const Eigen::MatrixXd& u) {
    // In Matlab, v = reshape(u, [], 1);
    // In Eigen, we can use Map to achieve the same effect without copying data.
    return Eigen::Map<const Eigen::VectorXd>(u.data(), u.size());
}
