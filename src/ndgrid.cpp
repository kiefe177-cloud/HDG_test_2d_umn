#include "hdg/ndgrid.h"

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ndgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // Returns grid in (i,j)/(row,column) form (ndgrid from matlab)

    // 1. Get sizes of input vectors
    long x_size = x.size();
    long y_size = y.size(); 

    // 2. Replicate vectors to form grid matrices
    Eigen::MatrixXd X = x.replicate(1,y_size);
    Eigen::MatrixXd Y = y.transpose().replicate(x_size,1);

    // 3. Return the grid matrices as a pair
    return {X,Y};
}
