#pragma once
#include <Eigen/Dense>
#include <vector>

class Spline{
public:

    void fit(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

    double eval(double query_x) const;

    Eigen::VectorXd eval_vector(const Eigen::VectorXd& query_x_vec) const;

    Eigen::VectorXd interp(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& xq){
        Spline s;
        s.fit(x,y);
        return s.eval_vector(xq);
    }
private:

std::vector<double> x_knots;
std::vector<double> y_vals;
std::vector<double> b_coefs;
std::vector<double> c_coefs;
std::vector<double> d_coefs;
};