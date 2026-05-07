#include <stdexcept>
#include <iostream>
#include <algorithm>

#include "hdg/Spline.h"

void Spline::fit(const Eigen::VectorXd& x, const Eigen::VectorXd& y){
    long n = x.size();
    if (n != y.size()) throw std::invalid_argument("x and y must have same size");
    if (n < 2) throw std::invalid_argument("Need at least 2 points for spline");

    // 1. Prepare Data
    x_knots.resize(n);
    y_vals.resize(n);
    Eigen::VectorXd::Map(&x_knots[0], n) = x;
    Eigen::VectorXd::Map(&y_vals[0], n) = y;

    // 2. Compute Intervals and Divided Differences
    // dx[i] = h_i = x_{i+1} - x_i
    // divdif[i] = \delta_i = (y_{i+1} - y_i) / h_i
    Eigen::VectorXd dx(n-1);
    Eigen::VectorXd divdif(n-1);
    for (int i = 0; i < n - 1; ++i){
        dx[i] = x[i + 1] - x[i];
        if (dx[i] <= 0) throw std::invalid_argument("x must be strictly increasing");
        divdif[i] = (y[i + 1] - y[i]) / dx[i];
    }

    // We declare 's' here so both the manual case (n=3) and Thomas case (n>3) can fill it.
    Eigen::VectorXd s(n);

    // 3. Boundary Conditions & Special Cases
    if (n == 2){
        // Line case: slopes are constant
        b_coefs = {divdif[0]};
        c_coefs = {0.0};
        d_coefs = {0.0};
        return;
    }
    else if (n == 3){
        // Parabola Case: Uniquely determined by 3 points
        // We manually calculate slopes w/o solving the matrix
        double w = (divdif[1] - divdif[0]) / (x[2] - x[0]);
        s[0] = divdif[0] - w * dx[0];
        s[1] = divdif[1] - w * dx[1];
        s[2] = divdif[1] + w * dx[1];
    }
    else {
        // General Case (n > 3): Not-a-Knot Boundary Conditions, solved via Thomas algorithm.
        //
        // The system is tridiagonal:
        //     b_di(i) * s(i)  + c_up(i) * s(i+1)                  = rhs(i)    (row 0)
        //     a_lo(i) * s(i-1) + b_di(i) * s(i) + c_up(i) * s(i+1) = rhs(i)   (interior)
        //     a_lo(i) * s(i-1) + b_di(i) * s(i)                    = rhs(i)   (row n-1)
        //
        // We build the three diagonals + rhs directly — no dense matrix ever allocated.

        Eigen::VectorXd a_lo = Eigen::VectorXd::Zero(n);  // sub-diagonal,   a_lo(i) couples s(i-1)
        Eigen::VectorXd b_di = Eigen::VectorXd::Zero(n);  // main diagonal,  b_di(i) couples s(i)
        Eigen::VectorXd c_up = Eigen::VectorXd::Zero(n);  // super-diagonal, c_up(i) couples s(i+1)
        Eigen::VectorXd rhs  = Eigen::VectorXd::Zero(n);

        // Interior continuity equations: y''_left(x_i) = y''_right(x_i)
        //   h_i*s_{i-1} + 2(h_i + h_{i-1})*s_i + h_{i-1}*s_{i+1} = 3(h_i*div_{i-1} + h_{i-1}*div_i)
        for (int i = 1; i < n - 1; ++i){
            a_lo(i) = dx[i];
            b_di(i) = 2.0 * (dx[i] + dx[i-1]);
            c_up(i) = dx[i-1];
            rhs(i)  = 3.0 * (dx[i] * divdif[i-1] + dx[i-1] * divdif[i]);
        }

        // Start Boundary (not-a-knot)
        double x20 = x[2] - x[0];
        b_di(0) = dx[1];
        c_up(0) = x20;
        rhs(0)  = ((dx[0] + 2.0 * x20) * dx[1] * divdif[0] + dx[0] * dx[0] * divdif[1]) / x20;

        // End Boundary (not-a-knot)
        double xn = x[n-1] - x[n-3];
        a_lo(n-1) = xn;
        b_di(n-1) = dx[n-2];
        rhs(n-1)  = (dx[n-2] * dx[n-2] * divdif[n-3] + (2.0 * xn + dx[n-2]) * dx[n-2] * divdif[n-2]) / xn;

        // ---- Thomas algorithm: O(n) tridiagonal solve ----
        //
        // Forward sweep: eliminate the sub-diagonal in place.
        // After this loop, the system is upper-bidiagonal.
        for (int i = 1; i < n; ++i) {
            double m = a_lo(i) / b_di(i - 1);
            b_di(i) -= m * c_up(i - 1);
            rhs(i)  -= m * rhs(i - 1);
        }

        // Back substitution: solve from the bottom row up.
        s(n - 1) = rhs(n - 1) / b_di(n - 1);
        for (int i = n - 2; i >= 0; --i) {
            s(i) = (rhs(i) - c_up(i) * s(i + 1)) / b_di(i);
        }
    }

    // 4. Convert Hermite Form to Power Basis Form
    // Polynomial: y(x) = y_i + b*(x-xi) + c*(x-xi)^2 + d*(x-xi)^3
    b_coefs.resize(n-1);
    c_coefs.resize(n-1);
    d_coefs.resize(n-1);

    for (int i = 0; i < n - 1; ++i){
        b_coefs[i] = s[i];          // simple slope
        double inv_dx = 1.0 / dx[i];

        // c = (3*secant_i - 2*s_i - s_{i+1})/ dx
        c_coefs[i] = (3.0 * divdif[i] - 2.0 * s[i] - s[i+1]) * inv_dx;

        // d = (s_{i+1} - 2*secant_i + s_i)/ dx^2
        d_coefs[i] = (s[i] + s[i+1] - 2.0 * divdif[i]) * (inv_dx * inv_dx);
    }
}

double Spline::eval(double query_x) const {
    if (x_knots.empty()) return 0.0;

    // 1. Extrapolation
    // First polynomial logic
    if (query_x <= x_knots.front()) return y_vals.front() + b_coefs.front() * (query_x - x_knots.front());

    // 2. Binary Search to find correct interval
    auto it = std::lower_bound(x_knots.begin(), x_knots.end(), query_x);
    int i = std::distance(x_knots.begin(), it) - 1;

    // Clamps
    if (i < 0) i = 0;
    if (i >= (int)b_coefs.size()) i = b_coefs.size() - 1;

    // Evaluate the Cubic Polynomial
    // y = y_i + b*dx + c*dx^2 + d*dx^3
    double dx = query_x - x_knots[i];
    return y_vals[i] + b_coefs[i] * dx + c_coefs[i] * dx * dx + d_coefs[i] * dx * dx * dx;
}

Eigen::VectorXd Spline::eval_vector(const Eigen::VectorXd& query_x_vec) const {
    Eigen::VectorXd result(query_x_vec.size());
    for (int i = 0; i < query_x_vec.size(); ++i){
        result[i] = eval(query_x_vec[i]);
    }
    return result;
}