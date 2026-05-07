// interp1.cpp
#include "hdg/interp1.h"
#include "hdg/Spline.h"
#include <stdexcept>
#include <cmath>

Eigen::VectorXd interp1(const Eigen::VectorXd& x,
                        const Eigen::VectorXd& y,
                        const Eigen::VectorXd& xq,
                        InterpMethod method,
                        double extrap)
{
    if (x.size() != y.size())
        throw std::invalid_argument("interp1: x and y must have the same size");
    if (x.size() < 2)
        throw std::invalid_argument("interp1: need at least 2 sample points");

    Eigen::VectorXd out(xq.size());

    switch (method) {
        case InterpMethod::Spline: {
            Spline spline;
            spline.fit(x, y);
            for (Eigen::Index i = 0; i < xq.size(); ++i)
                out(i) = spline.eval(xq(i));
            break;
        }
        case InterpMethod::Linear:
            throw std::runtime_error("interp1: Linear method not implemented yet");
    }

    // If the caller asked for natural extrapolation, leave the spline output alone.
    if (std::isinf(extrap)) return out;

    // Otherwise, override out-of-range queries with the user-specified value.
    const double x_lo = x(0);
    const double x_hi = x(x.size() - 1);
    for (Eigen::Index i = 0; i < xq.size(); ++i) {
        if (xq(i) < x_lo || xq(i) > x_hi) {
            out(i) = extrap;
        }
    }
    return out;
}