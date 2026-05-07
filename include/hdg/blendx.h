// blendx.h
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <limits>

// Smooth Hermite step on [0,1]: clamps z, then evaluates 10t^3 - 15t^4 + 6t^5.
// Templated so it works on any Eigen array expression (1D or 2D, dense or expression).
template <typename Derived>
auto smooth01(const Eigen::ArrayBase<Derived>& z) {
    auto t = z.cwiseMax(0.0).cwiseMin(1.0).eval();   // .eval() forces a concrete array
    return (t.cube() * (10.0 - 15.0 * t + 6.0 * t.square())).eval();
}

template <typename Derived>
auto blendx(const Eigen::ArrayBase<Derived>& x, double x0, double w) {
    double safe_w = std::max(w, std::numeric_limits<double>::epsilon());
    return smooth01( (x - (x0 - 0.5 * safe_w)) / safe_w );
}