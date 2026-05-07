// interp1.h
#pragma once
#include <Eigen/Dense>
#include <limits>

enum class InterpMethod { Linear, Spline };

// Sentinel: pass this as `extrap` to mean "use the spline's natural extrapolation"
// (equivalent to MATLAB's 'extrap' keyword).
inline constexpr double kExtrapNatural = std::numeric_limits<double>::infinity();

Eigen::VectorXd interp1(const Eigen::VectorXd& x,
                        const Eigen::VectorXd& y,
                        const Eigen::VectorXd& xq,
                        InterpMethod method = InterpMethod::Spline,
                        double extrap = std::numeric_limits<double>::quiet_NaN());