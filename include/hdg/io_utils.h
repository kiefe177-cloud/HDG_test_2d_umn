#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <string>

inline void save_csv(const std::string& path, const Eigen::MatrixXd& mat) {
    std::ofstream file(path);
    static const Eigen::IOFormat fmt(
        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    file << mat.format(fmt);
}