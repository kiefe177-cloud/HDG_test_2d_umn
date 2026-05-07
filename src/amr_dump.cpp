#include "hdg/amr_dump.h"

#include <fstream>
#include <vector>

namespace {

void collect_recursive(const AmrNode& node, std::vector<AmrLeaf>& out) {
    if (std::holds_alternative<AmrLeaf>(node.data)) {
        out.push_back(std::get<AmrLeaf>(node.data));
        return;
    }
    const AmrSubdivided& sub = std::get<AmrSubdivided>(node.data);
    collect_recursive(*sub.SW, out);
    collect_recursive(*sub.SE, out);
    collect_recursive(*sub.NW, out);
    collect_recursive(*sub.NE, out);
}

} // namespace

Eigen::MatrixXd collect_amr_leaves(const AmrNode& amr) {
    std::vector<AmrLeaf> leaves;
    collect_recursive(amr, leaves);

    Eigen::MatrixXd out(static_cast<int>(leaves.size()), 4);
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i) {
        out.row(i) << leaves[i].bbox.transpose();
    }
    return out;
}

void dump_amr_leaves_csv(const std::string& path, const AmrNode& amr) {
    Eigen::MatrixXd leaves = collect_amr_leaves(amr);
    std::ofstream file(path);
    static const Eigen::IOFormat fmt(
        Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    file << leaves.format(fmt);
}