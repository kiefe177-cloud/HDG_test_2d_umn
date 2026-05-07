#pragma once

#include <Eigen/Dense>
#include <variant>
#include <memory>
#include <functional>

// Forward-declare so the variant can refer to it.
struct AmrNode;

struct AmrLeaf {
    Eigen::Vector4d bbox;
    int id = 0;


    AmrLeaf() = default;

    AmrLeaf(double x0, double y0, double x2, double y2) 
        : bbox(x0, y0, x2, y2), id(0) {}


    AmrLeaf(const Eigen::Vector4d& b, int i = 0) 
        : bbox(b), id(i) {}
};

// A subdivided cell: four children, each itself an AmrNode.
// unique_ptr is needed because AmrNode contains AmrSubdivided which
// contains AmrNode — would be infinite size without indirection.
struct AmrSubdivided {
    std::unique_ptr<AmrNode> SW;
    std::unique_ptr<AmrNode> SE;
    std::unique_ptr<AmrNode> NW;
    std::unique_ptr<AmrNode> NE;
};

// A node is either a leaf or a subdivided cell. Move-only.
struct AmrNode {
    std::variant<AmrLeaf, AmrSubdivided> data;
};

// Predicate: takes a leaf bounding box [x0, y0, x2, y2] and returns
// true if the cell should be refined.
using AmrPredicate = std::function<bool(const Eigen::Vector4d&)>;

AmrNode amr_refine(const AmrPredicate& cond, const AmrNode& amr);