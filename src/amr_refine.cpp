#include "hdg/amr_refine.h"

AmrNode amr_refine(const AmrPredicate& cond, const AmrNode& amr) {
    if (std::holds_alternative<AmrLeaf>(amr.data)) {
        // Leaf cell — apply the predicate.
        const AmrLeaf& cell = std::get<AmrLeaf>(amr.data);
        if (cond(cell.bbox)) {
            // Subdivide into four child leaves.
            const double x0 = cell.bbox(0), y0 = cell.bbox(1);
            const double x2 = cell.bbox(2), y2 = cell.bbox(3);
            const double x1 = (x0 + x2) / 2.0;
            const double y1 = (y0 + y2) / 2.0;

            AmrSubdivided sub;
            sub.SW = std::make_unique<AmrNode>(AmrNode{AmrLeaf{Eigen::Vector4d(x0, y0, x1, y1),0}});
            sub.SE = std::make_unique<AmrNode>(AmrNode{AmrLeaf{Eigen::Vector4d(x1, y0, x2, y1),0}});
            sub.NW = std::make_unique<AmrNode>(AmrNode{AmrLeaf{Eigen::Vector4d(x0, y1, x1, y2),0}});
            sub.NE = std::make_unique<AmrNode>(AmrNode{AmrLeaf{Eigen::Vector4d(x1, y1, x2, y2),0}});
            return AmrNode{std::move(sub)};
        }
        // Predicate failed — return the leaf unchanged.
        return AmrNode{cell};
    }

    // Already subdivided — recurse into each child.
    const AmrSubdivided& sub = std::get<AmrSubdivided>(amr.data);
    AmrSubdivided out;
    out.SW = std::make_unique<AmrNode>(amr_refine(cond, *sub.SW));
    out.SE = std::make_unique<AmrNode>(amr_refine(cond, *sub.SE));
    out.NW = std::make_unique<AmrNode>(amr_refine(cond, *sub.NW));
    out.NE = std::make_unique<AmrNode>(amr_refine(cond, *sub.NE));
    return AmrNode{std::move(out)};
}