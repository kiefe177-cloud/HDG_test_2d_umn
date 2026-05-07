#include "hdg/amr_flatten.h"
#include <vector>

namespace {

    // Fast recursive helper using std::vector
    void flatten_recursive(const AmrNode& node, std::vector<AmrLeaf>& out) {
        
        if (std::holds_alternative<AmrLeaf>(node.data)) {
            // It's a leaf, add it to the list
            out.push_back(std::get<AmrLeaf>(node.data));
            
        } else if (std::holds_alternative<AmrSubdivided>(node.data)) {
            // It's a branch, recurse into children
            const auto& sub = std::get<AmrSubdivided>(node.data);
            if (sub.SW) flatten_recursive(*(sub.SW), out);
            if (sub.SE) flatten_recursive(*(sub.SE), out);
            if (sub.NW) flatten_recursive(*(sub.NW), out);
            if (sub.NE) flatten_recursive(*(sub.NE), out);
        }
    }

} // end anonymous namespace


// Exposed function
Eigen::MatrixXd amr_flatten(const AmrNode& amr) {
    
    // 1. Gather all leaves blazingly fast into a vector
    std::vector<AmrLeaf> leaves;
    flatten_recursive(amr, leaves);

    // 2. Allocate the Eigen matrix EXACTLY ONCE based on the final vector size.
    // Assuming you want the 4 coordinates + the 1 ID = 5 columns.
    Eigen::MatrixXd elem_matrix(static_cast<int>(leaves.size()), 4); 
    
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i) {
        // Copy the bbox and the ID into the matrix row
        elem_matrix.row(i) << leaves[i].bbox.transpose();
    }
    
    return elem_matrix;
}