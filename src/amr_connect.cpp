#include "hdg/amr_connect.h"
#include <variant>

// ==========================================
// ANONYMOUS NAMESPACE: Private Helpers
// ==========================================
namespace {

    struct Neighbor {
        const AmrNode* node = nullptr;
        int bound_id = 0;

        Neighbor(const AmrNode* n) : node(n), bound_id(0) {}
        Neighbor(int id) : node(nullptr), bound_id(id) {}

        bool is_leaf() const {
            if (node == nullptr) return true; 
            return std::holds_alternative<AmrLeaf>(node->data);
        }

        int get_id() const {
            if (node == nullptr) return bound_id;
            return std::get<AmrLeaf>(node->data).id;
        }
    };

    Neighbor SW_of(Neighbor n) {
        if (n.is_leaf()) return n;
        return Neighbor(std::get<AmrSubdivided>(n.node->data).SW.get());
    }
    Neighbor SE_of(Neighbor n) {
        if (n.is_leaf()) return n;
        return Neighbor(std::get<AmrSubdivided>(n.node->data).SE.get());
    }
    Neighbor NW_of(Neighbor n) {
        if (n.is_leaf()) return n;
        return Neighbor(std::get<AmrSubdivided>(n.node->data).NW.get());
    }
    Neighbor NE_of(Neighbor n) {
        if (n.is_leaf()) return n;
        return Neighbor(std::get<AmrSubdivided>(n.node->data).NE.get());
    }

    void amr_connect_rec(Neighbor tW, Neighbor tE, Neighbor tS, Neighbor tN, const AmrNode& t,
                         std::vector<std::pair<int, int>>& FtoE,
                         std::vector<std::array<int, 12>>& EtoF) {
        
        if (std::holds_alternative<AmrLeaf>(t.data)) {
            std::array<int, 12> leaf_EtoF;
            leaf_EtoF.fill(-1); 
            int t_id = std::get<AmrLeaf>(t.data).id;

            if (tW.is_leaf()) {
                FtoE.push_back({tW.get_id(), t_id});
                leaf_EtoF[0] = FtoE.size() - 1; 
            } else {
                FtoE.push_back({SE_of(tW).get_id(), t_id});
                leaf_EtoF[4] = FtoE.size() - 1;
                FtoE.push_back({NE_of(tW).get_id(), t_id});
                leaf_EtoF[8] = FtoE.size() - 1;
            }

            if (tE.is_leaf()) {
                FtoE.push_back({t_id, tE.get_id()});
                leaf_EtoF[1] = FtoE.size() - 1;
            } else {
                FtoE.push_back({t_id, SW_of(tE).get_id()});
                leaf_EtoF[5] = FtoE.size() - 1;
                FtoE.push_back({t_id, NW_of(tE).get_id()});
                leaf_EtoF[9] = FtoE.size() - 1;
            }

            if (tS.is_leaf()) {
                FtoE.push_back({tS.get_id(), t_id});
                leaf_EtoF[2] = FtoE.size() - 1;
            } else {
                FtoE.push_back({NW_of(tS).get_id(), t_id});
                leaf_EtoF[6] = FtoE.size() - 1;
                FtoE.push_back({NE_of(tS).get_id(), t_id});
                leaf_EtoF[10] = FtoE.size() - 1;
            }

            if (tN.is_leaf()) {
                FtoE.push_back({t_id, tN.get_id()});
                leaf_EtoF[3] = FtoE.size() - 1;
            } else {
                FtoE.push_back({t_id, SW_of(tN).get_id()});
                leaf_EtoF[7] = FtoE.size() - 1;
                FtoE.push_back({t_id, SE_of(tN).get_id()});
                leaf_EtoF[11] = FtoE.size() - 1;
            }

            EtoF.push_back(leaf_EtoF);

        } else {
            const auto& sub = std::get<AmrSubdivided>(t.data);
            
            amr_connect_rec(SE_of(tW), Neighbor(sub.SE.get()), NW_of(tS), Neighbor(sub.NW.get()), *sub.SW, FtoE, EtoF);
            amr_connect_rec(Neighbor(sub.SW.get()), SW_of(tE), NE_of(tS), Neighbor(sub.NE.get()), *sub.SE, FtoE, EtoF);
            amr_connect_rec(NE_of(tW), Neighbor(sub.NE.get()), Neighbor(sub.SW.get()), SW_of(tN), *sub.NW, FtoE, EtoF);
            amr_connect_rec(Neighbor(sub.NW.get()), NW_of(tE), Neighbor(sub.SE.get()), SE_of(tN), *sub.NE, FtoE, EtoF);
        }
    }

} // end anonymous namespace

// ==========================================
// EXPOSED PUBLIC FUNCTION
// ==========================================
void amr_connect(int bW, int bE, int bS, int bN, const AmrNode& t,
                 std::vector<std::pair<int, int>>& FtoE,
                 std::vector<std::array<int, 12>>& EtoF) {
    
    // Kick off the recursion using our private helper
    amr_connect_rec(Neighbor(bW), Neighbor(bE), Neighbor(bS), Neighbor(bN), t, FtoE, EtoF);
}