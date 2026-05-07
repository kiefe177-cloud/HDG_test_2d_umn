#include "hdg/msh_connect.h"
#include "hdg/amr_connect.h"
#include "hdg/amr_number.h"

#include <vector>
#include <map>
#include <Eigen/Dense>
// =======================================
// ANONYMOUS NAMESPACE: Local helpers only
// =======================================
namespace{

    Eigen::MatrixXi get_unique_columns(const Eigen::MatrixXi& FtoE, std::vector<int>& ic) {
        std::map<std::vector<int>, int> seen_faces;
        std::vector<std::vector<int>> unique_cols;
        
        ic.resize(FtoE.cols());
        int unique_count = 0;

        for (int j = 0; j < FtoE.cols(); ++j) {
            std::vector<int> col_data(FtoE.rows());
            for (int i = 0; i < FtoE.rows(); ++i) {
                col_data[i] = FtoE(i, j);
            }

            auto it = seen_faces.find(col_data);
            if (it == seen_faces.end()) {
                seen_faces[col_data] = unique_count;
                ic[j] = unique_count; 
                unique_cols.push_back(col_data);
                unique_count++;
            } else {
                ic[j] = it->second;
            }
        }

        Eigen::MatrixXi FtoE_unique(FtoE.rows(), unique_count);
        for (int j = 0; j < unique_count; ++j) {
            for (int i = 0; i < FtoE.rows(); ++i) {
                FtoE_unique(i, j) = unique_cols[j][i];
            }
        }

        return FtoE_unique;
    }
}

Connect msh_connect(AmrNode& amr){
    Connect result;
    
    int counter = 0;

    amr_number(amr, counter);

    std::vector<std::pair<int, int>> FtoE_vec;
    std::vector<std::array<int, 12>> EtoF_vec;
    
    // Call the newly exported function
    amr_connect(-1, -2, -3, -4, amr, FtoE_vec, EtoF_vec);

    // Convert to Eigen matrices
    result.FtoE.resize(2, FtoE_vec.size());
    for (size_t i = 0; i < FtoE_vec.size(); ++i) {
        result.FtoE(0, i) = FtoE_vec[i].first;
        result.FtoE(1, i) = FtoE_vec[i].second;
    }

    result.EtoF.resize(12, EtoF_vec.size());
    for (size_t i = 0; i < EtoF_vec.size(); ++i) {
        for (int j = 0; j < 12; ++j) {
            result.EtoF(j, i) = EtoF_vec[i][j];
        }
    }

    // 3. Combine duplicate faces
    std::vector<int> ic;
    result.FtoE = get_unique_columns(result.FtoE, ic);

    // 4. Update the EtoF matrix with new unique indices
    for (int i2 = 0; i2 < result.EtoF.cols(); ++i2) {
        for (int i1 = 0; i1 < result.EtoF.rows(); ++i1) {
            int old_index = result.EtoF(i1, i2);
            if (old_index >= 0) {
                result.EtoF(i1, i2) = ic[old_index]; 
            }
        }
    }

    return result;
}