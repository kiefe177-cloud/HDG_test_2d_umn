// hdg/mesh_utils.h
#pragma once

#include <Eigen/Dense>
#include <cstdlib>   // for std::abs

// Find the local face index (0-based) within element `elem` whose absolute
// global face index equals `face_global`. Returns -1 if not found.
//
// EtoF(f, elem) = signed global face index (sign encodes orientation).
inline int find_local_face(const Eigen::MatrixXi& EtoF,
                           int elem, int face_global,
                           int faces_per_elem)
{
    for (int f = 0; f < faces_per_elem; ++f) {
        if (std::abs(EtoF(f, elem)) == face_global) return f;
    }
    return -1;
}