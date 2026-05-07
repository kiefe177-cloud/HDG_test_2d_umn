#pragma once

#include "hdg/amr_refine.h"
#include <vector>
#include <array>
#include <utility>

void amr_connect(int bW, int bE, int bS, int bN, const AmrNode& t,
                    std::vector<std::pair<int, int>>& FtoE,
                    std::vector<std::array<int, 12>>& EtoF);