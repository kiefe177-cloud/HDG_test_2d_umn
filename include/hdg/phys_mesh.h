#pragma once

#include "hdg/comp_space_metrics.h"
#include "hdg/base_grid_metrics.h"

struct PhysMesh {
    GeomMetrics     geom;
    BaseGridMetrics base_grid;
};