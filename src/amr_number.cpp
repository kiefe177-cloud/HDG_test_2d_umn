#include "hdg/amr_number.h"

void amr_number(AmrNode& amr, int& n){

    if (std::holds_alternative<AmrLeaf>(amr.data)){
        std::get<AmrLeaf>(amr.data).id = n;
        n = n + 1;
    }

    else {
        auto& sub = std::get<AmrSubdivided>(amr.data);

        if (sub.SW) amr_number(*(sub.SW), n);
        if (sub.SE) amr_number(*(sub.SE), n);
        if (sub.NW) amr_number(*(sub.NW), n);
        if (sub.NE) amr_number(*(sub.NE), n);
    }
}