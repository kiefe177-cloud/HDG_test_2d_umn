Remove-Item -Recurse -Force build
mkdir build
cd build
cmake -G "MinGW Makefiles" .. 
mingw32-make
ctest

Remove-Item -Recurse -Force build
cmake -B build -G "MinGW Makefiles"
cmake --build build

using namespace Eigen

Need to cd into output to plot data.

py ..\scripts\{filename}.py

-march=native or -march=x86-64-v3 for Eigenspeedup?


For sparse matrices:

Eigen::SparseMatrix<double> A(rows, cols);          // 1. empty shell
std::vector<Eigen::Triplet<double>> trips;          // 2. fill the recipe
trips.reserve(estimated_nnz);
//   ... emplace_back(row, col, value) for every non-zero ...
A.setFromTriplets(trips.begin(), trips.end());      // 3. bake the matrix


My ranking, including pruning
Adding to the previous list:

Reserve A_triplets capacity X
Skip the elem_block intermediate X
Hoist invariants out of the i1 loop X
Replace to_complex(X) with X.cast<cdouble>() X
Prune intermediate sparse matrices to drop stored zeros ← new X
Merge the two face loops X
Skip sponge computation for elements outside the sponge region ← new X
OpenMP-parallelize the i1 loop
Reuse sparsity pattern across calls if doing parameter sweeps ← new
Store LU factorization instead of inv(D)

// In C++, load and diff
// ... load dg_matlab.mat ...
double err_A = (result.A - dg_matlab.A).norm() / dg_matlab.A.norm();


#include <omp.h>

const int nthreads = omp_get_max_threads();
std::vector<std::vector<Eigen::Triplet<cdouble>>> thread_triplets(nthreads);

#pragma omp parallel for
for (int i1 = 0; i1 < Ne; ++i1) {
    int tid = omp_get_thread_num();
    auto& local_trips = thread_triplets[tid];
    
    // ... all the element work, with local_trips replacing A_triplets ...
    // Also: result.QB[i1] = ..., etc. — these are independent writes, safe
}

// Merge
size_t total = 0;
for (auto& v : thread_triplets) total += v.size();
A_triplets.reserve(total);
for (auto& v : thread_triplets) A_triplets.insert(A_triplets.end(), v.begin(), v.end());