#pragma once
//
// Hungarian (Kuhn-Munkres) bipartite-matching algorithm. Used by
// the RF-DETR set loss (#65F) to match each predicted query to at
// most one ground-truth box, minimising a per-pair cost matrix.
//
// Implementation: O(n³) shortest-augmenting-path variant using
// potentials, also known as "Jonker-Volgenant for rectangular cost
// matrices." Pure C++, vendored — no scipy / no external lib (per
// `third_party/DEPS.md`).
//
// Input:  cost matrix `[n_pred, n_gt]` (rectangular OK; n_pred ≥ n_gt
//         expected since RF-DETR has many more queries than GTs).
// Output: vector of length n_gt mapping each GT index to the matched
//         prediction index. Unmatched preds are absent from the
//         result; they get the "no-object" class in the loss.

#include <cstdint>
#include <vector>

namespace yolocpp::losses {

// Solves the linear-assignment problem on a row-major `[rows, cols]`
// cost matrix (rows ≥ cols). Returns `assignment[c] = r` for each
// column c, where r is the matched row index.
std::vector<int64_t> hungarian_assign(const float* cost, int64_t rows,
                                       int64_t cols);

}  // namespace yolocpp::losses
