// Hungarian / Jonker-Volgenant assignment for rectangular cost
// matrices (#65F). O(rows·cols²) shortest-augmenting-path with
// potentials. Pure C++, no deps.
//
// Reference: Jonker & Volgenant, "A Shortest Augmenting Path
// Algorithm for Dense and Sparse Linear Assignment Problems"
// (Computing, 1987). The "rectangular" variant matches the smaller
// side; we use rows ≥ cols (queries ≥ GTs in RF-DETR).

#include "yolocpp/losses/hungarian.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace yolocpp::losses {

std::vector<int64_t> hungarian_assign(const float* cost, int64_t rows,
                                       int64_t cols) {
  if (cols == 0) return {};
  if (rows < cols) {
    throw std::invalid_argument(
        "hungarian_assign: rows must be >= cols (transpose your matrix)");
  }

  const float kInf = std::numeric_limits<float>::infinity();
  // u: row potentials [rows+1]; v: col potentials [cols+1].
  // p[c]:    matched row for column c (1-based; 0 = unmatched)
  // way[c]:  predecessor column on the augmenting path
  std::vector<float>   u(rows + 1, 0.0f);
  std::vector<float>   v(cols + 1, 0.0f);
  std::vector<int64_t> p(cols + 1, 0);
  std::vector<int64_t> way(cols + 1, 0);

  for (int64_t i = 1; i <= cols; ++i) {
    // i-th column to assign. p[0] holds the row currently being
    // matched (0-based row = p[0]-1).
    p[0] = i;
    int64_t j0 = 0;
    std::vector<float>   minv(cols + 1, kInf);
    std::vector<uint8_t> used(cols + 1, 0);

    do {
      used[j0] = 1;
      int64_t i0 = p[j0];
      float   delta = kInf;
      int64_t j1 = 0;

      for (int64_t j = 1; j <= cols; ++j) {
        if (used[j]) continue;
        // i0 is 1-based; cost row is (i0-1). j is 1-based; cost col is (j-1).
        float cur = cost[(i0 - 1) * cols + (j - 1)] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j]  = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1    = j;
        }
      }

      for (int64_t j = 0; j <= cols; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j]    -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    // Walk back through `way` to record the assignment.
    do {
      int64_t j1 = way[j0];
      p[j0]      = p[j1];
      j0         = j1;
    } while (j0 != 0);
  }

  // Build result: assignment[c] = matched row for column c (0-based).
  std::vector<int64_t> assignment(cols, -1);
  for (int64_t j = 1; j <= cols; ++j) {
    if (p[j] != 0) assignment[j - 1] = p[j] - 1;
  }
  return assignment;
}

}  // namespace yolocpp::losses
