// Clay's single-header implementation, compiled as C.
//
// The project sets CMAKE_CXX_EXTENSIONS OFF, and Clay's implementation body
// uses C compound literals `(Type){...}` that a strict-C++20 compiler rejects.
// Compiling the implementation as C (where compound literals are standard)
// sidesteps that entirely. Every other translation unit includes clay.h for
// declarations only — they are wrapped in `extern "C"`, so the C-linkage
// symbols defined here resolve cleanly from C++.
#define CLAY_IMPLEMENTATION
#include "clay.h"
