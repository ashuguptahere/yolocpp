#pragma once
//
// Server-side Clay UI. The dashboard layout is authored with clay.h (the
// layout engine computes every box), and `render_dashboard` walks Clay's
// render-command output into positioned HTML/CSS — no WASM, no JS framework.
// The only client-side script is a thin fetch/poll glue emitted in the page
// (see web_ui.cpp) so the buttons can POST to the native backend.
//
#include <string>
#include <vector>

namespace yolocpp::web {

// Everything the server-rendered page needs that isn't static. Live job
// state is fetched client-side from /api/jobs, so it is not part of this.
struct DashboardModel {
  std::string version;                 // YOLOCPP_VERSION_STRING
  std::vector<std::string> models;     // entries in ./models for the picker
};

// Build the full HTML document: the Clay-laid-out body + page chrome + the
// fetch/poll glue. Thread-safe (Clay's global context is mutex-guarded
// inside).
std::string render_dashboard(const DashboardModel& m);

}  // namespace yolocpp::web
