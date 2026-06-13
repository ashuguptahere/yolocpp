#include "yolocpp/core/log.hpp"

#include <unistd.h>  // isatty, STDERR_FILENO

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>

namespace yolocpp::log {
namespace {

std::atomic<Level> g_level{Level::Info};
std::mutex g_mutex;

bool colour_enabled() {
  static const bool on = ::isatty(STDERR_FILENO) && std::getenv("NO_COLOR") == nullptr;
  return on;
}

struct Style { const char* name; const char* colour; };
Style style_for(Level l) {
  switch (l) {
    case Level::Debug: return {"DEBUG", "\033[90m"};  // grey
    case Level::Info:  return {"INFO ", "\033[36m"};  // cyan
    case Level::Warn:  return {"WARN ", "\033[33m"};  // yellow
    case Level::Error: return {"ERROR", "\033[31m"};  // red
    default:           return {"     ", ""};
  }
}

std::string timestamp() {
  using namespace std::chrono;
  std::time_t t = system_clock::to_time_t(system_clock::now());
  std::tm tm{};
  ::localtime_r(&t, &tm);
  char buf[16];
  std::strftime(buf, sizeof buf, "%H:%M:%S", &tm);
  return buf;
}

}  // namespace

Level level_from_string(std::string_view s) {
  std::string v;
  for (char c : s) v += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (v == "debug" || v == "trace" || v == "verbose") return Level::Debug;
  if (v == "info")  return Level::Info;
  if (v == "warn" || v == "warning") return Level::Warn;
  if (v == "error" || v == "err") return Level::Error;
  if (v == "silent" || v == "off" || v == "none" || v == "quiet") return Level::Silent;
  return Level::Info;
}

void init_from_env() {
  if (const char* e = std::getenv("YOLOCPP_LOG"); e && *e)
    g_level.store(level_from_string(e));
}
void set_level(Level lvl) { g_level.store(lvl); }
Level get_level() { return g_level.load(); }
bool enabled(Level lvl) { return lvl >= g_level.load(); }

void emit(Level lvl, std::string_view tag, std::string_view msg, std::string_view hint) {
  if (!enabled(lvl)) return;
  Style s = style_for(lvl);
  std::lock_guard<std::mutex> lock(g_mutex);
  std::ostream& o = std::cerr;
  const bool col = colour_enabled();
  if (col) o << "\033[90m" << '[' << timestamp() << "]\033[0m " << s.colour << s.name << "\033[0m ";
  else     o << '[' << timestamp() << "] " << s.name << ' ';
  if (!tag.empty()) o << tag << ": ";
  o << msg << '\n';
  if (!hint.empty()) {
    if (col) o << "  \033[90m→ " << hint << "\033[0m\n";
    else     o << "  -> " << hint << '\n';
  }
  o.flush();
}

}  // namespace yolocpp::log
