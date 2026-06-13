#pragma once
//
// Tiny dependency-free leveled logger. Stream-style to match the codebase's
// existing `std::cerr << ...` idiom, with a level gate so DEBUG detail is free
// when disabled (the `<<` chain is skipped). Human-friendly errors carry an
// optional `→ hint` line. Output goes to stderr; coloured when stderr is a TTY
// (and NO_COLOR is unset).
//
// Verbosity: YOLOCPP_LOG=debug|info|warn|error|silent (read by init_from_env),
// or set_level() — the CLI maps --debug/--verbose onto it. Default is Info.
//
//   LOG_INFO("train")  << ds_size << " images, " << steps << " steps/epoch";
//   LOG_DEBUG("resolve") << "trying " << candidate;
//   LOG_ERROR("load") << "could not open '" << path << "'"
//                     << log::hint("check the path, or run --mode download");
//
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace yolocpp::log {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3, Silent = 4 };

// Read YOLOCPP_LOG once and set the level (call at process entry). Safe to call
// more than once; later set_level() calls win.
void init_from_env();
void set_level(Level lvl);
Level get_level();
Level level_from_string(std::string_view s);  // unknown → Info
bool  enabled(Level lvl);

// Low-level sink (mutex-guarded). Prefer the LOG_* macros.
void emit(Level lvl, std::string_view tag, std::string_view msg,
          std::string_view hint = {});

// Attach a remediation hint to an error/warn line: `<< log::hint("...")`.
struct Hint { std::string text; };
inline Hint hint(std::string text) { return {std::move(text)}; }

// Accumulates a line and flushes on destruction iff the level is enabled.
class LogLine {
 public:
  LogLine(Level lvl, std::string_view tag) : lvl_(lvl), tag_(tag), on_(enabled(lvl)) {}
  ~LogLine() { if (on_) emit(lvl_, tag_, os_.str(), hint_); }

  LogLine(const LogLine&) = delete;
  LogLine& operator=(const LogLine&) = delete;

  template <class T>
  LogLine& operator<<(const T& v) { if (on_) os_ << v; return *this; }
  LogLine& operator<<(const Hint& h) { if (on_) hint_ = h.text; return *this; }

 private:
  Level lvl_;
  std::string_view tag_;
  bool on_;
  std::ostringstream os_;
  std::string hint_;
};

}  // namespace yolocpp::log

#define LOG_DEBUG(tag) ::yolocpp::log::LogLine(::yolocpp::log::Level::Debug, tag)
#define LOG_INFO(tag)  ::yolocpp::log::LogLine(::yolocpp::log::Level::Info,  tag)
#define LOG_WARN(tag)  ::yolocpp::log::LogLine(::yolocpp::log::Level::Warn,  tag)
#define LOG_ERROR(tag) ::yolocpp::log::LogLine(::yolocpp::log::Level::Error, tag)
