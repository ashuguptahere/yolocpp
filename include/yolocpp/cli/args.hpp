#pragma once
//
// Ultralytics-style key=value argument parser.
//
// Accepts Ultralytics-compatible invocations such as
//
//   yolocpp task=detect mode=train model=yolo8n.pt data=coco.yaml \
//           epochs=100 imgsz=640 batch=16 device=cuda:0
//
// All arguments are unordered `key=value` pairs (no leading dashes).
// Boolean values accept {true, false, 1, 0, yes, no, on, off}.
// Anything not matching `key=value` is recorded as a positional arg.
//

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolocpp::cli {

class Args {
 public:
  static Args parse(int argc, char** argv);

  // Look up by key. Returns empty optional if absent.
  std::optional<std::string> get(const std::string& key) const;

  // Typed accessors with defaults.
  std::string get_str   (const std::string& key, std::string  d = "") const;
  int         get_int   (const std::string& key, int          d = 0) const;
  double      get_double(const std::string& key, double       d = 0.0) const;
  bool        get_bool  (const std::string& key, bool         d = false) const;

  bool has(const std::string& key) const { return kv_.count(key) > 0; }

  const std::map<std::string, std::string>& kv() const { return kv_; }
  const std::vector<std::string>& positional() const { return pos_; }

  // Throws if the given keys don't appear in the canonical set; useful
  // for catching typos like "epoch=10" instead of "epochs=10".
  void warn_unknown(const std::vector<std::string>& canonical) const;

 private:
  std::map<std::string, std::string> kv_;
  std::vector<std::string>           pos_;
};

}  // namespace yolocpp::cli
