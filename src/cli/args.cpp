#include "yolocpp/cli/args.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <set>
#include <stdexcept>

namespace yolocpp::cli {

namespace {
std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}
}  // namespace

Args Args::parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    if (eq != std::string::npos) {
      std::string k = s.substr(0, eq);
      std::string v = s.substr(eq + 1);
      while (!k.empty() && k.front() == '-') k.erase(0, 1);
      a.kv_[std::move(k)] = std::move(v);
      continue;
    }
    // Also accept "--key value" (two tokens). If this token looks like a
    // flag and the next one isn't, eat both as a kv pair.
    if (s.size() > 2 && s[0] == '-' && s[1] == '-' && i + 1 < argc) {
      std::string next = argv[i + 1];
      if (next.empty() || (next[0] != '-' && next.find('=') == std::string::npos)) {
        std::string k = s.substr(2);
        a.kv_[std::move(k)] = std::move(next);
        ++i;
        continue;
      }
    }
    a.pos_.push_back(std::move(s));
  }
  return a;
}

std::optional<std::string> Args::get(const std::string& key) const {
  auto it = kv_.find(key);
  if (it == kv_.end()) return std::nullopt;
  return it->second;
}

std::string Args::get_str(const std::string& key, std::string d) const {
  auto v = get(key);
  return v ? *v : d;
}
int Args::get_int(const std::string& key, int d) const {
  auto v = get(key);
  if (!v) return d;
  try { return std::stoi(*v); }
  catch (const std::exception&) {
    throw std::runtime_error("cannot parse int for " + key + "=" + *v);
  }
}
double Args::get_double(const std::string& key, double d) const {
  auto v = get(key);
  if (!v) return d;
  try { return std::stod(*v); }
  catch (const std::exception&) {
    throw std::runtime_error("cannot parse number for " + key + "=" + *v);
  }
}
bool Args::get_bool(const std::string& key, bool d) const {
  auto v = get(key);
  if (!v) return d;
  std::string lv = lower(*v);
  if (lv == "true"  || lv == "1" || lv == "yes" || lv == "on")  return true;
  if (lv == "false" || lv == "0" || lv == "no"  || lv == "off") return false;
  throw std::runtime_error("cannot parse bool for " + key + "=" + *v);
}

void Args::warn_unknown(const std::vector<std::string>& canonical) const {
  std::set<std::string> ok(canonical.begin(), canonical.end());
  for (const auto& [k, _] : kv_) {
    if (ok.count(k) == 0) {
      std::cerr << "[warn] unknown argument: " << k << " (typo?)\n";
    }
  }
}

}  // namespace yolocpp::cli
