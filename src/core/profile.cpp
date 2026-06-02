#include "yolocpp/core/profile.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>

namespace yolocpp::core {

namespace {

double median(std::vector<double> xs) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  return xs[xs.size() / 2];
}

double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  std::size_t i = static_cast<std::size_t>(p * (xs.size() - 1));
  return xs[i];
}

double mean(const std::vector<double>& xs) {
  if (xs.empty()) return 0.0;
  double s = 0.0;
  for (auto v : xs) s += v;
  return s / xs.size();
}

double sum(const std::vector<double>& xs) {
  double s = 0.0;
  for (auto v : xs) s += v;
  return s;
}

}  // namespace

void Profile::print_summary(std::ostream& os) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (samples_.empty()) {
    os << "\n[profile] no samples recorded (was --profile on?)\n";
    return;
  }
  // Total time across all phases, for share %.
  double total_ms = 0.0;
  for (auto& kv : samples_) total_ms += sum(kv.second);

  os << "\n=== profile summary ===\n";
  os << std::left << std::setw(28) << "phase"
     << std::right << std::setw(7) << "calls"
     << std::setw(11) << "median ms"
     << std::setw(11) << "p95 ms"
     << std::setw(11) << "mean ms"
     << std::setw(11) << "total ms"
     << std::setw(8)  << "share%"
     << "\n";
  os << std::string(87, '-') << "\n";
  // Sort by total ms descending so the worst offenders are on top.
  std::vector<std::pair<std::string, std::vector<double>>> rows(
      samples_.begin(), samples_.end());
  std::sort(rows.begin(), rows.end(),
            [](auto& a, auto& b) { return sum(a.second) > sum(b.second); });
  for (auto& kv : rows) {
    double t  = sum(kv.second);
    double sh = total_ms > 0 ? 100.0 * t / total_ms : 0.0;
    os << std::left << std::setw(28) << kv.first
       << std::right << std::setw(7) << kv.second.size()
       << std::fixed << std::setprecision(3)
       << std::setw(11) << median(kv.second)
       << std::setw(11) << percentile(kv.second, 0.95)
       << std::setw(11) << mean(kv.second)
       << std::setw(11) << t
       << std::setprecision(1) << std::setw(7) << sh << "%\n";
  }
  os << std::string(87, '-') << "\n";
  os << std::left << std::setw(28) << "TOTAL"
     << std::right << std::setw(7) << ""
     << std::fixed << std::setprecision(3)
     << std::setw(11) << "" << std::setw(11) << ""
     << std::setw(11) << "" << std::setw(11) << total_ms
     << "\n";
}

}  // namespace yolocpp::core
