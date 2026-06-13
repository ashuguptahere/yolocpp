// yolocpp web console — native C++20 backend.
//
// Serves the Clay-rendered dashboard (see web_ui.cpp) and exposes a tiny JSON
// API the page's fetch glue calls: POST /api/{predict,val,train,export} queue a
// job, GET /api/jobs returns their live state. Jobs run one-at-a-time on a
// single worker thread (the GPU is serial), each invoking the same public
// `yolocpp::YOLO` API the CLI uses. The browser never touches LibTorch/CUDA —
// it only drives this process.

#include <yolocpp/config.hpp>

#include "yolocpp/api.hpp"
#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/core/log.hpp"
#include "yolocpp/web/web_ui.hpp"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

int   as_int(const std::string& s, int d)   { try { return s.empty() ? d : std::stoi(s); } catch (...) { return d; } }
float as_flt(const std::string& s, float d)  { try { return s.empty() ? d : std::stof(s); } catch (...) { return d; } }

std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof b, "%.4f", v); return b; }

std::string jesc(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n";  break;
      case '\r': break;
      case '\t': o += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
  }
  return o;
}

// ── job model + registry ───────────────────────────────────────────────────
struct Job {
  int id = 0;
  std::string mode, model, status, message, log;
  std::map<std::string, std::string> params;
};

class Jobs {
 public:
  Jobs() : worker_([this] { run(); }) {}
  ~Jobs() {
    stop_ = true;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  int submit(const std::string& mode, std::map<std::string, std::string> p) {
    auto j = std::make_shared<Job>();
    {
      std::lock_guard<std::mutex> l(mu_);
      j->id = next_++;
      j->mode = mode;
      j->model = p["model"];
      j->status = "queued";
      j->params = std::move(p);
      all_.push_front(j);
      while (all_.size() > 30) all_.pop_back();
      queue_.push_back(j);
    }
    cv_.notify_one();
    return j->id;
  }

  std::string json() {
    std::lock_guard<std::mutex> l(mu_);
    std::string s = "[";
    bool first = true;
    for (const auto& jp : all_) {
      const Job& j = *jp;
      if (!first) s += ",";
      first = false;
      s += "{\"id\":" + std::to_string(j.id) +
           ",\"mode\":\"" + jesc(j.mode) + "\"" +
           ",\"model\":\"" + jesc(j.model) + "\"" +
           ",\"status\":\"" + jesc(j.status) + "\"" +
           ",\"message\":\"" + jesc(j.message) + "\"" +
           ",\"log\":\"" + jesc(j.log) + "\"}";
    }
    return s + "]";
  }

 private:
  void run() {
    for (;;) {
      std::shared_ptr<Job> j;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        j = queue_.front();
        queue_.pop_front();
        j->status = "running";
      }
      execute(j);
    }
  }

  void set(const std::shared_ptr<Job>& j, std::string status, std::string msg, std::string log) {
    std::lock_guard<std::mutex> l(mu_);
    j->status = std::move(status);
    j->message = std::move(msg);
    j->log = std::move(log);
  }

  void execute(const std::shared_ptr<Job>& j) {
    auto& p = j->params;
    std::ostringstream cap;
    std::streambuf* o = std::cout.rdbuf(cap.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(cap.rdbuf());
    std::string msg, status = "done";
    try {
      const std::string dev = (p["device"].empty() || p["device"] == "auto") ? "" : p["device"];
      const std::string task = p["task"].empty() ? "detect" : p["task"];
      // Resolve the model the same way the CLI dispatcher does (searches
      // ./models, ./data, downloads on miss). The YOLO ctor itself takes a
      // literal path, so a bare name like "yolo8n.pt" must be resolved here.
      const std::string weights = yolocpp::cli::resolve_weights(p["model"]);
      yolocpp::YOLO m(weights);
      if (!dev.empty()) m.to(dev);
      m.task(task);

      if (j->mode == "predict") {
        if (p["source"].empty()) throw std::runtime_error("source is required for predict");
        yolocpp::PredictArgs a;
        a.source = p["source"]; a.imgsz = as_int(p["imgsz"], 640);
        a.conf = as_flt(p["conf"], 0.25f); a.task = task; a.device = dev;
        auto dets = m.predict(a);
        msg = std::to_string(dets.size()) + " detections — runs/predict/";
      } else if (j->mode == "val") {
        if (p["data"].empty()) throw std::runtime_error("data is required for val");
        yolocpp::ValArgs a;
        a.data = p["data"]; a.imgsz = as_int(p["imgsz"], 640); a.task = task; a.device = dev;
        auto r = m.val(a);
        msg = "mAP@0.5=" + fmt(r.map_50) + "  mAP@0.5:0.95=" + fmt(r.map_50_95);
      } else if (j->mode == "train") {
        if (p["data"].empty()) throw std::runtime_error("data is required for train");
        yolocpp::TrainArgs a;
        a.data = p["data"]; a.imgsz = as_int(p["imgsz"], 640);
        a.epochs = as_int(p["epochs"], 1); a.batch = as_int(p["batch"], 16);
        a.task = task; a.device = dev; a.save = "runs/train";
        m.train(a);
        msg = "training complete — runs/train/";
      } else if (j->mode == "export") {
        yolocpp::ExportArgs a;
        a.format = p["format"].empty() ? "onnx" : p["format"];
        a.precision = p["precision"].empty() ? "fp16" : p["precision"];
        a.imgsz = as_int(p["imgsz"], 640); a.task = task;
        m.export_(a);
        msg = "export complete (" + a.format + "/" + a.precision + ") — runs/export/";
      } else {
        throw std::runtime_error("unknown mode: " + j->mode);
      }
    } catch (const std::exception& ex) {
      status = "error";
      msg = ex.what();
    }
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    std::string log = cap.str();
    if (log.size() > 4000) log = "…\n" + log.substr(log.size() - 4000);
    std::cerr << "[web] job #" << j->id << " " << j->mode << " → " << status
              << " (" << msg << ")\n";
    set(j, status, msg, log);
  }

  std::mutex mu_;
  std::deque<std::shared_ptr<Job>> all_, queue_;
  std::condition_variable cv_;
  int next_ = 1;
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

std::vector<std::string> list_models() {
  std::vector<std::string> out;
  std::error_code ec;
  fs::path dir = fs::current_path() / "models";
  if (fs::is_directory(dir, ec)) {
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      if (e.is_regular_file() && e.path().extension() == ".pt")
        out.push_back(e.path().filename().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  yolocpp::log::init_from_env();  // YOLOCPP_LOG=debug for verbose job logs
  std::string host = "127.0.0.1";
  int port = 8080;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) host = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (a == "-h" || a == "--help") {
      std::cout << "usage: yolocpp_web [--host H] [--port P]\n";
      return 0;
    }
  }

  Jobs jobs;
  httplib::Server svr;

  svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
    yolocpp::web::DashboardModel m;
    m.version = YOLOCPP_VERSION_STRING;
    m.models = list_models();
    res.set_content(yolocpp::web::render_dashboard(m), "text/html; charset=utf-8");
  });

  svr.Get("/api/jobs", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(jobs.json(), "application/json");
  });

  auto post = [&](const std::string& mode) {
    return [&jobs, mode](const httplib::Request& req, httplib::Response& res) {
      std::map<std::string, std::string> p;
      for (const char* k : {"model", "task", "source", "data", "epochs", "batch",
                            "imgsz", "device", "conf", "format", "precision"})
        p[k] = req.has_param(k) ? req.get_param_value(k) : "";
      if (p["model"].empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"model is required\"}", "application/json");
        return;
      }
      int id = jobs.submit(mode, std::move(p));
      res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    };
  };
  svr.Post("/api/predict", post("predict"));
  svr.Post("/api/val", post("val"));
  svr.Post("/api/train", post("train"));
  svr.Post("/api/export", post("export"));

  // Serve predict/export artefacts so the browser can link to outputs.
  svr.set_mount_point("/runs", "runs");

  std::cerr << "yolocpp " << YOLOCPP_VERSION_STRING << " — web console on http://"
            << host << ":" << port << "\n";
  if (!svr.listen(host, port)) {
    std::cerr << "[web] failed to bind " << host << ":" << port << "\n";
    return 1;
  }
  return 0;
}
