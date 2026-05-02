#include "yolocpp/engine/validator.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "yolocpp/inference/letterbox.hpp"

namespace yolocpp::engine {

template <typename M>
metrics::mAPResult validate(M& model, const datasets::YoloDataset& dataset,
                            torch::Device device, inference::NMSConfig nms_cfg) {
  model->to(device);
  model->eval();

  std::vector<metrics::DetectionRow>   all_dets;
  std::vector<metrics::GroundTruthRow> all_gts;

  size_t n = dataset.size();
  for (size_t i = 0; i < n; ++i) {
    // We need the original image size + letterbox info to map detections
    // back. Re-create with augment=false → deterministic letterbox.
    auto ex = dataset.get(i);
    inference::LetterboxResult lb;
    lb.gain   = ex.gain;
    lb.pad_x  = ex.pad_x;
    lb.pad_y  = ex.pad_y;
    lb.orig_w = ex.orig_w;
    lb.orig_h = ex.orig_h;

    // Forward.
    auto x = ex.img.unsqueeze(0).to(device);
    torch::Tensor pred;
    {
      torch::NoGradGuard ng;
      pred = model->forward_eval(x);
    }
    auto outs = inference::nms(pred, nms_cfg);
    auto& det = outs[0];           // [k, 6]: xyxy + conf + cls (lb coords)

    if (det.size(0) > 0) {
      auto boxes = det.slice(1, 0, 4).contiguous();
      inference::scale_boxes(boxes, lb);
      det.slice(1, 0, 4) = boxes;
      auto a = det.accessor<float, 2>();
      for (int j = 0; j < det.size(0); ++j) {
        metrics::DetectionRow d{a[j][0], a[j][1], a[j][2], a[j][3],
                                a[j][4], (int)a[j][5], (int)i};
        all_dets.push_back(d);
      }
    }

    // GTs in original image coords (un-letterbox the targets).
    if (ex.targets.size(0) > 0) {
      auto t = ex.targets.clone();
      auto a = t.accessor<float, 2>();
      for (int j = 0; j < t.size(0); ++j) {
        float cx = a[j][1], cy = a[j][2], w = a[j][3], h = a[j][4];
        float x1 = cx - w / 2, y1 = cy - h / 2;
        float x2 = cx + w / 2, y2 = cy + h / 2;
        // Inverse of the dataset's letterboxing.
        x1 = (float)((x1 - lb.pad_x) / lb.gain);
        x2 = (float)((x2 - lb.pad_x) / lb.gain);
        y1 = (float)((y1 - lb.pad_y) / lb.gain);
        y2 = (float)((y2 - lb.pad_y) / lb.gain);
        metrics::GroundTruthRow g{x1, y1, x2, y2, (int)a[j][0], (int)i};
        all_gts.push_back(g);
      }
    }
  }

  return metrics::compute_map(all_dets, all_gts, dataset.num_classes());
}

template <typename M>
ValidationOutput validate_with_records(M& model,
                                       const datasets::YoloDataset& dataset,
                                       torch::Device device,
                                       inference::NMSConfig nms_cfg) {
  // Mirror of validate() but returns the rows alongside the mAP.
  model->to(device);
  model->eval();
  ValidationOutput out;
  size_t n = dataset.size();
  for (size_t i = 0; i < n; ++i) {
    auto ex = dataset.get(i);
    inference::LetterboxResult lb;
    lb.gain = ex.gain; lb.pad_x = ex.pad_x; lb.pad_y = ex.pad_y;
    lb.orig_w = ex.orig_w; lb.orig_h = ex.orig_h;

    auto x = ex.img.unsqueeze(0).to(device);
    torch::Tensor pred;
    {
      torch::NoGradGuard ng;
      pred = model->forward_eval(x);
    }
    auto outs = inference::nms(pred, nms_cfg);
    auto& det = outs[0];
    if (det.size(0) > 0) {
      auto boxes = det.slice(1, 0, 4).contiguous();
      inference::scale_boxes(boxes, lb);
      det.slice(1, 0, 4) = boxes;
      auto a = det.accessor<float, 2>();
      for (int j = 0; j < det.size(0); ++j) {
        out.dets.push_back({a[j][0], a[j][1], a[j][2], a[j][3],
                            a[j][4], (int)a[j][5], (int)i});
      }
    }
    if (ex.targets.size(0) > 0) {
      auto t = ex.targets.clone();
      auto a = t.accessor<float, 2>();
      for (int j = 0; j < t.size(0); ++j) {
        float cx = a[j][1], cy = a[j][2], w = a[j][3], h = a[j][4];
        float x1 = (cx - w / 2 - lb.pad_x) / lb.gain;
        float x2 = (cx + w / 2 - lb.pad_x) / lb.gain;
        float y1 = (cy - h / 2 - lb.pad_y) / lb.gain;
        float y2 = (cy + h / 2 - lb.pad_y) / lb.gain;
        out.gts.push_back({x1, y1, x2, y2, (int)a[j][0], (int)i});
      }
    }
  }
  out.map = metrics::compute_map(out.dets, out.gts, dataset.num_classes());
  return out;
}

void render_confusion_matrix(const std::vector<std::vector<int>>& m,
                             const std::vector<std::string>& names,
                             const std::string& out_path) {
  int n = (int)m.size();              // (nc + 1)
  int cell = std::max(40, 800 / std::max(1, n));
  int label_w = 120, label_h = 120;
  int W = label_w + cell * n + 60;
  int H = label_h + cell * n + 60;
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

  // Find max for color scaling.
  int max_v = 1;
  for (auto& row : m) for (int v : row) max_v = std::max(max_v, v);

  // Cell color: per-row normalized (upstream-style).
  for (int r = 0; r < n; ++r) {
    int row_sum = 0;
    for (int c = 0; c < n; ++c) row_sum += m[r][c];
    for (int c = 0; c < n; ++c) {
      double frac = row_sum > 0 ? (double)m[r][c] / row_sum : 0.0;
      // Blue→white gradient (lighter = stronger)
      int b = 255;
      int g = (int)(255 * (1.0 - frac));
      int rr = (int)(255 * (1.0 - frac));
      cv::rectangle(img,
                    {label_w + c * cell, label_h + r * cell},
                    {label_w + (c + 1) * cell, label_h + (r + 1) * cell},
                    cv::Scalar(b, g, rr), cv::FILLED);
      cv::rectangle(img,
                    {label_w + c * cell, label_h + r * cell},
                    {label_w + (c + 1) * cell, label_h + (r + 1) * cell},
                    cv::Scalar(180, 180, 180), 1);
      if (m[r][c] > 0) {
        char buf[16]; std::snprintf(buf, sizeof(buf), "%d", m[r][c]);
        cv::Scalar text_col = frac > 0.5 ? cv::Scalar(255, 255, 255)
                                          : cv::Scalar(50, 50, 50);
        cv::putText(img, buf,
                    {label_w + c * cell + cell / 4,
                     label_h + r * cell + cell / 2 + 5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, text_col, 1);
      }
    }
  }

  // Axis labels.
  auto label_for = [&](int i) {
    if (i < (int)names.size()) return names[i];
    return std::string("background");
  };
  for (int i = 0; i < n; ++i) {
    auto lbl = label_for(i);
    if ((int)lbl.size() > 12) lbl = lbl.substr(0, 11) + ".";
    // top: column labels (rotated effectively — we just stack)
    cv::putText(img, lbl,
                {label_w + i * cell + 4, label_h - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {30, 30, 30}, 1);
    // left: row labels
    cv::putText(img, lbl,
                {6, label_h + i * cell + cell / 2 + 4},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {30, 30, 30}, 1);
  }
  cv::putText(img, "Predicted",
              {6, label_h - 30},
              cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
  cv::putText(img, "Ground truth",
              {label_w + 4, 30},
              cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);

  cv::imwrite(out_path, img);
}

void render_curve_png(const std::vector<std::vector<double>>& curves,
                      const std::vector<double>& xs,
                      const std::vector<std::string>& names,
                      const std::vector<int>& n_gt_per_class,
                      const std::string& xlabel, const std::string& ylabel,
                      const std::string& title,
                      const std::string& out_path) {
  // Plot canvas: 1200×900; padding for axes + legend.
  int W = 1200, H = 900;
  int pad_l = 90, pad_r = 280, pad_t = 60, pad_b = 70;
  int plot_w = W - pad_l - pad_r;
  int plot_h = H - pad_t - pad_b;
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

  // Axes box.
  cv::rectangle(img, {pad_l, pad_t}, {pad_l + plot_w, pad_t + plot_h},
                {200, 200, 200}, 1);

  // Gridlines + axis ticks.
  for (int i = 0; i <= 10; ++i) {
    double frac = i / 10.0;
    int xx = pad_l + (int)(frac * plot_w);
    int yy = pad_t + plot_h - (int)(frac * plot_h);
    cv::line(img, {xx, pad_t}, {xx, pad_t + plot_h}, {235, 235, 235}, 1);
    cv::line(img, {pad_l, yy}, {pad_l + plot_w, yy}, {235, 235, 235}, 1);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f", frac);
    cv::putText(img, buf, {xx - 12, pad_t + plot_h + 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {30, 30, 30}, 1);
    cv::putText(img, buf, {pad_l - 38, yy + 4},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {30, 30, 30}, 1);
  }

  // Per-class lines (alpha-thin) + an "all classes" mean (thick).
  int nc = (int)curves.size();
  std::vector<double> mean(xs.size(), 0.0);
  int n_active = 0;

  auto color_for = [](int c) {
    return cv::Scalar((c * 41) % 256, (c * 73) % 256, (c * 137) % 256);
  };

  auto draw_line = [&](const std::vector<double>& y, cv::Scalar color, int thickness) {
    for (size_t i = 1; i < xs.size(); ++i) {
      int x0 = pad_l + (int)(xs[i - 1] * plot_w);
      int x1 = pad_l + (int)(xs[i]     * plot_w);
      int y0 = pad_t + plot_h - (int)(std::clamp(y[i - 1], 0.0, 1.0) * plot_h);
      int y1 = pad_t + plot_h - (int)(std::clamp(y[i],     0.0, 1.0) * plot_h);
      cv::line(img, {x0, y0}, {x1, y1}, color, thickness, cv::LINE_AA);
    }
  };

  for (int c = 0; c < nc; ++c) {
    if (n_gt_per_class.empty() || n_gt_per_class[c] == 0) continue;
    draw_line(curves[c], color_for(c), 1);
    for (size_t i = 0; i < xs.size(); ++i) mean[i] += curves[c][i];
    ++n_active;
  }
  if (n_active > 0) for (auto& m : mean) m /= n_active;
  draw_line(mean, {0, 0, 0}, 3);

  // Title + axis labels.
  cv::putText(img, title, {pad_l, pad_t - 22},
              cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 2);
  cv::putText(img, xlabel, {pad_l + plot_w / 2 - 50, H - 25},
              cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 1);
  // Y axis label drawn vertically (rotated text → write 1 char per row).
  for (size_t i = 0; i < ylabel.size(); ++i) {
    cv::putText(img, std::string(1, ylabel[i]),
                {15, pad_t + plot_h / 2 - (int)(ylabel.size() * 8) + (int)i * 16},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
  }

  // Legend (right side).
  int legend_x = pad_l + plot_w + 20;
  int row_h    = 22;
  int max_rows = std::min<int>(plot_h / row_h, std::max(1, n_active + 1));
  // "all classes" line first.
  cv::line(img, {legend_x, pad_t + 12}, {legend_x + 20, pad_t + 12},
           {0, 0, 0}, 3);
  cv::putText(img, "all classes", {legend_x + 26, pad_t + 17},
              cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);
  int row = 1;
  for (int c = 0; c < nc && row < max_rows; ++c) {
    if (!n_gt_per_class.empty() && n_gt_per_class[c] == 0) continue;
    int y = pad_t + 12 + row * row_h;
    cv::line(img, {legend_x, y}, {legend_x + 20, y}, color_for(c), 2);
    auto label = c < (int)names.size() ? names[c] : std::to_string(c);
    if (label.size() > 18) label = label.substr(0, 17) + ".";
    cv::putText(img, label, {legend_x + 26, y + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 0, 0}, 1);
    ++row;
  }

  cv::imwrite(out_path, img);
}

void render_labels_histogram(const std::vector<metrics::GroundTruthRow>& gts,
                             const std::vector<std::string>& names,
                             const std::string& out_path) {
  int nc = (int)names.size();
  std::vector<int> counts(nc, 0);
  for (const auto& g : gts)
    if (g.cls >= 0 && g.cls < nc) ++counts[g.cls];

  int W = 1200, H = 600, pad_l = 90, pad_r = 30, pad_t = 50, pad_b = 90;
  int plot_w = W - pad_l - pad_r, plot_h = H - pad_t - pad_b;
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(img, {pad_l, pad_t}, {pad_l + plot_w, pad_t + plot_h},
                {200, 200, 200}, 1);

  int max_v = 1;
  for (int v : counts) max_v = std::max(max_v, v);
  // Bars.
  int bar_w = plot_w / std::max(1, nc);
  for (int c = 0; c < nc; ++c) {
    double frac = (double)counts[c] / max_v;
    int bh = (int)(frac * plot_h);
    int x0 = pad_l + c * bar_w + 2;
    int x1 = pad_l + (c + 1) * bar_w - 2;
    int y0 = pad_t + plot_h - bh, y1 = pad_t + plot_h;
    cv::Scalar col((c * 41) % 256, (c * 73) % 256, (c * 137) % 256);
    cv::rectangle(img, {x0, y0}, {x1, y1}, col, cv::FILLED);
    char buf[16]; std::snprintf(buf, sizeof(buf), "%d", counts[c]);
    cv::putText(img, buf, {x0, y0 - 4},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {30, 30, 30}, 1);
    auto label = c < (int)names.size() ? names[c] : std::to_string(c);
    if (label.size() > 8) label = label.substr(0, 7) + ".";
    cv::putText(img, label, {x0, pad_t + plot_h + 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {30, 30, 30}, 1);
  }
  cv::putText(img, "Per-class instance count",
              {pad_l, pad_t - 20},
              cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 2);
  cv::imwrite(out_path, img);
}

void render_results_png(const std::string& csv_path,
                        const std::string& out_path) {
  std::ifstream f(csv_path);
  if (!f.is_open()) return;
  std::string line;
  std::getline(f, line);  // header
  // Header: epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,
  //         metrics/mAP50(B),metrics/mAP50-95(B),lr0
  std::vector<double> ep, box_l, cls_l, dfl_l, m50, m5095, lr;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::vector<std::string> cols;
    std::string cur;
    for (char ch : line) {
      if (ch == ',') { cols.push_back(cur); cur.clear(); }
      else cur += ch;
    }
    cols.push_back(cur);
    if (cols.size() < 8) continue;
    auto safe_d = [](const std::string& s) -> double {
      try { return std::stod(s); } catch (...) { return 0.0; }
    };
    ep.push_back(safe_d(cols[0]));
    box_l.push_back(safe_d(cols[2]));
    cls_l.push_back(safe_d(cols[3]));
    dfl_l.push_back(safe_d(cols[4]));
    m50.push_back(safe_d(cols[5]));
    m5095.push_back(safe_d(cols[6]));
    lr.push_back(safe_d(cols[7]));
  }
  if (ep.empty()) return;
  // 2×3 grid of subplots.
  int W = 1500, H = 800, gx = 3, gy = 2;
  int pad = 12;
  int sub_w = (W - (gx + 1) * pad) / gx;
  int sub_h = (H - (gy + 1) * pad) / gy;
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

  auto plot = [&](int row, int col, const std::string& title,
                   const std::vector<double>& y, cv::Scalar c) {
    int x0 = pad + col * (sub_w + pad);
    int y0 = pad + row * (sub_h + pad);
    cv::rectangle(img, {x0, y0}, {x0 + sub_w, y0 + sub_h},
                  {220, 220, 220}, 1);
    double lo = *std::min_element(y.begin(), y.end());
    double hi = *std::max_element(y.begin(), y.end());
    if (hi <= lo) hi = lo + 1.0;
    auto map_xy = [&](size_t i) {
      double xf = ep.size() > 1 ? (double)i / (ep.size() - 1) : 0.5;
      double yf = (y[i] - lo) / (hi - lo);
      return cv::Point(x0 + 30 + (int)(xf * (sub_w - 50)),
                       y0 + sub_h - 25 - (int)(yf * (sub_h - 50)));
    };
    for (size_t i = 1; i < y.size(); ++i)
      cv::line(img, map_xy(i - 1), map_xy(i), c, 2, cv::LINE_AA);
    cv::putText(img, title, {x0 + 6, y0 + 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", lo);
    cv::putText(img, buf, {x0 + 4, y0 + sub_h - 4},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {80, 80, 80}, 1);
    std::snprintf(buf, sizeof(buf), "%.4g", hi);
    cv::putText(img, buf, {x0 + 4, y0 + 36},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {80, 80, 80}, 1);
  };

  plot(0, 0, "train/box_loss",       box_l, {0, 0, 200});
  plot(0, 1, "train/cls_loss",       cls_l, {0, 0, 200});
  plot(0, 2, "train/dfl_loss",       dfl_l, {0, 0, 200});
  plot(1, 0, "metrics/mAP50(B)",     m50,   {0, 150, 0});
  plot(1, 1, "metrics/mAP50-95(B)",  m5095, {0, 150, 0});
  plot(1, 2, "lr0",                  lr,    {150, 0, 150});

  cv::imwrite(out_path, img);
}

void render_train_batch(const torch::Tensor& imgs, const torch::Tensor& targets,
                        const std::vector<std::string>& names,
                        const std::string& out_path) {
  if (imgs.size(0) == 0) return;
  int B = (int)imgs.size(0), H = (int)imgs.size(2), W = (int)imgs.size(3);
  // 4×4 grid (16 cells); pad with blanks if B < 16.
  int gx = 4, gy = 4;
  int cell_h = H / 2, cell_w = W / 2;  // half-size per cell to keep file small
  int total_w = gx * cell_w, total_h = gy * cell_h;
  cv::Mat grid(total_h, total_w, CV_8UC3, cv::Scalar(20, 20, 20));

  // Image tensor → CPU, contiguous, scaled to 0..255 BGR.
  auto cpu = imgs.detach().to(torch::kCPU).contiguous();
  auto a   = cpu.accessor<float, 4>();

  // Bucket targets by image index.
  std::vector<std::vector<std::array<float, 5>>> per_img(B);
  if (targets.size(0) > 0) {
    auto t = targets.detach().to(torch::kCPU).contiguous();
    auto ta = t.accessor<float, 2>();
    for (int i = 0; i < (int)t.size(0); ++i) {
      int bi = (int)ta[i][0];
      if (bi < 0 || bi >= B) continue;
      per_img[bi].push_back({ta[i][1], ta[i][2], ta[i][3], ta[i][4], ta[i][5]});
    }
  }

  for (int idx = 0; idx < std::min(gx * gy, B); ++idx) {
    // Render image idx.
    cv::Mat img(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
      auto* row = img.ptr<cv::Vec3b>(y);
      for (int x = 0; x < W; ++x) {
        int b = std::clamp((int)(a[idx][2][y][x] * 255), 0, 255);
        int g = std::clamp((int)(a[idx][1][y][x] * 255), 0, 255);
        int r = std::clamp((int)(a[idx][0][y][x] * 255), 0, 255);
        row[x] = cv::Vec3b((uint8_t)b, (uint8_t)g, (uint8_t)r);
      }
    }
    // Draw targets in original-resolution before resize so labels stay legible.
    for (const auto& t : per_img[idx]) {
      int cls = (int)t[0];
      cv::Scalar color((cls * 41) % 256, (cls * 73) % 256, (cls * 137) % 256);
      int x1 = (int)(t[1] - t[3] / 2);
      int y1 = (int)(t[2] - t[4] / 2);
      int x2 = (int)(t[1] + t[3] / 2);
      int y2 = (int)(t[2] + t[4] / 2);
      cv::rectangle(img, {x1, y1}, {x2, y2}, color, 2);
      auto label = (cls < (int)names.size() ? names[cls] : std::to_string(cls));
      cv::putText(img, label, {x1, std::max(12, y1 - 4)},
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    }
    // Resize and paste into grid.
    cv::Mat resized;
    cv::resize(img, resized, {cell_w, cell_h}, 0, 0, cv::INTER_LINEAR);
    int gx_i = idx % gx, gy_i = idx / gx;
    resized.copyTo(grid(cv::Rect(gx_i * cell_w, gy_i * cell_h,
                                  cell_w, cell_h)));
  }
  cv::imwrite(out_path, grid);
}

// Explicit instantiations of the validator templates for both model holders.
template metrics::mAPResult validate<models::Yolo8Detect>(
    models::Yolo8Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo5Detect>(
    models::Yolo5Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo8Detect>(
    models::Yolo8Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo5Detect>(
    models::Yolo5Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo11Detect>(
    models::Yolo11Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo11Detect>(
    models::Yolo11Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo26Detect>(
    models::Yolo26Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo26Detect>(
    models::Yolo26Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo12Detect>(
    models::Yolo12Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo12Detect>(
    models::Yolo12Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo13Detect>(
    models::Yolo13Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo13Detect>(
    models::Yolo13Detect&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::RFDetr>(
    models::RFDetr&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::RFDetr>(
    models::RFDetr&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);

// v3/v4/v6/v7/v9/v10 — predict was wired earlier; val plugs in to the same
// templated runner since each holder exposes `forward_eval(x) → [B,4+nc,A]`
// in xyxy + sigmoid'd cls form.
template metrics::mAPResult validate<models::Yolo3>(
    models::Yolo3&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo3>(
    models::Yolo3&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo4>(
    models::Yolo4&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo4>(
    models::Yolo4&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo6>(
    models::Yolo6&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo6>(
    models::Yolo6&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo7>(
    models::Yolo7&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo7>(
    models::Yolo7&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo9>(
    models::Yolo9&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo9>(
    models::Yolo9&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template metrics::mAPResult validate<models::Yolo10>(
    models::Yolo10&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);
template ValidationOutput validate_with_records<models::Yolo10>(
    models::Yolo10&, const datasets::YoloDataset&, torch::Device,
    inference::NMSConfig);

}  // namespace yolocpp::engine
