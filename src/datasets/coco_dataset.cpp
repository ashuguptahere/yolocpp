// COCO JSON dataset (#54B).
//
// We parse the JSON with a hand-rolled minimal tokenizer (no
// libjson dep per `third_party/DEPS.md` — adding one for a single
// schema would dwarf the schema). The COCO format is
// well-structured and only three arrays + a handful of fields per
// row are interesting (images.{id, file_name, width, height},
// annotations.{image_id, category_id, bbox}, categories.{id,
// name}), so a streaming key-extraction approach over the raw
// bytes works fine.
//
// Implementation: scan once, emit per-image label vectors,
// materialise tensors in YOLO-normalised form. Same letterbox + aug
// pipeline as the other dataset classes.

#include "yolocpp/datasets/coco_dataset.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolocpp/inference/letterbox.hpp"

namespace fs = std::filesystem;

namespace yolocpp::datasets {

namespace {

void hsv_jitter(cv::Mat& bgr, std::mt19937& rng,
                float h_amp, float s_amp, float v_amp) {
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  float h = u(rng) * h_amp;
  float s = u(rng) * s_amp;
  float v = u(rng) * v_amp;
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch; cv::split(hsv, ch);
  ch[0].convertTo(ch[0], CV_32F);
  ch[1].convertTo(ch[1], CV_32F);
  ch[2].convertTo(ch[2], CV_32F);
  ch[0] = (ch[0] + h * 180.f);
  for (int r = 0; r < ch[0].rows; ++r) {
    auto* p = ch[0].ptr<float>(r);
    for (int c = 0; c < ch[0].cols; ++c) {
      float x = p[c]; while (x < 0)   x += 180.f;
                       while (x >= 180) x -= 180.f;
      p[c] = x;
    }
  }
  ch[1] = cv::min(cv::max(ch[1] * (1.f + s), 0.f), 255.f);
  ch[2] = cv::min(cv::max(ch[2] * (1.f + v), 0.f), 255.f);
  ch[0].convertTo(ch[0], CV_8U);
  ch[1].convertTo(ch[1], CV_8U);
  ch[2].convertTo(ch[2], CV_8U);
  cv::merge(ch, hsv);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

// ─── Minimal JSON tokenizer ──────────────────────────────────────────────
// Just enough to walk COCO's structure. Tokens we recognise:
//   { } [ ] : ,        — structural
//   "..."              — string (with \" + \\ escapes)
//   number             — integer or float literal
//   true/false/null    — bare keywords
// Any other input is a parse error. Comments aren't part of the JSON
// spec; we don't accept them.

enum class TokKind { LBrace, RBrace, LBracket, RBracket, Colon, Comma,
                      String, Number, True, False, Null, Eof };

struct Token { TokKind kind; std::string sval; double nval = 0; };

class JsonLexer {
 public:
  explicit JsonLexer(const std::string& body) : src_(body) {}
  Token next() {
    skip_ws();
    if (pos_ >= src_.size()) return {TokKind::Eof, "", 0};
    char c = src_[pos_];
    if (c == '{') { ++pos_; return {TokKind::LBrace,   "", 0}; }
    if (c == '}') { ++pos_; return {TokKind::RBrace,   "", 0}; }
    if (c == '[') { ++pos_; return {TokKind::LBracket, "", 0}; }
    if (c == ']') { ++pos_; return {TokKind::RBracket, "", 0}; }
    if (c == ':') { ++pos_; return {TokKind::Colon,    "", 0}; }
    if (c == ',') { ++pos_; return {TokKind::Comma,    "", 0}; }
    if (c == '"') return read_string();
    if (c == '-' || (c >= '0' && c <= '9')) return read_number();
    if (kw("true"))  return {TokKind::True,  "", 0};
    if (kw("false")) return {TokKind::False, "", 0};
    if (kw("null"))  return {TokKind::Null,  "", 0};
    throw std::runtime_error("CocoDataset: JSON parse error at offset " +
                              std::to_string(pos_) + " near '" + c + "'");
  }

 private:
  const std::string& src_;
  std::size_t pos_ = 0;

  void skip_ws() {
    while (pos_ < src_.size() &&
            (src_[pos_] == ' ' || src_[pos_] == '\t' ||
             src_[pos_] == '\n' || src_[pos_] == '\r'))
      ++pos_;
  }
  bool kw(const char* lit) {
    auto n = std::strlen(lit);
    if (src_.compare(pos_, n, lit) != 0) return false;
    pos_ += n;
    return true;
  }
  Token read_string() {
    Token t; t.kind = TokKind::String;
    ++pos_;  // consume opening "
    while (pos_ < src_.size() && src_[pos_] != '"') {
      if (src_[pos_] == '\\' && pos_ + 1 < src_.size()) {
        char esc = src_[pos_ + 1];
        switch (esc) {
          case '"':  t.sval += '"';  break;
          case '\\': t.sval += '\\'; break;
          case '/':  t.sval += '/';  break;
          case 'n':  t.sval += '\n'; break;
          case 't':  t.sval += '\t'; break;
          case 'r':  t.sval += '\r'; break;
          case 'b':  t.sval += '\b'; break;
          case 'f':  t.sval += '\f'; break;
          // \uXXXX: not needed for COCO field values; skip 4 hex chars.
          case 'u':  if (pos_ + 5 < src_.size()) { t.sval += '?'; pos_ += 4; }
                     break;
          default:   t.sval += esc; break;
        }
        pos_ += 2;
      } else {
        t.sval += src_[pos_++];
      }
    }
    if (pos_ >= src_.size())
      throw std::runtime_error("CocoDataset: unterminated string");
    ++pos_;  // consume closing "
    return t;
  }
  Token read_number() {
    std::size_t start = pos_;
    if (src_[pos_] == '-') ++pos_;
    while (pos_ < src_.size() &&
            (std::isdigit((unsigned char)src_[pos_]) ||
             src_[pos_] == '.' || src_[pos_] == 'e' ||
             src_[pos_] == 'E' || src_[pos_] == '+' || src_[pos_] == '-'))
      ++pos_;
    Token t; t.kind = TokKind::Number;
    t.sval.assign(src_, start, pos_ - start);
    t.nval = std::stod(t.sval);
    return t;
  }
};

// Parser focused on COCO's known structure. We don't build a full
// AST — we just walk to "images", "annotations", "categories" and
// extract fields by name.

struct CocoImageMeta { int id; std::string file_name; int W; int H; };
struct CocoAnnoMeta  { int image_id; int category_id; float x, y, w, h; };
struct CocoCatMeta   { int id; std::string name; };

void expect(JsonLexer& lex, TokKind k, const char* what) {
  auto t = lex.next();
  if (t.kind != k)
    throw std::runtime_error("CocoDataset: expected " + std::string(what));
}

// Skip the value starting at the next token (used to hop over
// fields we don't care about).
void skip_value(JsonLexer& lex);
void skip_array(JsonLexer& lex) {
  while (true) {
    auto t = lex.next();
    if (t.kind == TokKind::RBracket) return;
    if (t.kind == TokKind::Comma)    continue;
    // First token of element already consumed; skip the rest.
    if (t.kind == TokKind::LBrace)   { while (true) { auto u = lex.next();
        if (u.kind == TokKind::RBrace) break; } }
    else if (t.kind == TokKind::LBracket) skip_array(lex);
    // primitives are already consumed
  }
}
void skip_object(JsonLexer& lex) {
  while (true) {
    auto t = lex.next();
    if (t.kind == TokKind::RBrace) return;
    if (t.kind == TokKind::Comma)  continue;
    // expect String key, Colon, then any value
    if (t.kind != TokKind::String) continue;
    expect(lex, TokKind::Colon, ":");
    skip_value(lex);
  }
}
void skip_value(JsonLexer& lex) {
  auto t = lex.next();
  switch (t.kind) {
    case TokKind::LBrace:   skip_object(lex); break;
    case TokKind::LBracket: skip_array(lex);  break;
    default: break;  // primitives are already consumed
  }
}

// Walk an array of objects, calling `f` once per object (which
// itself walks the inner fields).
template <typename F>
void parse_obj_array(JsonLexer& lex, F f) {
  expect(lex, TokKind::LBracket, "[");
  while (true) {
    auto t = lex.next();
    if (t.kind == TokKind::RBracket) break;
    if (t.kind == TokKind::Comma)    continue;
    if (t.kind != TokKind::LBrace)
      throw std::runtime_error("CocoDataset: expected '{' in array");
    f(lex);
  }
}

// Inside an object, walk key/value pairs; call `key_handler(name)`
// for each; the handler MUST consume the value.
template <typename F>
void parse_obj_fields(JsonLexer& lex, F key_handler) {
  while (true) {
    auto t = lex.next();
    if (t.kind == TokKind::RBrace) return;
    if (t.kind == TokKind::Comma)  continue;
    if (t.kind != TokKind::String)
      throw std::runtime_error("CocoDataset: expected field name");
    auto name = t.sval;
    expect(lex, TokKind::Colon, ":");
    key_handler(lex, name);
  }
}

}  // namespace

CocoDataset::CocoDataset(std::string json_path, std::string images_dir,
                          int imgsz, AugConfig aug)
    : imgsz_(imgsz), aug_(std::move(aug)) {
  std::ifstream f(json_path);
  if (!f.is_open())
    throw std::runtime_error("CocoDataset: cannot open " + json_path);
  std::stringstream ss; ss << f.rdbuf();
  std::string body = ss.str();

  if (images_dir.empty()) {
    images_dir = fs::path(json_path).parent_path().string();
  }

  std::vector<CocoImageMeta> images;
  std::vector<CocoAnnoMeta>  annos;
  std::vector<CocoCatMeta>   cats;

  JsonLexer lex(body);
  expect(lex, TokKind::LBrace, "{");
  parse_obj_fields(lex, [&](JsonLexer& l, const std::string& k) {
    if (k == "images") {
      parse_obj_array(l, [&](JsonLexer& ll) {
        CocoImageMeta m{};
        parse_obj_fields(ll, [&](JsonLexer& lll, const std::string& kk) {
          auto t = lll.next();
          if      (kk == "id"        && t.kind == TokKind::Number) m.id = (int)t.nval;
          else if (kk == "file_name" && t.kind == TokKind::String) m.file_name = t.sval;
          else if (kk == "width"     && t.kind == TokKind::Number) m.W = (int)t.nval;
          else if (kk == "height"    && t.kind == TokKind::Number) m.H = (int)t.nval;
          else { /* unknown key — t is a primitive token already consumed,
                    or we need to skip a compound */
            if (t.kind == TokKind::LBrace)   skip_object(lll);
            else if (t.kind == TokKind::LBracket) skip_array(lll);
          }
        });
        images.push_back(m);
      });
    } else if (k == "annotations") {
      parse_obj_array(l, [&](JsonLexer& ll) {
        CocoAnnoMeta m{};
        bool got_bbox = false;
        parse_obj_fields(ll, [&](JsonLexer& lll, const std::string& kk) {
          auto t = lll.next();
          if      (kk == "image_id"    && t.kind == TokKind::Number) m.image_id    = (int)t.nval;
          else if (kk == "category_id" && t.kind == TokKind::Number) m.category_id = (int)t.nval;
          else if (kk == "bbox" && t.kind == TokKind::LBracket) {
            // Read up to 4 numbers, ignore extras, accept comma-separated.
            std::array<float, 4> bb{0,0,0,0};
            int idx = 0;
            while (true) {
              auto u = lll.next();
              if (u.kind == TokKind::RBracket) break;
              if (u.kind == TokKind::Comma)    continue;
              if (u.kind == TokKind::Number && idx < 4) bb[idx++] = (float)u.nval;
            }
            m.x = bb[0]; m.y = bb[1]; m.w = bb[2]; m.h = bb[3];
            got_bbox = true;
          } else {
            if (t.kind == TokKind::LBrace)   skip_object(lll);
            else if (t.kind == TokKind::LBracket) skip_array(lll);
          }
        });
        if (got_bbox) annos.push_back(m);
      });
    } else if (k == "categories") {
      parse_obj_array(l, [&](JsonLexer& ll) {
        CocoCatMeta m{};
        parse_obj_fields(ll, [&](JsonLexer& lll, const std::string& kk) {
          auto t = lll.next();
          if      (kk == "id"   && t.kind == TokKind::Number) m.id = (int)t.nval;
          else if (kk == "name" && t.kind == TokKind::String) m.name = t.sval;
          else {
            if (t.kind == TokKind::LBrace)   skip_object(lll);
            else if (t.kind == TokKind::LBracket) skip_array(lll);
          }
        });
        cats.push_back(m);
      });
    } else {
      skip_value(l);
    }
  });

  if (images.empty())
    throw std::runtime_error("CocoDataset: no images in " + json_path);
  if (cats.empty())
    throw std::runtime_error("CocoDataset: no categories in " + json_path);

  // Compress sparse COCO category IDs to dense [0, N).
  std::unordered_map<int, int> cat_id_to_idx;
  for (int i = 0; i < (int)cats.size(); ++i) {
    cat_id_to_idx[cats[i].id] = i;
    names_.push_back(cats[i].name);
  }

  std::unordered_map<int, std::size_t> img_id_to_idx;
  for (std::size_t i = 0; i < images.size(); ++i) {
    img_id_to_idx[images[i].id] = i;
    fs::path p(images[i].file_name);
    if (p.is_relative()) p = fs::path(images_dir) / p;
    img_paths_.push_back(p.lexically_normal().string());
    labels_.emplace_back();  // placeholder
  }
  std::vector<std::vector<float>> per_image(images.size());

  for (const auto& a : annos) {
    auto it = img_id_to_idx.find(a.image_id);
    if (it == img_id_to_idx.end()) continue;
    auto& m = images[it->second];
    auto cit = cat_id_to_idx.find(a.category_id);
    if (cit == cat_id_to_idx.end()) continue;
    if (m.W <= 0 || m.H <= 0) continue;
    if (a.w <= 0 || a.h <= 0) continue;
    float cx = (a.x + a.w * 0.5f) / (float)m.W;
    float cy = (a.y + a.h * 0.5f) / (float)m.H;
    float w  = a.w / (float)m.W;
    float h  = a.h / (float)m.H;
    auto& v = per_image[it->second];
    v.push_back((float)cit->second);
    v.push_back(cx); v.push_back(cy);
    v.push_back(w);  v.push_back(h);
  }

  // Materialise per-image label tensors. Skip images with no
  // bbox AND no entry: keeping every image around lets training on
  // "negatives" work.
  std::vector<std::string>    kept_paths;
  std::vector<torch::Tensor>  kept_labels;
  std::size_t total_labels = 0;
  for (std::size_t i = 0; i < img_paths_.size(); ++i) {
    auto& v = per_image[i];
    int n = (int)(v.size() / 5);
    kept_paths.push_back(img_paths_[i]);
    if (n == 0) {
      kept_labels.push_back(torch::zeros({0, 5}, torch::kFloat32));
    } else {
      kept_labels.push_back(
          torch::from_blob(v.data(), {n, 5}, torch::kFloat32).clone());
      total_labels += (std::size_t)n;
    }
  }
  img_paths_ = std::move(kept_paths);
  labels_    = std::move(kept_labels);

  std::cout << "[coco] " << json_path << ": "
            << img_paths_.size() << " images, "
            << total_labels << " labels (over "
            << names_.size() << " classes)\n";
}

YoloExample CocoDataset::get(std::size_t idx, std::uint64_t aug_seed) const {
  YoloExample ex;
  ex.img_path = img_paths_[idx];

  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("CocoDataset: failed to load " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;

  std::mt19937 rng(aug_seed ? aug_seed
                              : (std::uint64_t)idx * 0x9E3779B97F4A7C15ULL);
  if (aug_.augment) {
    hsv_jitter(bgr, rng, aug_.hsv_h, aug_.hsv_s, aug_.hsv_v);
  }

  auto lb = inference::letterbox(bgr, imgsz_,
                                  /*pad_color=*/cv::Scalar(114, 114, 114),
                                  /*scale_up=*/false,
                                  /*auto_minrec=*/aug_.rect);
  cv::Mat lb_img = lb.img;
  ex.gain  = lb.gain;
  ex.pad_x = lb.pad_x;
  ex.pad_y = lb.pad_y;

  bool flip = false;
  if (aug_.augment) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    flip = u(rng) < aug_.flip_p;
    if (flip) cv::flip(lb_img, lb_img, /*flipCode=*/1);
  }
  ex.img = inference::image_to_tensor(lb_img);

  const auto& labels = labels_[idx];
  if (labels.size(0) == 0) {
    ex.targets = torch::zeros({0, 5}, torch::kFloat32);
    return ex;
  }
  auto t = labels.clone();
  auto a = t.accessor<float, 2>();
  for (int i = 0; i < (int)t.size(0); ++i) {
    float cx = a[i][1] * (float)ex.orig_w;
    float cy = a[i][2] * (float)ex.orig_h;
    float w  = a[i][3] * (float)ex.orig_w;
    float h  = a[i][4] * (float)ex.orig_h;
    cx = (float)(cx * lb.gain + lb.pad_x);
    cy = (float)(cy * lb.gain + lb.pad_y);
    w  = (float)(w  * lb.gain);
    h  = (float)(h  * lb.gain);
    if (flip) cx = (float)imgsz_ - 1.f - cx;
    a[i][1] = cx; a[i][2] = cy; a[i][3] = w; a[i][4] = h;
  }
  ex.targets = t;
  return ex;
}

CocoDataset::Batch CocoDataset::sample_batch(std::size_t bsz,
                                              std::mt19937& rng) const {
  std::uniform_int_distribution<std::size_t> u(0, img_paths_.size() - 1);
  Batch b;
  std::vector<torch::Tensor> imgs, tgts_with_b;
  for (std::size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), /*aug_seed=*/((std::uint64_t)rng()) << 32 | i);
    imgs.push_back(ex.img);
    if (ex.targets.size(0) > 0) {
      auto bcol = torch::full({ex.targets.size(0), 1}, (int64_t)i,
                                torch::kFloat32);
      tgts_with_b.push_back(torch::cat({bcol, ex.targets}, /*dim=*/1));
    }
    b.examples.push_back(std::move(ex));
  }
  b.imgs = torch::stack(imgs, /*dim=*/0);
  b.targets = tgts_with_b.empty()
                  ? torch::zeros({0, 6}, torch::kFloat32)
                  : torch::cat(tgts_with_b, /*dim=*/0);
  return b;
}

}  // namespace yolocpp::datasets
