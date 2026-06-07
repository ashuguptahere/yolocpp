#include "yolocpp/web/web_ui.hpp"

#include <clay.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Server-side Clay UI. Clay computes the whole layout; `render_cmds` turns its
// render commands into absolutely-positioned HTML/CSS. Form controls are Clay
// CUSTOM elements whose payload is the real <input>/<select>/<button> markup,
// placed at the box Clay computed for them — so the UI is genuinely Clay-laid
// -out, not a hand-written form.

namespace yolocpp::web {
namespace {

using Arena = std::deque<std::string>;  // stable c_str() across push_back

// ── palette (0-255, GitHub-dark-ish) ───────────────────────────────────────
constexpr Clay_Color BG     {13, 17, 23, 255};
constexpr Clay_Color PANEL  {22, 27, 34, 255};
constexpr Clay_Color BORDER {48, 54, 61, 255};
constexpr Clay_Color TEXT   {230, 237, 243, 255};
constexpr Clay_Color MUTED  {139, 148, 158, 255};
constexpr Clay_Color ACCENT {47, 129, 247, 255};

constexpr float kWidth = 1160.f;

// ── Clay struct helpers (imperative — dodges designated-init ordering) ──────
Clay_SizingAxis fixed_(float v) {
  Clay_SizingAxis a{}; a.size.minMax = {v, v}; a.type = CLAY__SIZING_TYPE_FIXED; return a;
}
Clay_SizingAxis grow_(float min = 0) {
  Clay_SizingAxis a{}; a.size.minMax = {min, 0}; a.type = CLAY__SIZING_TYPE_GROW; return a;
}

struct Box {
  Clay_SizingAxis w{}, h{};                       // default {} == FIT
  Clay_Padding pad{};
  uint16_t gap = 0;
  Clay_ChildAlignment align{};
  Clay_LayoutDirection dir = CLAY_TOP_TO_BOTTOM;
  Clay_Color bg{};
  float radius = 0;
  Clay_BorderWidth border{};
  Clay_Color borderColor{};
};

Clay_ElementDeclaration decl(const Box& b) {
  Clay_ElementDeclaration d{};
  d.layout.sizing = {b.w, b.h};
  d.layout.padding = b.pad;
  d.layout.childGap = b.gap;
  d.layout.childAlignment = b.align;
  d.layout.layoutDirection = b.dir;
  d.backgroundColor = b.bg;
  d.cornerRadius = {b.radius, b.radius, b.radius, b.radius};
  d.border.color = b.borderColor;
  d.border.width = b.border;
  return d;
}

void open(const Clay_ElementDeclaration& d) {
  Clay__OpenElement();
  Clay__ConfigureOpenElement(d);
}
void close_() { Clay__CloseElement(); }

Clay_TextElementConfig tcfg(Clay_Color color, uint16_t size) {
  Clay_TextElementConfig c{};
  c.textColor = color;
  c.fontSize = size;
  c.wrapMode = CLAY_TEXT_WRAP_WORDS;
  c.textAlignment = CLAY_TEXT_ALIGN_LEFT;
  return c;
}

void txt(Arena& a, const std::string& s, Clay_TextElementConfig cfg) {
  a.push_back(s);
  Clay_String cs{false, static_cast<int32_t>(a.back().size()), a.back().c_str()};
  // v0.14 takes a config pointer; Clay__StoreTextElementConfig copies it into
  // Clay's internal array and returns a stable pointer valid through layout.
  Clay__OpenTextElement(cs, Clay__StoreTextElementConfig(cfg));
}

// A leaf CUSTOM element carrying raw HTML, sized by Clay.
void custom(Arena& a, std::string html, Clay_SizingAxis w, Clay_SizingAxis h) {
  a.push_back(std::move(html));
  Clay_ElementDeclaration d{};
  d.layout.sizing = {w, h};
  d.custom.customData = const_cast<char*>(a.back().c_str());
  open(d);
  close_();
}

// ── HTML control payloads ───────────────────────────────────────────────────
std::string inp(const std::string& id, const std::string& val,
                const char* type = "text", const char* extra = "") {
  return "<input id='" + id + "' class='ctl' type='" + type + "' value='" + val +
         "' " + extra + ">";
}
std::string sel(const std::string& id, const std::vector<std::string>& opts,
                const std::string& cur) {
  std::string s = "<select id='" + id + "' class='ctl'>";
  for (const auto& o : opts)
    s += "<option" + std::string(o == cur ? " selected" : "") + ">" + o + "</option>";
  return s + "</select>";
}

// ── Clay → HTML renderer ────────────────────────────────────────────────────
std::string esc(const std::string& s) {
  std::string o; o.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': o += "&amp;"; break;
      case '<': o += "&lt;"; break;
      case '>': o += "&gt;"; break;
      case '"': o += "&quot;"; break;
      default: o += c;
    }
  }
  return o;
}
std::string ipx(float v) { return std::to_string(static_cast<int>(v + 0.5f)) + "px"; }
std::string rgba(Clay_Color c) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "rgba(%d,%d,%d,%g)", (int)c.r, (int)c.g, (int)c.b,
                c.a / 255.0);
  return buf;
}
std::string pos(const Clay_BoundingBox& b) {
  return "position:absolute;left:" + ipx(b.x) + ";top:" + ipx(b.y) +
         ";width:" + ipx(b.width) + ";height:" + ipx(b.height) + ";box-sizing:border-box;";
}
std::string radius_css(Clay_CornerRadius r) {
  if (r.topLeft == 0 && r.topRight == 0 && r.bottomLeft == 0 && r.bottomRight == 0)
    return "";
  // CSS order: TL TR BR BL.
  return "border-radius:" + ipx(r.topLeft) + " " + ipx(r.topRight) + " " +
         ipx(r.bottomRight) + " " + ipx(r.bottomLeft) + ";";
}
std::string border_css(const Clay_BorderRenderData& bd) {
  std::string c = rgba(bd.color), s;
  if (bd.width.left)   s += "border-left:"   + ipx(bd.width.left)   + " solid " + c + ";";
  if (bd.width.right)  s += "border-right:"  + ipx(bd.width.right)  + " solid " + c + ";";
  if (bd.width.top)    s += "border-top:"    + ipx(bd.width.top)    + " solid " + c + ";";
  if (bd.width.bottom) s += "border-bottom:" + ipx(bd.width.bottom) + " solid " + c + ";";
  return s;
}

std::string render_cmds(Clay_RenderCommandArray cmds, float& maxBottom) {
  std::string out;
  maxBottom = 0;
  for (int32_t i = 0; i < cmds.length; ++i) {
    Clay_RenderCommand* c = Clay_RenderCommandArray_Get(&cmds, i);
    const Clay_BoundingBox& b = c->boundingBox;
    if (b.y + b.height > maxBottom) maxBottom = b.y + b.height;
    std::string st = pos(b);
    switch (c->commandType) {
      case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
        const auto& r = c->renderData.rectangle;
        st += "background:" + rgba(r.backgroundColor) + ";" + radius_css(r.cornerRadius);
        out += "<div style=\"" + st + "\"></div>";
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_BORDER: {
        const auto& bd = c->renderData.border;
        st += border_css(bd) + radius_css(bd.cornerRadius);
        out += "<div style=\"" + st + "\"></div>";
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_TEXT: {
        const auto& t = c->renderData.text;
        std::string s(t.stringContents.chars, static_cast<size_t>(t.stringContents.length));
        st += "color:" + rgba(t.textColor) +
              ";font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:" +
              ipx(t.fontSize) + ";line-height:" + ipx(b.height) +
              ";white-space:pre;overflow:hidden;";
        out += "<div style=\"" + st + "\">" + esc(s) + "</div>";
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_CUSTOM: {
        const char* p = static_cast<const char*>(c->renderData.custom.customData);
        out += "<div style=\"" + st + "\">" + (p ? std::string(p) : "") + "</div>";
        break;
      }
      default:
        break;  // scissor / image / overlay unused in this UI
    }
  }
  return out;
}

// ── Clay lifecycle ──────────────────────────────────────────────────────────
std::mutex g_mutex;

Clay_Dimensions measure_text(Clay_StringSlice text, Clay_TextElementConfig* cfg, void*) {
  float fs = cfg->fontSize ? cfg->fontSize : 14;
  float cw = fs * 0.62f;  // monospace advance ≈ 0.6em; matches the page font
  float w = text.length * cw + (text.length > 0 ? (text.length - 1) * cfg->letterSpacing : 0);
  float h = cfg->lineHeight ? cfg->lineHeight : fs * 1.35f;
  return Clay_Dimensions{w, h};
}
void clay_err(Clay_ErrorData e) {
  std::fprintf(stderr, "[clay] %.*s\n", e.errorText.length, e.errorText.chars);
}
void init_clay_once() {
  static std::vector<char> mem;
  static bool done = false;
  if (done) return;
  uint32_t need = Clay_MinMemorySize();
  mem.resize(need);
  Clay_Arena arena = Clay_CreateArenaWithCapacityAndMemory(need, mem.data());
  Clay_Initialize(arena, Clay_Dimensions{kWidth, 4000.f}, Clay_ErrorHandler{clay_err, nullptr});
  Clay_SetMeasureTextFunction(measure_text, nullptr);
  done = true;
}

// ── page template ───────────────────────────────────────────────────────────
const char* kCss = R"CSS(
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#e6edf3;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.stage{position:relative;margin:24px auto}
.ctl{width:100%;height:100%;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:0 8px;font:13px ui-monospace,monospace}
.ctl:focus{outline:none;border-color:#2f81f7}
.grp{display:flex;gap:8px;width:100%;height:100%}.grp>*{flex:1}
.btn{width:100%;height:100%;border:0;border-radius:6px;background:#21262d;color:#e6edf3;font:600 13px ui-monospace,monospace;cursor:pointer}
.btn:hover{background:#2a3038}.btn.run{background:#238636}.btn.run:hover{background:#2ea043}
.jobs{font:12px ui-monospace,monospace;color:#c9d1d9;overflow:auto}
.job{border:1px solid #30363d;border-radius:6px;padding:8px 10px;margin-bottom:8px}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;margin:0 4px}
.b-running{background:#1f6feb33;color:#58a6ff}.b-done{background:#23863633;color:#3fb950}
.b-error{background:#da363333;color:#f85149}.b-queued{background:#6e768166;color:#8b949e}
.job pre{white-space:pre-wrap;margin-top:6px;color:#8b949e;max-height:170px;overflow:auto}
.job a{color:#58a6ff}
)CSS";

const char* kJs = R"JS(
const $=id=>document.getElementById(id);
function gather(){return{model:$('f_model').value,task:$('f_task').value,source:$('f_source').value,
 data:$('f_data').value,epochs:$('f_epochs').value,batch:$('f_batch').value,imgsz:$('f_imgsz').value,
 device:$('f_device').value,conf:$('f_conf').value,format:$('f_format').value,precision:$('f_precision').value};}
async function runJob(mode){
 const b=new URLSearchParams(gather()).toString();
 await fetch('/api/'+mode,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:b});
 poll();
}
const E=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const badge=s=>`<span class="badge b-${s}">${s}</span>`;
async function poll(){
 try{const r=await fetch('/api/jobs');const js=await r.json();
  $('jobs').innerHTML=js.length?js.map(j=>`<div class="job"><b>#${j.id}</b> ${j.mode} ${badge(j.status)}`+
   `<span style="color:#8b949e">${E(j.model)}</span><div>${E(j.message)}</div>`+
   (j.log?`<pre>${E(j.log)}</pre>`:'')+`</div>`).join(''):
   '<div style="color:#8b949e">No jobs yet — configure on the left and launch one.</div>';
 }catch(e){}
}
setInterval(poll,1500);poll();
)JS";

std::string page(const std::string& body, float height) {
  return std::string("<!doctype html><html lang=en><head><meta charset=utf-8>") +
         "<meta name=viewport content=\"width=device-width,initial-scale=1\">" +
         "<title>yolocpp · web console</title><style>" + kCss + "</style></head><body>" +
         "<div class=stage style=\"width:" + ipx(kWidth) + ";height:" + ipx(height) + "\">" +
         body + "</div><script>" + kJs + "</script></body></html>";
}

}  // namespace

std::string render_dashboard(const DashboardModel& m) {
  std::lock_guard<std::mutex> lock(g_mutex);
  init_clay_once();
  Clay_SetLayoutDimensions(Clay_Dimensions{kWidth, 4000.f});

  Arena arena;
  Clay_BeginLayout();

  auto field = [&](const std::string& label, std::string ctl, float h = 34) {
    Box row; row.w = grow_(); row.h = fixed_(h); row.dir = CLAY_LEFT_TO_RIGHT;
    row.gap = 12; row.align = {CLAY_ALIGN_X_LEFT, CLAY_ALIGN_Y_CENTER};
    open(decl(row));
      Box lab; lab.w = fixed_(154); lab.h = fixed_(h); lab.dir = CLAY_LEFT_TO_RIGHT;
      lab.align = {CLAY_ALIGN_X_LEFT, CLAY_ALIGN_Y_CENTER};
      open(decl(lab)); txt(arena, label, tcfg(MUTED, 13)); close_();
      custom(arena, std::move(ctl), grow_(), fixed_(h));
    close_();
  };

  Box root; root.w = fixed_(kWidth); root.dir = CLAY_TOP_TO_BOTTOM; root.bg = BG;
  open(decl(root));

  // header
  Box hd; hd.w = grow_(); hd.h = fixed_(60); hd.dir = CLAY_LEFT_TO_RIGHT;
  hd.pad = {20, 20, 0, 0}; hd.gap = 12; hd.align = {CLAY_ALIGN_X_LEFT, CLAY_ALIGN_Y_CENTER};
  hd.bg = PANEL; hd.border = {0, 0, 0, 1}; hd.borderColor = BORDER;
  open(decl(hd));
    txt(arena, "yolocpp", tcfg(ACCENT, 22));
    txt(arena, "web console", tcfg(MUTED, 13));
    { Box sp; sp.w = grow_(); open(decl(sp)); close_(); }
    txt(arena, "v" + m.version, tcfg(MUTED, 12));
  close_();

  // body
  Box body; body.w = grow_(); body.dir = CLAY_LEFT_TO_RIGHT; body.pad = {16, 16, 16, 16};
  body.gap = 16; body.bg = BG;
  open(decl(body));

    // left: form
    Box left; left.w = fixed_(560); left.dir = CLAY_TOP_TO_BOTTOM; left.pad = {18, 18, 18, 18};
    left.gap = 10; left.bg = PANEL; left.radius = 10; left.border = {1, 1, 1, 1};
    left.borderColor = BORDER;
    open(decl(left));
      txt(arena, "Run a job", tcfg(TEXT, 16));
      txt(arena, "train / val / predict / export on this box's GPU", tcfg(MUTED, 12));

      std::string modelCtl = "<input id='f_model' class='ctl' list='models' value='yolo11n.pt'><datalist id='models'>";
      for (const auto& mm : m.models) modelCtl += "<option value='" + mm + "'>";
      modelCtl += "</datalist>";

      field("model / weights", modelCtl);
      field("task", sel("f_task", {"detect", "classify", "segment", "pose", "obb"}, "detect"));
      field("source (predict)", inp("f_source", "data/bus.jpg"));
      field("dataset (train/val)", inp("f_data", ""));
      field("epochs / batch / imgsz",
            "<div class='grp'><input id='f_epochs' class='ctl' type='number' value='1'>"
            "<input id='f_batch' class='ctl' type='number' value='16'>"
            "<input id='f_imgsz' class='ctl' type='number' value='640'></div>");
      field("device", inp("f_device", "auto"));
      field("confidence (predict)", inp("f_conf", "0.25", "number", "step=0.05 min=0 max=1"));
      field("export format / precision",
            "<div class='grp'><select id='f_format' class='ctl'><option>onnx</option><option>trt</option></select>"
            "<select id='f_precision' class='ctl'><option>fp16</option><option>fp32</option><option>int8</option></select></div>");

      Box br; br.w = grow_(); br.h = fixed_(40); br.dir = CLAY_LEFT_TO_RIGHT; br.gap = 10;
      open(decl(br));
        custom(arena, "<button class='btn run' onclick=\"runJob('predict')\">Predict</button>", grow_(), fixed_(40));
        custom(arena, "<button class='btn' onclick=\"runJob('val')\">Validate</button>", grow_(), fixed_(40));
        custom(arena, "<button class='btn' onclick=\"runJob('train')\">Train</button>", grow_(), fixed_(40));
        custom(arena, "<button class='btn' onclick=\"runJob('export')\">Export</button>", grow_(), fixed_(40));
      close_();
    close_();

    // right: jobs
    Box right; right.w = grow_(); right.dir = CLAY_TOP_TO_BOTTOM; right.pad = {18, 18, 18, 18};
    right.gap = 10; right.bg = PANEL; right.radius = 10; right.border = {1, 1, 1, 1};
    right.borderColor = BORDER;
    open(decl(right));
      txt(arena, "Jobs", tcfg(TEXT, 16));
      custom(arena, "<div id='jobs' class='jobs'></div>", grow_(), grow_(440));
    close_();

  close_();   // body
  close_();   // root

  float maxBottom = 0;
  std::string body_html = render_cmds(Clay_EndLayout(), maxBottom);
  return page(body_html, maxBottom);
}

}  // namespace yolocpp::web
