// Self-contained ONNX exporter for YOLO8.
//
// We emit the ONNX protobuf wire format directly. The encoding is
// straightforward enough that bringing in libprotobuf would cost more than
// it saves. References:
//   • https://protobuf.dev/programming-guides/encoding/
//   • https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
//
// Field numbers below are taken from onnx.proto rev for opset 17 and are
// stable across recent ONNX versions.

#include "yolocpp/serialization/onnx_export.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolocpp::serialization {

namespace {

// ──────────────────────────────────────────────────────────────────────────
// Protobuf wire-format encoder (varints, length-delimited, fixed64).

class Pb {
 public:
  std::string& bytes() { return data_; }

  void write_varint(uint64_t v) {
    while (v >= 0x80) {
      data_.push_back((char)((v & 0x7f) | 0x80));
      v >>= 7;
    }
    data_.push_back((char)v);
  }
  void write_tag(uint32_t field, uint8_t wire) {
    write_varint(((uint64_t)field << 3) | wire);
  }
  // Length-delimited field (wire type 2): tag + length + bytes
  void write_bytes_field(uint32_t field, const std::string& s) {
    write_tag(field, 2);
    write_varint((uint64_t)s.size());
    data_.append(s);
  }
  void write_string_field(uint32_t field, const std::string& s) {
    write_bytes_field(field, s);
  }
  void write_int64_field(uint32_t field, int64_t v) {
    write_tag(field, 0);
    // protobuf int64 is stored as varint; signed values cast to uint64.
    write_varint((uint64_t)v);
  }
  void write_int32_field(uint32_t field, int32_t v) {
    write_tag(field, 0);
    write_varint((uint64_t)(int64_t)v);
  }
  // packed repeated int64: encode all values as varints into a single
  // length-delimited record.
  void write_packed_int64s(uint32_t field, const std::vector<int64_t>& xs) {
    Pb inner;
    for (auto v : xs) inner.write_varint((uint64_t)v);
    write_bytes_field(field, inner.data_);
  }
  // Repeated string: emit one length-delimited field per element.
  void write_repeated_string(uint32_t field, const std::vector<std::string>& xs) {
    for (auto& s : xs) write_bytes_field(field, s);
  }
  void write_raw(const std::string& raw) { data_ += raw; }

 private:
  std::string data_;
};

// Map at::ScalarType → ONNX TensorProto.DataType enum.
//   1 FLOAT, 2 UINT8, 3 INT8, 4 UINT16, 5 INT16, 6 INT32, 7 INT64,
//   9 BOOL, 10 FLOAT16, 11 DOUBLE, 16 BFLOAT16
int onnx_dtype(c10::ScalarType s) {
  switch (s) {
    case c10::ScalarType::Float:    return 1;
    case c10::ScalarType::Byte:     return 2;
    case c10::ScalarType::Char:     return 3;
    case c10::ScalarType::Short:    return 5;
    case c10::ScalarType::Int:      return 6;
    case c10::ScalarType::Long:     return 7;
    case c10::ScalarType::Bool:     return 9;
    case c10::ScalarType::Half:     return 10;
    case c10::ScalarType::Double:   return 11;
    case c10::ScalarType::BFloat16: return 16;
    default:
      throw std::runtime_error("unsupported dtype for ONNX export");
  }
}

// ──────────────────────────────────────────────────────────────────────────
// ONNX message builders.

// AttributeProto field numbers (from onnx.proto — IMPORTANT: numbers are
// NOT contiguous):
//   1  name (string)
//   2  f (float)
//   3  i (int64)
//   4  s (bytes/string)
//   5  t (TensorProto)
//   6  g (GraphProto)
//   7  floats (repeated float)
//   8  ints (repeated int64)
//   9  strings (repeated bytes)
//   10 tensors (repeated TensorProto)
//   13 doc_string
//   20 type (AttributeType enum):
//      0 UNDEFINED, 1 FLOAT, 2 INT, 3 STRING, 4 TENSOR, 5 GRAPH,
//      6 FLOATS, 7 INTS, 8 STRINGS, 9 TENSORS, 10 GRAPHS,
//      11 SPARSE_TENSOR, 12 SPARSE_TENSORS, 13 TYPE_PROTO, 14 TYPE_PROTOS
//   21 ref_attr_name
//
// Older ONNX parsers (TensorRT's nvonnxparser, older onnxruntime) infer the
// `type` by which value field is set, so a missing `type=20` is forgiven.
// Modern `onnx.checker.check_model` requires it, so we always emit it.

std::string attr_int(const std::string& name, int64_t v) {
  Pb a;
  a.write_string_field(1, name);
  a.write_int64_field(3, v);              // 'i'
  a.write_int32_field(20, /*INT=*/2);     // type
  return a.bytes();
}
std::string attr_ints(const std::string& name, const std::vector<int64_t>& xs) {
  Pb a;
  a.write_string_field(1, name);
  a.write_packed_int64s(8, xs);           // 'ints'
  a.write_int32_field(20, /*INTS=*/7);    // type
  return a.bytes();
}
std::string attr_float(const std::string& name, float v) {
  Pb a;
  a.write_string_field(1, name);
  // 'f' = field 2, wire type 5 (fixed32).
  a.write_tag(2, 5);
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  a.bytes().push_back((char)(bits & 0xff));
  a.bytes().push_back((char)((bits >> 8) & 0xff));
  a.bytes().push_back((char)((bits >> 16) & 0xff));
  a.bytes().push_back((char)((bits >> 24) & 0xff));
  a.write_int32_field(20, /*FLOAT=*/1);   // type
  return a.bytes();
}
std::string attr_string(const std::string& name, const std::string& v) {
  Pb a;
  a.write_string_field(1, name);
  a.write_bytes_field(4, v);              // 's'
  a.write_int32_field(20, /*STRING=*/3);  // type
  return a.bytes();
}

// TensorProto fields:
//   1 dims (repeated int64, packed)
//   2 data_type (int32)
//   3 segment (skipped)
//   4 float_data (packed repeated float)
//   ...
//   8 name (string)
//   9 raw_data (bytes)
//   12 doc_string (skipped)
std::string tensor_proto(const std::string& name,
                         const std::vector<int64_t>& dims,
                         int data_type,
                         const void* raw_data, size_t raw_data_len) {
  Pb t;
  t.write_packed_int64s(1, dims);
  t.write_int32_field(2, data_type);
  t.write_string_field(8, name);
  // raw_data
  t.write_tag(9, 2);
  t.write_varint((uint64_t)raw_data_len);
  t.bytes().append(reinterpret_cast<const char*>(raw_data), raw_data_len);
  return t.bytes();
}

std::string tensor_proto_from_at(const std::string& name,
                                 const at::Tensor& tt) {
  auto t = tt.detach().to(torch::kCPU).contiguous();
  std::vector<int64_t> dims(t.sizes().begin(), t.sizes().end());
  return tensor_proto(name, dims, onnx_dtype(t.scalar_type()),
                      t.data_ptr(), t.numel() * t.element_size());
}

// NodeProto:
//   1 input (repeated string)
//   2 output (repeated string)
//   3 name (string)
//   4 op_type (string)
//   5 attribute (repeated AttributeProto)
//   7 domain (string)
std::string node_proto(const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs,
                       const std::string& op_type,
                       const std::string& name,
                       const std::vector<std::string>& attrs = {}) {
  Pb n;
  n.write_repeated_string(1, inputs);
  n.write_repeated_string(2, outputs);
  if (!name.empty()) n.write_string_field(3, name);
  n.write_string_field(4, op_type);
  for (const auto& a : attrs) n.write_bytes_field(5, a);
  return n.bytes();
}

// ValueInfoProto:
//   1 name (string)
//   2 type (TypeProto)
//
// TypeProto:
//   1 tensor_type (Tensor)
//
// Tensor:
//   1 elem_type (int32)
//   2 shape (TensorShapeProto)
//
// TensorShapeProto:
//   1 dim (repeated Dimension)
// Dimension:
//   1 dim_value (int64)
//   2 dim_param (string)
std::string value_info(const std::string& name,
                       int elem_type,
                       const std::vector<int64_t>& shape) {
  // Build inner Tensor.shape
  Pb shape_pb;
  for (auto d : shape) {
    Pb dim;
    if (d >= 0) dim.write_int64_field(1, d);
    else        dim.write_string_field(2, "N");  // dynamic batch placeholder
    shape_pb.write_bytes_field(1, dim.bytes());
  }
  Pb tensor;
  tensor.write_int32_field(1, elem_type);
  tensor.write_bytes_field(2, shape_pb.bytes());
  Pb type;
  type.write_bytes_field(1, tensor.bytes());
  Pb vi;
  vi.write_string_field(1, name);
  vi.write_bytes_field(2, type.bytes());
  return vi.bytes();
}

// GraphProto:
//   1 node (repeated NodeProto)
//   2 name (string)
//   5 initializer (repeated TensorProto)
//   11 input (repeated ValueInfoProto)
//   12 output (repeated ValueInfoProto)
//   13 value_info (repeated ValueInfoProto)
//
// ModelProto:
//   1 ir_version (int64)
//   2 opset_import (repeated OperatorSetIdProto) — but field 8
//   3 producer_name
//   4 producer_version
//   5 domain
//   6 model_version
//   7 doc_string
//   8 graph (GraphProto)
//   14 metadata_props
// Wait: in onnx.proto the actual numbering is:
//   1 ir_version, 8 opset_import, 2 producer_name, 3 producer_version,
//   4 domain, 5 model_version, 6 doc_string, 7 graph, ...
//
// We follow the actual onnx.proto layout (verified against onnx 1.16):
//   1 ir_version (int64)
//   2 producer_name (string)
//   3 producer_version (string)
//   4 domain (string)
//   5 model_version (int64)
//   6 doc_string (string)
//   7 graph (GraphProto)
//   8 opset_import (repeated OperatorSetIdProto)
//   14 metadata_props
//
// OperatorSetIdProto:
//   1 domain (string), 2 version (int64)
std::string opset_id(const std::string& domain, int64_t version) {
  Pb o;
  if (!domain.empty()) o.write_string_field(1, domain);
  o.write_int64_field(2, version);
  return o.bytes();
}

// ──────────────────────────────────────────────────────────────────────────
// Graph builder: incrementally accumulates nodes/initializers/etc.

class GraphBuilder {
 public:
  // Add an initializer. Returns the tensor's name.
  std::string add_init(const std::string& name, const at::Tensor& t) {
    inits_.push_back(tensor_proto_from_at(name, t));
    return name;
  }
  std::string add_init_int64(const std::string& name,
                             const std::vector<int64_t>& v) {
    auto t = torch::tensor(v, torch::kLong);
    inits_.push_back(tensor_proto_from_at(name, t));
    return name;
  }
  std::string add_init_float(const std::string& name,
                             const std::vector<float>& v,
                             const std::vector<int64_t>& shape) {
    auto t = torch::from_blob(const_cast<float*>(v.data()), shape,
                              torch::kFloat32).clone();
    inits_.push_back(tensor_proto_from_at(name, t));
    return name;
  }

  // Add a node and return its (single) output name.
  std::string node(const std::string& op_type,
                   const std::vector<std::string>& inputs,
                   const std::vector<std::string>& attrs = {},
                   std::string out_name = "") {
    if (out_name.empty()) out_name = unique(op_type);
    nodes_.push_back(node_proto(inputs, {out_name}, op_type,
                                /*name=*/out_name, attrs));
    return out_name;
  }

  void set_input(const std::string& name, int dt,
                 const std::vector<int64_t>& shape) {
    input_name_  = name;
    input_dt_    = dt;
    input_shape_ = shape;
  }
  // Single-output convenience: clears any previously declared outputs.
  void set_output(const std::string& name, int dt,
                  const std::vector<int64_t>& shape) {
    if (outputs_.empty()) {
      outputs_.push_back({name, dt, shape});
      return;
    }
    // If the name was already added (multi-output sequence), update its
    // type/shape; otherwise append.
    for (auto& o : outputs_) {
      if (o.name == name) { o.dt = dt; o.shape = shape; return; }
    }
    outputs_.push_back({name, dt, shape});
  }
  // Re-export the last node under a new tensor name. Used when emitting a
  // multi-output head: the cv4 chain returns its concat name; we rename it
  // to "coefs"/"keypoints"/"angle"/"protos" via an Identity passthrough so
  // it appears as a graph output with that name.
  std::string rename_output(const std::string& src,
                            const std::string& new_name) {
    return node("Identity", {src}, {}, new_name);
  }

  std::string build_graph_bytes(const std::string& graph_name) const {
    Pb g;
    for (const auto& n : nodes_)        g.write_bytes_field(1, n);
    g.write_string_field(2, graph_name);
    for (const auto& t : inits_)        g.write_bytes_field(5, t);
    g.write_bytes_field(11, value_info(input_name_,  input_dt_,  input_shape_));
    for (const auto& o : outputs_) {
      g.write_bytes_field(12, value_info(o.name, o.dt, o.shape));
    }
    return g.bytes();
  }

  std::string unique(const std::string& base) {
    return base + "_" + std::to_string(seq_++);
  }

 private:
  struct Output {
    std::string name;
    int dt;
    std::vector<int64_t> shape;
  };
  std::vector<std::string> nodes_;
  std::vector<std::string> inits_;
  std::string              input_name_;
  int                      input_dt_ = 1;
  std::vector<int64_t>     input_shape_;
  std::vector<Output>      outputs_;
  int                      seq_ = 0;
};

// ──────────────────────────────────────────────────────────────────────────
// YOLO8 → ONNX: the layer emitters.

// Fold BN into preceding Conv weights & bias (since our Conv has bias=False
// it becomes pure conv-with-bias post-fusion).
struct FusedConv {
  at::Tensor weight;  // [Cout, Cin/groups, kH, kW]
  at::Tensor bias;    // [Cout]
};
FusedConv fuse_conv_bn(const at::Tensor& cw,
                       const at::Tensor& bw, const at::Tensor& bb,
                       const at::Tensor& brm, const at::Tensor& brv,
                       double eps) {
  // eps must match what the BN was constructed with (read it via
  // module->bn->options.eps() at the call site). Ultralytics uses 1e-3
  // for detect/seg/pose/obb and 1e-5 for the *cls models — passing the
  // wrong value silently re-introduces the ~3% per-BN scale drift bug
  // we fixed at the libtorch level.
  auto bn_scale = bw / torch::sqrt(brv + eps);   // [Cout]
  auto fw = cw * bn_scale.view({-1, 1, 1, 1});
  auto fb = bb - brm * bn_scale;
  return {fw.to(torch::kFloat32), fb.to(torch::kFloat32)};
}

// Emit Conv + (optional) SiLU. Returns output tensor name.
std::string emit_conv_block(GraphBuilder& g, const std::string& in,
                            const std::string& prefix,
                            const at::Tensor& cw,
                            const at::Tensor& bw, const at::Tensor& bb,
                            const at::Tensor& brm, const at::Tensor& brv,
                            int stride, int padding, int groups,
                            bool act_silu, double bn_eps) {
  auto fused = fuse_conv_bn(cw, bw, bb, brm, brv, bn_eps);
  auto wname = g.add_init(prefix + ".weight", fused.weight);
  auto bname = g.add_init(prefix + ".bias",   fused.bias);

  std::vector<std::string> attrs = {
      attr_ints("dilations", {1, 1}),
      attr_int("group", groups),
      attr_ints("kernel_shape", {fused.weight.size(2), fused.weight.size(3)}),
      attr_ints("pads", {padding, padding, padding, padding}),
      attr_ints("strides", {stride, stride}),
  };
  auto y = g.node("Conv", {in, wname, bname}, attrs, prefix + ".out");
  if (!act_silu) return y;
  // SiLU = x * sigmoid(x). Could also use HardSigmoid but YOLO8 uses SiLU.
  auto sig = g.node("Sigmoid", {y}, {}, prefix + ".sig");
  auto out = g.node("Mul",     {y, sig}, {}, prefix + ".silu");
  return out;
}

// Emit Bottleneck. Returns output name.
std::string emit_bottleneck(GraphBuilder& g, const std::string& in,
                            const std::string& prefix,
                            models::BottleneckImpl* bn,
                            int /*c_in*/, int /*c_out*/) {
  // bn->cv1: Conv (3x3, stride 1, pad 1)
  // bn->cv2: Conv (3x3, stride 1, pad 1)
  // bn->add: residual if input/output channels match
  auto* cv1 = bn->cv1.get();
  auto* cv2 = bn->cv2.get();
  auto y1 = emit_conv_block(g, in, prefix + ".cv1",
                            cv1->conv->weight, cv1->bn->weight,
                            cv1->bn->bias, cv1->bn->running_mean,
                            cv1->bn->running_var,
                            cv1->conv->options.stride()->at(0),
                            std::get<torch::ExpandingArray<2>>(cv1->conv->options.padding())->at(0),
                            (int)cv1->conv->options.groups(), true,
                            cv1->bn->options.eps());
  auto y2 = emit_conv_block(g, y1, prefix + ".cv2",
                            cv2->conv->weight, cv2->bn->weight,
                            cv2->bn->bias, cv2->bn->running_mean,
                            cv2->bn->running_var,
                            cv2->conv->options.stride()->at(0),
                            std::get<torch::ExpandingArray<2>>(cv2->conv->options.padding())->at(0),
                            (int)cv2->conv->options.groups(), true,
                            cv2->bn->options.eps());
  if (bn->add) {
    return g.node("Add", {in, y2}, {}, prefix + ".add");
  }
  return y2;
}

std::string emit_conv_module(GraphBuilder& g, const std::string& in,
                             const std::string& prefix,
                             models::ConvImpl* m) {
  auto* c = m->conv.get();
  return emit_conv_block(g, in, prefix,
                         c->weight, m->bn->weight, m->bn->bias,
                         m->bn->running_mean, m->bn->running_var,
                         c->options.stride()->at(0),
                         std::get<torch::ExpandingArray<2>>(c->options.padding())->at(0),
                         (int)c->options.groups(),
                         m->act_silu,
                         m->bn->options.eps());
}

std::string emit_split2(GraphBuilder& g, const std::string& in,
                        const std::string& prefix, int total_c) {
  // Split input along channel into 2 equal halves.
  // We emit two Slice ops because Split with `num_outputs` semantics differ
  // by opset version; Slice is more portable.
  // a = input[:, 0:total_c/2, :, :]
  // b = input[:, total_c/2:total_c, :, :]
  auto half = total_c / 2;
  auto starts0 = g.add_init_int64(prefix + ".s0", {0});
  auto ends0   = g.add_init_int64(prefix + ".e0", {(int64_t)half});
  auto axes    = g.add_init_int64(prefix + ".ax", {1});
  auto steps   = g.add_init_int64(prefix + ".st", {1});
  g.node("Slice", {in, starts0, ends0, axes, steps}, {}, prefix + ".a");

  auto starts1 = g.add_init_int64(prefix + ".s1", {(int64_t)half});
  auto ends1   = g.add_init_int64(prefix + ".e1", {(int64_t)total_c});
  g.node("Slice", {in, starts1, ends1, axes, steps}, {}, prefix + ".b");
  return prefix;  // caller refers to ".a" / ".b"
}

std::string emit_c2f(GraphBuilder& g, const std::string& in,
                     const std::string& prefix, models::C2fImpl* m) {
  // 1) cv1: 1x1 Conv → 2*c_inner
  auto y = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  // 2) Split into two halves a, b along channel.
  int total_c = 2 * m->c_inner;
  emit_split2(g, y, prefix + ".split", total_c);
  std::string a = prefix + ".split.a";
  std::string b = prefix + ".split.b";
  std::vector<std::string> outs = {a, b};
  // 3) Run each Bottleneck on `last` and append.
  std::string last = b;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto* bot = m->m[i]->as<models::BottleneckImpl>();
    last = emit_bottleneck(g, last,
                           prefix + ".m." + std::to_string(i),
                           bot, m->c_inner, m->c_inner);
    outs.push_back(last);
  }
  // 4) Concat along channel and 1x1 Conv.
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)},
                    prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

std::string emit_sppf(GraphBuilder& g, const std::string& in,
                      const std::string& prefix, models::SPPFImpl* m) {
  auto y0 = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  // MaxPool 5x5 stride 1 pad 2, applied 3 times sequentially.
  std::vector<std::string> outs = {y0};
  std::string cur = y0;
  for (int i = 0; i < 3; ++i) {
    cur = g.node("MaxPool", {cur},
                 {attr_ints("kernel_shape", {5, 5}),
                  attr_ints("strides",      {1, 1}),
                  attr_ints("pads",         {2, 2, 2, 2})},
                 prefix + ".mp" + std::to_string(i));
    outs.push_back(cur);
  }
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

std::string emit_upsample_2x(GraphBuilder& g, const std::string& in,
                             const std::string& prefix) {
  // ONNX Resize op. We provide empty roi (deprecated) and `scales`
  // [1, 1, 2, 2] for nearest 2x upsample.
  auto roi    = g.add_init_float(prefix + ".roi", {}, {0});
  auto scales = g.add_init_float(prefix + ".scales", {1.f, 1.f, 2.f, 2.f}, {4});
  return g.node("Resize", {in, roi, scales},
                {attr_string("mode", "nearest"),
                 attr_string("nearest_mode", "floor"),
                 attr_string("coordinate_transformation_mode", "asymmetric")},
                prefix + ".up");
}

// Detect head: emits cv2 (regression branch), cv3 (cls branch), DFL projection,
// distance→xyxy decoding using anchor/stride initializers.
//
// Outputs a single tensor named output_name: [N, 4 + nc, A]
std::string emit_detect(GraphBuilder& g,
                        const std::vector<std::string>& detect_ins,
                        const std::vector<int>&           detect_in_ch,
                        const std::vector<double>&        strides,
                        models::DetectImpl* d, int imgsz,
                        const std::string& prefix,
                        const std::string& final_out_name) {
  int nl      = d->nl;
  int nc      = d->nc;
  int reg_max = d->reg_max;
  int no      = nc + 4 * reg_max;

  // Per-level: cv2 (3x3 conv, 3x3 conv, 1x1 conv → 4*reg_max) and
  //            cv3 (3x3 conv, 3x3 conv, 1x1 conv → nc).
  std::vector<std::string> level_outs;        // [N, no, h*w] per level
  std::vector<int>         spatial_per_level; // h*w
  std::vector<std::vector<float>> anchors_xy; // (h*w, 2)

  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    int hw     = feat_h * feat_w;
    spatial_per_level.push_back(hw);

    // Build anchor centers for this level.
    std::vector<float> a;
    a.reserve(hw * 2);
    for (int y = 0; y < feat_h; ++y) {
      for (int x = 0; x < feat_w; ++x) {
        a.push_back(((float)x + 0.5f) * (float)strides[i]);
        a.push_back(((float)y + 0.5f) * (float)strides[i]);
      }
    }
    anchors_xy.push_back(std::move(a));

    // Get the SequentialImpl for cv2[i] and cv3[i].
    auto* reg = d->cv2[i]->as<torch::nn::SequentialImpl>();
    auto* cls = d->cv3[i]->as<torch::nn::SequentialImpl>();

    // Reg branch: Conv (3x3) → Conv (3x3) → Conv2d (1x1).
    // First two are our Conv (with BN+SiLU); third is plain Conv2d (no BN).
    auto* r0 = reg->ptr(0)->as<models::ConvImpl>();
    auto* r1 = reg->ptr(1)->as<models::ConvImpl>();
    auto  r2 = reg->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_r = emit_conv_module(g, detect_ins[i],
                                prefix + ".cv2." + std::to_string(i) + ".0", r0);
    y_r = emit_conv_module(g, y_r,
                           prefix + ".cv2." + std::to_string(i) + ".1", r1);
    auto rw = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.weight",
                         r2->weight);
    auto rb = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.bias",
                         r2->bias);
    y_r = g.node("Conv", {y_r, rw, rb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv2." + std::to_string(i) + ".2.out");

    // Cls branch.
    auto* c0 = cls->ptr(0)->as<models::ConvImpl>();
    auto* c1 = cls->ptr(1)->as<models::ConvImpl>();
    auto  c2 = cls->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_c = emit_conv_module(g, detect_ins[i],
                                prefix + ".cv3." + std::to_string(i) + ".0", c0);
    y_c = emit_conv_module(g, y_c,
                           prefix + ".cv3." + std::to_string(i) + ".1", c1);
    auto cw = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.weight",
                         c2->weight);
    auto cb = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.bias",
                         c2->bias);
    y_c = g.node("Conv", {y_c, cw, cb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv3." + std::to_string(i) + ".2.out");

    // Concat reg + cls along channel → [N, no, h, w]
    auto cat = g.node("Concat", {y_r, y_c}, {attr_int("axis", 1)},
                      prefix + ".level." + std::to_string(i) + ".cat");
    // Reshape to [N, no, h*w].
    auto rshape = g.add_init_int64(
        prefix + ".level." + std::to_string(i) + ".rshape",
        {0, no, hw});
    auto flat = g.node("Reshape", {cat, rshape}, {},
                       prefix + ".level." + std::to_string(i) + ".flat");
    level_outs.push_back(flat);
  }

  // Concat all levels along anchor dim → [N, no, A]
  auto pred = g.node("Concat", level_outs, {attr_int("axis", 2)},
                     prefix + ".pred");

  // Slice into box (4*reg_max channels) and cls (nc channels).
  auto axes_ch  = g.add_init_int64(prefix + ".axch",  {1});
  auto starts_b = g.add_init_int64(prefix + ".sb",    {0});
  auto ends_b   = g.add_init_int64(prefix + ".eb",    {(int64_t)(4 * reg_max)});
  auto steps_1  = g.add_init_int64(prefix + ".st1",   {1});
  auto box      = g.node("Slice", {pred, starts_b, ends_b, axes_ch, steps_1}, {},
                          prefix + ".box");

  auto starts_c = g.add_init_int64(prefix + ".sc",    {(int64_t)(4 * reg_max)});
  auto ends_c   = g.add_init_int64(prefix + ".ec",    {(int64_t)no});
  auto cls      = g.node("Slice", {pred, starts_c, ends_c, axes_ch, steps_1}, {},
                          prefix + ".cls");

  // Sigmoid the cls branch.
  auto cls_sig = g.node("Sigmoid", {cls}, {}, prefix + ".cls.sig");

  // DFL: reshape box to [N, 4, reg_max, A] → softmax(reg_max) → expectation.
  int total_A = 0;
  for (int hw : spatial_per_level) total_A += hw;
  auto dfl_rshape = g.add_init_int64(prefix + ".dfl.rshape",
                                     {0, 4, (int64_t)reg_max, (int64_t)total_A});
  auto dfl_box    = g.node("Reshape", {box, dfl_rshape}, {},
                            prefix + ".dfl.box");
  auto dfl_soft   = g.node("Softmax", {dfl_box}, {attr_int("axis", 2)},
                            prefix + ".dfl.soft");
  // arange(reg_max) reshaped to [1, 1, reg_max, 1]
  std::vector<float> proj_v(reg_max);
  for (int i = 0; i < reg_max; ++i) proj_v[i] = (float)i;
  auto proj = g.add_init_float(prefix + ".dfl.proj", proj_v,
                               {1, 1, (int64_t)reg_max, 1});
  auto wmul = g.node("Mul", {dfl_soft, proj}, {}, prefix + ".dfl.wmul");
  // Sum along reg_max (axis=2) → [N, 4, A]
  auto axes_red = g.add_init_int64(prefix + ".dfl.red", {2});
  auto dist = g.node("ReduceSum", {wmul, axes_red},
                     {attr_int("keepdims", 0)}, prefix + ".dfl.dist");

  // Build per-anchor stride tensor [1, 1, A] and anchor xy [1, 2, A].
  std::vector<float> stride_per_a(total_A);
  std::vector<float> anc_per_a(2 * total_A);
  int idx = 0;
  for (int i = 0; i < nl; ++i) {
    for (int k = 0; k < spatial_per_level[i]; ++k) {
      stride_per_a[idx] = (float)strides[i];
      anc_per_a[0 * total_A + idx] = anchors_xy[i][2 * k + 0];
      anc_per_a[1 * total_A + idx] = anchors_xy[i][2 * k + 1];
      ++idx;
    }
  }
  auto str_init = g.add_init_float(prefix + ".strides", stride_per_a,
                                   {1, 1, (int64_t)total_A});
  auto anc_init = g.add_init_float(prefix + ".anchors", anc_per_a,
                                   {1, 2, (int64_t)total_A});

  // dist * stride → [N, 4, A]
  auto dist_pix = g.node("Mul", {dist, str_init}, {}, prefix + ".dist.pix");

  // Slice into lt [N, 2, A], rb [N, 2, A]
  auto starts_lt = g.add_init_int64(prefix + ".dec.slt", {0});
  auto ends_lt   = g.add_init_int64(prefix + ".dec.elt", {2});
  auto axes_dec  = g.add_init_int64(prefix + ".dec.ax",  {1});
  auto steps_dec = g.add_init_int64(prefix + ".dec.st",  {1});
  auto lt = g.node("Slice", {dist_pix, starts_lt, ends_lt, axes_dec, steps_dec},
                   {}, prefix + ".dec.lt");

  auto starts_rb = g.add_init_int64(prefix + ".dec.srb", {2});
  auto ends_rb   = g.add_init_int64(prefix + ".dec.erb", {4});
  auto rb = g.node("Slice", {dist_pix, starts_rb, ends_rb, axes_dec, steps_dec},
                   {}, prefix + ".dec.rb");

  // x1y1 = anchor - lt;  x2y2 = anchor + rb
  auto x1y1 = g.node("Sub", {anc_init, lt}, {}, prefix + ".dec.x1y1");
  auto x2y2 = g.node("Add", {anc_init, rb}, {}, prefix + ".dec.x2y2");
  auto xyxy = g.node("Concat", {x1y1, x2y2}, {attr_int("axis", 1)},
                     prefix + ".dec.xyxy");

  // Final output: concat xyxy + cls_sig → [N, 4+nc, A]
  return g.node("Concat", {xyxy, cls_sig}, {attr_int("axis", 1)},
                final_out_name);
}

// ─── v11 module emitters ──────────────────────────────────────────────────

// DWConv: Conv with groups = c (depthwise). The state-dict has both .conv
// and .bn just like ConvImpl; we fuse BN into the conv weight as for ConvImpl.
std::string emit_dwconv_module(GraphBuilder& g, const std::string& in,
                                const std::string& prefix,
                                models::DWConvImpl* m) {
  auto* c = m->conv.get();
  return emit_conv_block(g, in, prefix,
                         c->weight, m->bn->weight, m->bn->bias,
                         m->bn->running_mean, m->bn->running_var,
                         c->options.stride()->at(0),
                         std::get<torch::ExpandingArray<2>>(c->options.padding())->at(0),
                         (int)c->options.groups(),
                         m->act_silu,
                         m->bn->options.eps());
}

// DWConvBlock = DWConv (named "0") + Conv 1×1 (named "1").
std::string emit_dwconv_block(GraphBuilder& g, const std::string& in,
                               const std::string& prefix,
                               models::DWConvBlockImpl* b) {
  auto y = emit_dwconv_module(g, in, prefix + ".0", b->dw.get());
  return emit_conv_module(g, y, prefix + ".1", b->pw.get());
}

// Bottleneck with k=(3,3) inner conv kernel. Same emitter as
// emit_bottleneck — k is read from each conv's options.kernel_size at
// runtime, so it works for both k=(3,3) and k=(1,3).

// C3k: cv1 + cv2 + cv3 + ModuleList of n Bottlenecks.
std::string emit_c3k(GraphBuilder& g, const std::string& in,
                      const std::string& prefix, models::C3kImpl* m) {
  auto a = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto* bot = m->m[i]->as<models::BottleneckImpl>();
    a = emit_bottleneck(g, a, prefix + ".m." + std::to_string(i), bot, 0, 0);
  }
  auto b = emit_conv_module(g, in, prefix + ".cv2", m->cv2.get());
  auto cat = g.node("Concat", {a, b}, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv3", m->cv3.get());
}

// C3k2: like C2f but inner blocks may be Bottleneck (c3k=False) or C3k.
std::string emit_c3k2(GraphBuilder& g, const std::string& in,
                       const std::string& prefix, models::C3k2Impl* m) {
  auto y = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  int total_c = 2 * m->c_inner;
  emit_split2(g, y, prefix + ".split", total_c);
  std::string a = prefix + ".split.a";
  std::string b = prefix + ".split.b";
  std::vector<std::string> outs = {a, b};
  std::string last = b;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto idx = std::to_string(i);
    if (m->c3k) {
      auto* sub = m->m[i]->as<models::C3kImpl>();
      last = emit_c3k(g, last, prefix + ".m." + idx, sub);
    } else {
      auto* sub = m->m[i]->as<models::BottleneckImpl>();
      last = emit_bottleneck(g, last, prefix + ".m." + idx, sub, 0, 0);
    }
    outs.push_back(last);
  }
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

// PSA Attention. Emits qkv conv → reshape → split → matmul/softmax/matmul →
// reshape → +pe → proj. Pure-graph form (TRT-compatible).
std::string emit_psa_attention(GraphBuilder& g, const std::string& in,
                                int dim, int H, int W,
                                const std::string& prefix,
                                models::PSAAttentionImpl* a) {
  // 1) qkv conv (no activation).
  auto qkv = emit_conv_module(g, in, prefix + ".qkv", a->qkv.get());
  // qkv shape: [N, h, H, W] where h = num_heads * (2*key_dim + head_dim).
  int nh = a->num_heads;
  int kd = a->key_dim;
  int hd = a->head_dim;
  int N_tokens = H * W;
  int per_h    = 2 * kd + hd;

  // Reshape to [N, nh, per_h, H*W]
  auto rshape0 = g.add_init_int64(prefix + ".rshape0",
                                  {0, (int64_t)nh, (int64_t)per_h, (int64_t)N_tokens});
  auto y4 = g.node("Reshape", {qkv, rshape0}, {}, prefix + ".y4");

  // Split into q/k/v along axis=2 with sizes [kd, kd, hd]. Use Slice ops
  // because Split's `split` attribute moves to input in opset 13.
  auto axes2  = g.add_init_int64(prefix + ".ax2", {2});
  auto step1  = g.add_init_int64(prefix + ".st1", {1});
  auto sq0 = g.add_init_int64(prefix + ".sq0", {0});
  auto sq1 = g.add_init_int64(prefix + ".sq1", {(int64_t)kd});
  auto q   = g.node("Slice", {y4, sq0, sq1, axes2, step1}, {}, prefix + ".q");
  auto sk0 = g.add_init_int64(prefix + ".sk0", {(int64_t)kd});
  auto sk1 = g.add_init_int64(prefix + ".sk1", {(int64_t)(2 * kd)});
  auto k   = g.node("Slice", {y4, sk0, sk1, axes2, step1}, {}, prefix + ".k");
  auto sv0 = g.add_init_int64(prefix + ".sv0", {(int64_t)(2 * kd)});
  auto sv1 = g.add_init_int64(prefix + ".sv1", {(int64_t)per_h});
  auto v   = g.node("Slice", {y4, sv0, sv1, axes2, step1}, {}, prefix + ".v");

  // Transpose q on the last two dims: [N, nh, kd, T] → [N, nh, T, kd].
  auto qT = g.node("Transpose", {q},
                   {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".qT");
  // attn = qT @ k → [N, nh, T, T]
  auto attn0 = g.node("MatMul", {qT, k}, {}, prefix + ".attn0");
  // Scale.
  std::vector<float> scale_v = {(float)a->scale};
  auto scale_init = g.add_init_float(prefix + ".scale", scale_v, {1});
  auto attn = g.node("Mul", {attn0, scale_init}, {}, prefix + ".attn.s");
  attn = g.node("Softmax", {attn}, {attr_int("axis", -1)}, prefix + ".attn.sm");

  // out = v @ attn^T → [N, nh, hd, T]
  auto attnT = g.node("Transpose", {attn},
                       {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".attn.T");
  auto out0 = g.node("MatMul", {v, attnT}, {}, prefix + ".out0");
  // Reshape to [N, dim, H, W]
  auto rshape1 = g.add_init_int64(prefix + ".rshape1",
                                  {0, (int64_t)dim, (int64_t)H, (int64_t)W});
  auto out_spatial = g.node("Reshape", {out0, rshape1}, {}, prefix + ".out.sp");

  // pe(v_spatial): reshape v to [N, dim, H, W] then conv (depthwise).
  auto v_sp = g.node("Reshape", {v, rshape1}, {}, prefix + ".v.sp");
  auto pe_out = emit_conv_module(g, v_sp, prefix + ".pe", a->pe.get());

  auto added = g.node("Add", {out_spatial, pe_out}, {}, prefix + ".add");
  return emit_conv_module(g, added, prefix + ".proj", a->proj.get());
}

// PSABlock: x ← x + attn(x), x ← x + ffn(x). FFN = Conv (act=true) → Conv (no act).
std::string emit_psa_block(GraphBuilder& g, const std::string& in,
                            int dim, int H, int W,
                            const std::string& prefix,
                            models::PSABlockImpl* b) {
  auto* attn = b->attn.get();
  auto attn_out = emit_psa_attention(g, in, dim, H, W, prefix + ".attn", attn);
  std::string after_attn = b->add
      ? g.node("Add", {in, attn_out}, {}, prefix + ".attn.skip")
      : attn_out;

  // FFN = Sequential(Conv(c, 2c, 1), Conv(2c, c, 1, act=false))
  auto* ffn0 = b->ffn->ptr(0)->as<models::ConvImpl>();
  auto* ffn1 = b->ffn->ptr(1)->as<models::ConvImpl>();
  auto y0 = emit_conv_module(g, after_attn, prefix + ".ffn.0", ffn0);
  auto y1 = emit_conv_module(g, y0,         prefix + ".ffn.1", ffn1);
  return b->add ? g.node("Add", {after_attn, y1}, {}, prefix + ".ffn.skip")
                : y1;
}

// C2PSA: cv1 → split → second half through n PSABlocks → concat → cv2.
std::string emit_c2psa(GraphBuilder& g, const std::string& in,
                        int H, int W,
                        const std::string& prefix, models::C2PSAImpl* m) {
  auto y = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  int total_c = 2 * m->c;
  emit_split2(g, y, prefix + ".split", total_c);
  std::string a = prefix + ".split.a";
  std::string b = prefix + ".split.b";
  std::string cur = b;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto* sub = m->m->ptr(i)->as<models::PSABlockImpl>();
    cur = emit_psa_block(g, cur, m->c, H, W,
                         prefix + ".m." + std::to_string(i), sub);
  }
  auto cat = g.node("Concat", {a, cur}, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

// C2PSAf: like C3k2 (cv1 chunk → n inner blocks → cv2 concat) but each
// inner block is `Sequential(Bottleneck, PSABlock)` (children "0" / "1").
// Used at v26 layer 22 only.
std::string emit_c2psaf(GraphBuilder& g, const std::string& in,
                         int H, int W,
                         const std::string& prefix, models::C2PSAfImpl* m) {
  auto y = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  int total_c = 2 * m->c_inner;
  emit_split2(g, y, prefix + ".split", total_c);
  std::string a = prefix + ".split.a";
  std::string b = prefix + ".split.b";
  std::vector<std::string> outs = {a, b};
  std::string last = b;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto idx = std::to_string(i);
    auto* seq = m->m->ptr(i)->as<torch::nn::SequentialImpl>();
    auto* bot = seq->ptr(0)->as<models::BottleneckImpl>();
    auto* psa = seq->ptr(1)->as<models::PSABlockImpl>();
    auto y_b = emit_bottleneck(g, last, prefix + ".m." + idx + ".0", bot, 0, 0);
    last     = emit_psa_block(g, y_b, m->c_inner, H, W,
                               prefix + ".m." + idx + ".1", psa);
    outs.push_back(last);
  }
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

// v11 Detect head: same as v8's emit_detect but cv3 uses two DWConvBlocks
// (each = DWConv→Conv 1×1) plus a final 1×1 Conv2d.
std::string emit_detect_v11(GraphBuilder& g,
                             const std::vector<std::string>& detect_ins,
                             const std::vector<int>&           detect_in_ch,
                             const std::vector<double>&        strides,
                             models::DetectImpl* d, int imgsz,
                             const std::string& prefix,
                             const std::string& final_out_name) {
  int nl      = d->nl;
  int nc      = d->nc;
  int reg_max = d->reg_max;
  int no      = nc + 4 * reg_max;

  std::vector<std::string> level_outs;
  std::vector<int>         spatial_per_level;
  std::vector<std::vector<float>> anchors_xy;

  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    int hw     = feat_h * feat_w;
    spatial_per_level.push_back(hw);
    std::vector<float> anc;
    anc.reserve(hw * 2);
    for (int y = 0; y < feat_h; ++y)
      for (int x = 0; x < feat_w; ++x) {
        anc.push_back(((float)x + 0.5f) * (float)strides[i]);
        anc.push_back(((float)y + 0.5f) * (float)strides[i]);
      }
    anchors_xy.push_back(std::move(anc));

    // cv2 (regression) — same as v8.
    auto* reg = d->cv2[i]->as<torch::nn::SequentialImpl>();
    auto* r0 = reg->ptr(0)->as<models::ConvImpl>();
    auto* r1 = reg->ptr(1)->as<models::ConvImpl>();
    auto  r2 = reg->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_r = emit_conv_module(g, detect_ins[i],
                                prefix + ".cv2." + std::to_string(i) + ".0", r0);
    y_r = emit_conv_module(g, y_r,
                           prefix + ".cv2." + std::to_string(i) + ".1", r1);
    auto rw = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.weight",
                         r2->weight);
    auto rb = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.bias",
                         r2->bias);
    y_r = g.node("Conv", {y_r, rw, rb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv2." + std::to_string(i) + ".2.out");

    // cv3 (cls) — v11: DWConvBlock × 2 + Conv2d.
    auto* cls = d->cv3[i]->as<torch::nn::SequentialImpl>();
    auto* db0 = cls->ptr(0)->as<models::DWConvBlockImpl>();
    auto* db1 = cls->ptr(1)->as<models::DWConvBlockImpl>();
    auto  c2  = cls->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_c = emit_dwconv_block(g, detect_ins[i],
                                  prefix + ".cv3." + std::to_string(i) + ".0", db0);
    y_c = emit_dwconv_block(g, y_c,
                             prefix + ".cv3." + std::to_string(i) + ".1", db1);
    auto cw = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.weight",
                         c2->weight);
    auto cb = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.bias",
                         c2->bias);
    y_c = g.node("Conv", {y_c, cw, cb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv3." + std::to_string(i) + ".2.out");

    auto cat = g.node("Concat", {y_r, y_c}, {attr_int("axis", 1)},
                      prefix + ".level." + std::to_string(i) + ".cat");
    auto rshape = g.add_init_int64(
        prefix + ".level." + std::to_string(i) + ".rshape",
        {0, no, hw});
    auto flat = g.node("Reshape", {cat, rshape}, {},
                       prefix + ".level." + std::to_string(i) + ".flat");
    level_outs.push_back(flat);
  }

  // From here on identical to emit_detect: concat levels, slice, DFL, decode.
  auto pred = g.node("Concat", level_outs, {attr_int("axis", 2)},
                     prefix + ".pred");
  auto axes_ch  = g.add_init_int64(prefix + ".axch",  {1});
  auto starts_b = g.add_init_int64(prefix + ".sb",    {0});
  auto ends_b   = g.add_init_int64(prefix + ".eb",    {(int64_t)(4 * reg_max)});
  auto steps_1  = g.add_init_int64(prefix + ".st1",   {1});
  auto box      = g.node("Slice", {pred, starts_b, ends_b, axes_ch, steps_1}, {},
                          prefix + ".box");
  auto starts_c = g.add_init_int64(prefix + ".sc",    {(int64_t)(4 * reg_max)});
  auto ends_c   = g.add_init_int64(prefix + ".ec",    {(int64_t)no});
  auto cls      = g.node("Slice", {pred, starts_c, ends_c, axes_ch, steps_1}, {},
                          prefix + ".cls");
  auto cls_sig = g.node("Sigmoid", {cls}, {}, prefix + ".cls.sig");
  int total_A = 0;
  for (int hw : spatial_per_level) total_A += hw;
  auto dfl_rshape = g.add_init_int64(prefix + ".dfl.rshape",
                                     {0, 4, (int64_t)reg_max, (int64_t)total_A});
  auto dfl_box    = g.node("Reshape", {box, dfl_rshape}, {},
                            prefix + ".dfl.box");
  auto dfl_soft   = g.node("Softmax", {dfl_box}, {attr_int("axis", 2)},
                            prefix + ".dfl.soft");
  std::vector<float> proj_v(reg_max);
  for (int i = 0; i < reg_max; ++i) proj_v[i] = (float)i;
  auto proj = g.add_init_float(prefix + ".dfl.proj", proj_v,
                               {1, 1, (int64_t)reg_max, 1});
  auto wmul = g.node("Mul", {dfl_soft, proj}, {}, prefix + ".dfl.wmul");
  auto axes_red = g.add_init_int64(prefix + ".dfl.red", {2});
  auto dist = g.node("ReduceSum", {wmul, axes_red},
                     {attr_int("keepdims", 0)}, prefix + ".dfl.dist");
  std::vector<float> stride_per_a(total_A);
  std::vector<float> anc_per_a(2 * total_A);
  int idx = 0;
  for (int i = 0; i < nl; ++i) {
    for (int k = 0; k < spatial_per_level[i]; ++k) {
      stride_per_a[idx] = (float)strides[i];
      anc_per_a[0 * total_A + idx] = anchors_xy[i][2 * k + 0];
      anc_per_a[1 * total_A + idx] = anchors_xy[i][2 * k + 1];
      ++idx;
    }
  }
  auto str_init = g.add_init_float(prefix + ".strides", stride_per_a,
                                   {1, 1, (int64_t)total_A});
  auto anc_init = g.add_init_float(prefix + ".anchors", anc_per_a,
                                   {1, 2, (int64_t)total_A});
  auto dist_pix = g.node("Mul", {dist, str_init}, {}, prefix + ".dist.pix");
  auto starts_lt = g.add_init_int64(prefix + ".dec.slt", {0});
  auto ends_lt   = g.add_init_int64(prefix + ".dec.elt", {2});
  auto axes_dec  = g.add_init_int64(prefix + ".dec.ax",  {1});
  auto steps_dec = g.add_init_int64(prefix + ".dec.st",  {1});
  auto lt = g.node("Slice", {dist_pix, starts_lt, ends_lt, axes_dec, steps_dec},
                   {}, prefix + ".dec.lt");
  auto starts_rb = g.add_init_int64(prefix + ".dec.srb", {2});
  auto ends_rb   = g.add_init_int64(prefix + ".dec.erb", {4});
  auto rb = g.node("Slice", {dist_pix, starts_rb, ends_rb, axes_dec, steps_dec},
                   {}, prefix + ".dec.rb");
  auto x1y1 = g.node("Sub", {anc_init, lt}, {}, prefix + ".dec.x1y1");
  auto x2y2 = g.node("Add", {anc_init, rb}, {}, prefix + ".dec.x2y2");
  auto xyxy = g.node("Concat", {x1y1, x2y2}, {attr_int("axis", 1)},
                     prefix + ".dec.xyxy");
  return g.node("Concat", {xyxy, cls_sig}, {attr_int("axis", 1)},
                final_out_name);
}

// v26 Detect head: DFL-free. cv2 emits 4 channels directly (l, t, r, b
// distances in feature units, passed through Softplus to enforce positivity).
// cv3 uses DWConvBlock × 2 + Conv2d 1×1 (same as v11 cls branch). No DFL
// projection — boxes decoded directly from the 4 distance channels.
std::string emit_detect_v26(GraphBuilder& g,
                             const std::vector<std::string>& detect_ins,
                             const std::vector<int>&           /*detect_in_ch*/,
                             const std::vector<double>&        strides,
                             models::Detect26Impl* d, int imgsz,
                             const std::string& prefix,
                             const std::string& final_out_name) {
  int nl = d->nl;
  int nc = d->nc;
  int no = 4 + nc;

  std::vector<std::string> level_outs;
  std::vector<int>         spatial_per_level;
  std::vector<std::vector<float>> anchors_xy;

  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    int hw     = feat_h * feat_w;
    spatial_per_level.push_back(hw);
    std::vector<float> anc;
    anc.reserve(hw * 2);
    for (int y = 0; y < feat_h; ++y)
      for (int x = 0; x < feat_w; ++x) {
        anc.push_back(((float)x + 0.5f) * (float)strides[i]);
        anc.push_back(((float)y + 0.5f) * (float)strides[i]);
      }
    anchors_xy.push_back(std::move(anc));

    // cv2 (regression — regular Conv 3×3 → Conv 3×3 → 1×1 Conv2d → 4 channels).
    auto* reg = d->cv2[i]->as<torch::nn::SequentialImpl>();
    auto* r0  = reg->ptr(0)->as<models::ConvImpl>();
    auto* r1  = reg->ptr(1)->as<models::ConvImpl>();
    auto  r2  = reg->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_r = emit_conv_module(g, detect_ins[i],
                                 prefix + ".cv2." + std::to_string(i) + ".0", r0);
    y_r = emit_conv_module(g, y_r,
                            prefix + ".cv2." + std::to_string(i) + ".1", r1);
    auto rw = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.weight",
                         r2->weight);
    auto rbi = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.bias",
                           r2->bias);
    y_r = g.node("Conv", {y_r, rw, rbi},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv2." + std::to_string(i) + ".2.out");

    // cv3 (cls — DWConvBlock × 2 + 1×1 Conv2d → nc channels).
    auto* cls = d->cv3[i]->as<torch::nn::SequentialImpl>();
    auto* db0 = cls->ptr(0)->as<models::DWConvBlockImpl>();
    auto* db1 = cls->ptr(1)->as<models::DWConvBlockImpl>();
    auto  c2  = cls->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_c = emit_dwconv_block(g, detect_ins[i],
                                  prefix + ".cv3." + std::to_string(i) + ".0", db0);
    y_c = emit_dwconv_block(g, y_c,
                             prefix + ".cv3." + std::to_string(i) + ".1", db1);
    auto cw = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.weight",
                         c2->weight);
    auto cb = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.bias",
                         c2->bias);
    y_c = g.node("Conv", {y_c, cw, cb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv3." + std::to_string(i) + ".2.out");

    auto cat = g.node("Concat", {y_r, y_c}, {attr_int("axis", 1)},
                      prefix + ".level." + std::to_string(i) + ".cat");
    auto rshape = g.add_init_int64(
        prefix + ".level." + std::to_string(i) + ".rshape",
        {0, no, hw});
    auto flat = g.node("Reshape", {cat, rshape}, {},
                       prefix + ".level." + std::to_string(i) + ".flat");
    level_outs.push_back(flat);
  }

  // Concat all levels: [N, 4+nc, A]
  auto pred = g.node("Concat", level_outs, {attr_int("axis", 2)},
                     prefix + ".pred");
  auto axes_ch  = g.add_init_int64(prefix + ".axch",  {1});
  auto starts_b = g.add_init_int64(prefix + ".sb",    {0});
  auto ends_b   = g.add_init_int64(prefix + ".eb",    {4});
  auto steps_1  = g.add_init_int64(prefix + ".st1",   {1});
  auto box      = g.node("Slice", {pred, starts_b, ends_b, axes_ch, steps_1}, {},
                          prefix + ".box");
  auto starts_c = g.add_init_int64(prefix + ".sc",    {4});
  auto ends_c   = g.add_init_int64(prefix + ".ec",    {(int64_t)no});
  auto cls_t    = g.node("Slice", {pred, starts_c, ends_c, axes_ch, steps_1}, {},
                          prefix + ".cls");
  auto cls_sig  = g.node("Sigmoid", {cls_t}, {}, prefix + ".cls.sig");

  // box → softplus → distances in feature units → multiply by stride.
  auto box_pos = g.node("Softplus", {box}, {}, prefix + ".box.softplus");

  int total_A = 0;
  for (int hw : spatial_per_level) total_A += hw;
  std::vector<float> stride_per_a(total_A);
  std::vector<float> anc_per_a(2 * total_A);
  int idx = 0;
  for (int i = 0; i < nl; ++i) {
    for (int k = 0; k < spatial_per_level[i]; ++k) {
      stride_per_a[idx] = (float)strides[i];
      anc_per_a[0 * total_A + idx] = anchors_xy[i][2 * k + 0];
      anc_per_a[1 * total_A + idx] = anchors_xy[i][2 * k + 1];
      ++idx;
    }
  }
  auto str_init = g.add_init_float(prefix + ".strides", stride_per_a,
                                   {1, 1, (int64_t)total_A});
  auto anc_init = g.add_init_float(prefix + ".anchors", anc_per_a,
                                   {1, 2, (int64_t)total_A});
  auto dist_pix = g.node("Mul", {box_pos, str_init}, {}, prefix + ".dist.pix");
  auto starts_lt = g.add_init_int64(prefix + ".dec.slt", {0});
  auto ends_lt   = g.add_init_int64(prefix + ".dec.elt", {2});
  auto axes_dec  = g.add_init_int64(prefix + ".dec.ax",  {1});
  auto steps_dec = g.add_init_int64(prefix + ".dec.st",  {1});
  auto lt = g.node("Slice", {dist_pix, starts_lt, ends_lt, axes_dec, steps_dec},
                   {}, prefix + ".dec.lt");
  auto starts_rb = g.add_init_int64(prefix + ".dec.srb", {2});
  auto ends_rb   = g.add_init_int64(prefix + ".dec.erb", {4});
  auto rb_slice = g.node("Slice", {dist_pix, starts_rb, ends_rb, axes_dec, steps_dec},
                          {}, prefix + ".dec.rb");
  auto x1y1 = g.node("Sub", {anc_init, lt}, {}, prefix + ".dec.x1y1");
  auto x2y2 = g.node("Add", {anc_init, rb_slice}, {}, prefix + ".dec.x2y2");
  auto xyxy = g.node("Concat", {x1y1, x2y2}, {attr_int("axis", 1)},
                     prefix + ".dec.xyxy");
  return g.node("Concat", {xyxy, cls_sig}, {attr_int("axis", 1)},
                final_out_name);
}

// ─── Task-head helpers ────────────────────────────────────────────────────
//
// All four task heads (Segment / Pose / OBB / Classify) emit one or more
// outputs in addition to the standard detect [N, 4+nc, A]. The helpers
// below are shared across versions; the per-version differences live in
// (a) the backbone+neck walker, and (b) the inner Detect call (DFL vs
// DFL-free). The cv4 chain itself is the same shape — Conv 3×3 → Conv 3×3
// → Conv2d 1×1 — across v8/v11/v26 task heads.

// Emit cv4 chain per level and concat → [N, ch_out, sum(h*w)].
// `prefix` is the head's base path (e.g. "model.22" for v8 task heads).
std::string emit_task_cv4_concat(GraphBuilder& g,
                                  const std::vector<std::string>& level_ins,
                                  const std::vector<int>&         spatial_per_level,
                                  torch::nn::ModuleList&          cv4,
                                  int ch_out,
                                  const std::string& prefix) {
  std::vector<std::string> level_outs;
  for (size_t i = 0; i < level_ins.size(); ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto* c0 = seq->ptr(0)->as<models::ConvImpl>();
    auto* c1 = seq->ptr(1)->as<models::ConvImpl>();
    auto  c2 = seq->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y = emit_conv_module(g, level_ins[i],
                               prefix + ".cv4." + std::to_string(i) + ".0", c0);
    y = emit_conv_module(g, y,
                         prefix + ".cv4." + std::to_string(i) + ".1", c1);
    auto cw = g.add_init(prefix + ".cv4." + std::to_string(i) + ".2.weight",
                         c2->weight);
    auto cb = g.add_init(prefix + ".cv4." + std::to_string(i) + ".2.bias",
                         c2->bias);
    y = g.node("Conv", {y, cw, cb},
               {attr_ints("dilations", {1, 1}),
                attr_int("group", 1),
                attr_ints("kernel_shape", {1, 1}),
                attr_ints("pads", {0, 0, 0, 0}),
                attr_ints("strides", {1, 1})},
               prefix + ".cv4." + std::to_string(i) + ".2.out");
    auto rshape = g.add_init_int64(
        prefix + ".cv4." + std::to_string(i) + ".rshape",
        {0, (int64_t)ch_out, (int64_t)spatial_per_level[i]});
    auto flat = g.node("Reshape", {y, rshape}, {},
                       prefix + ".cv4." + std::to_string(i) + ".flat");
    level_outs.push_back(flat);
  }
  return g.node("Concat", level_outs, {attr_int("axis", 2)},
                prefix + ".cv4.concat");
}

// Emit the Proto module: cv1 (3×3 Conv+BN+SiLU) → ConvTranspose 2×2 →
// cv2 (3×3 Conv+BN+SiLU) → cv3 (1×1 Conv+BN+SiLU). Output spatial size
// is 2× input.
std::string emit_proto(GraphBuilder& g, const std::string& in,
                        const std::string& prefix,
                        models::ProtoImpl* p) {
  auto y = emit_conv_module(g, in, prefix + ".cv1", p->cv1.get());
  // ConvTranspose: stride=2, kernel=2, pad=0; bias is present.
  auto* up = p->upsample.get();
  auto wname = g.add_init(prefix + ".upsample.weight", up->weight);
  auto bname = g.add_init(prefix + ".upsample.bias",   up->bias);
  y = g.node("ConvTranspose", {y, wname, bname},
             {attr_ints("dilations",   {1, 1}),
              attr_int ("group",       1),
              attr_ints("kernel_shape",{2, 2}),
              attr_ints("pads",        {0, 0, 0, 0}),
              attr_ints("strides",     {2, 2})},
             prefix + ".upsample.out");
  y = emit_conv_module(g, y, prefix + ".cv2", p->cv2.get());
  y = emit_conv_module(g, y, prefix + ".cv3", p->cv3.get());
  return y;
}

// Build flat anchors_xy [2*A] and strides [A] arrays from per-level (h, w)
// and per-level stride. Layout mirrors emit_detect (channel-major: row 0 =
// all x's, row 1 = all y's).
struct AnchorTables {
  std::vector<float> anc_xy;        // 2*A — first A x's then A y's
  std::vector<float> strides_per_a; // A
  int total_A = 0;
};
AnchorTables build_anchor_tables(const std::vector<int>& spatial_per_level,
                                  const std::vector<double>& strides,
                                  const std::vector<std::pair<int,int>>& hw_per_level) {
  AnchorTables t;
  for (int hw : spatial_per_level) t.total_A += hw;
  t.anc_xy.assign(2 * t.total_A, 0.f);
  t.strides_per_a.assign(t.total_A, 0.f);
  int idx = 0;
  for (size_t i = 0; i < spatial_per_level.size(); ++i) {
    int H = hw_per_level[i].first;
    int W = hw_per_level[i].second;
    double st = strides[i];
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        t.anc_xy[0 * t.total_A + idx] = ((float)x + 0.5f) * (float)st;
        t.anc_xy[1 * t.total_A + idx] = ((float)y + 0.5f) * (float)st;
        t.strides_per_a[idx] = (float)st;
        ++idx;
      }
    }
  }
  return t;
}

// Emit pose keypoint decoder. raw is [N, nk, A] where nk = K*kpt_dim.
// Per-anchor: kpts[:K, :2] = (raw_xy * 2 - 1) * stride + anchor_pix
//             kpts[:K,  2] = sigmoid(raw_conf)        (only if kpt_dim==3)
// Output is [N, nk, A].
std::string emit_kpt_decode(GraphBuilder& g, const std::string& raw,
                             int num_kpts, int kpt_dim, int total_A,
                             const std::vector<float>& anc_xy,
                             const std::vector<float>& str_per_a,
                             const std::string& prefix,
                             const std::string& out_name) {
  // Reshape [N, nk, A] → [N, K, kpt_dim, A].
  auto rshape4 = g.add_init_int64(prefix + ".kpt.rs4",
      {0, (int64_t)num_kpts, (int64_t)kpt_dim, (int64_t)total_A});
  auto k4 = g.node("Reshape", {raw, rshape4}, {}, prefix + ".kpt.r4");

  // Slice xy [: ,:, :2, :] and conf [..., 2:3, :].
  auto axes2 = g.add_init_int64(prefix + ".kpt.ax2", {2});
  auto step1 = g.add_init_int64(prefix + ".kpt.st1", {1});
  auto sxy0  = g.add_init_int64(prefix + ".kpt.sxy0", {0});
  auto sxy1  = g.add_init_int64(prefix + ".kpt.sxy1", {2});
  auto xy = g.node("Slice", {k4, sxy0, sxy1, axes2, step1}, {},
                   prefix + ".kpt.xy");

  // anchor_pix tensor: [1, 1, 2, A], strides [1, 1, 1, A]
  auto anc_init = g.add_init_float(prefix + ".kpt.anchors",
      anc_xy, {1, 1, 2, (int64_t)total_A});
  auto str_init = g.add_init_float(prefix + ".kpt.strides",
      str_per_a, {1, 1, 1, (int64_t)total_A});

  // Ultralytics decode in pixel coords: (xy * 2 * stride) + (anchor_pix − 0.5*stride)
  // (anchor_pix = (cell_idx + 0.5) * stride, so this evaluates to
  //  (xy*2 + cell_idx) * stride — matches kpts_decode().
  std::vector<float> two = {2.f};
  std::vector<float> half = {0.5f};
  auto two_init  = g.add_init_float(prefix + ".kpt.two",  two,  {1});
  auto half_init = g.add_init_float(prefix + ".kpt.half", half, {1});
  auto xy_2  = g.node("Mul", {xy, two_init}, {}, prefix + ".kpt.xy2");
  auto xy_s  = g.node("Mul", {xy_2, str_init}, {}, prefix + ".kpt.xys");
  auto half_s = g.node("Mul", {str_init, half_init}, {}, prefix + ".kpt.halfs");
  auto anc_off = g.node("Sub", {anc_init, half_s}, {}, prefix + ".kpt.ancoff");
  auto xy_dec = g.node("Add", {xy_s, anc_off}, {}, prefix + ".kpt.xydec");

  std::string final_node;
  if (kpt_dim >= 3) {
    auto sc0 = g.add_init_int64(prefix + ".kpt.sc0", {2});
    auto sc1 = g.add_init_int64(prefix + ".kpt.sc1", {3});
    auto conf = g.node("Slice", {k4, sc0, sc1, axes2, step1}, {},
                        prefix + ".kpt.conf");
    auto cs   = g.node("Sigmoid", {conf}, {}, prefix + ".kpt.confs");
    if (kpt_dim == 3) {
      final_node = g.node("Concat", {xy_dec, cs}, {attr_int("axis", 2)},
                          prefix + ".kpt.dec3");
    } else {
      // kpt_dim > 3 — unusual; pass-through any extra channels untouched.
      auto se0 = g.add_init_int64(prefix + ".kpt.se0", {3});
      auto se1 = g.add_init_int64(prefix + ".kpt.se1", {(int64_t)kpt_dim});
      auto rest = g.node("Slice", {k4, se0, se1, axes2, step1}, {},
                          prefix + ".kpt.rest");
      final_node = g.node("Concat", {xy_dec, cs, rest},
                           {attr_int("axis", 2)}, prefix + ".kpt.dec_full");
    }
  } else {
    final_node = xy_dec;
  }

  // Reshape back to [N, nk, A].
  auto rshape3 = g.add_init_int64(prefix + ".kpt.rs3",
      {0, (int64_t)(num_kpts * kpt_dim), (int64_t)total_A});
  return g.node("Reshape", {final_node, rshape3}, {}, out_name);
}

// Emit OBB angle decoder: raw [N, ne, A] → (sigmoid - 0.25) * π → squeeze
// to [N, A]. (ne=1 in practice.)
std::string emit_obb_angle_decode(GraphBuilder& g, const std::string& raw,
                                   int /*ne*/, int total_A,
                                   const std::string& prefix,
                                   const std::string& out_name) {
  auto sg = g.node("Sigmoid", {raw}, {}, prefix + ".obb.sig");
  std::vector<float> shift = {0.25f};
  std::vector<float> scale = {(float)M_PI};
  auto shift_init = g.add_init_float(prefix + ".obb.shift", shift, {1});
  auto scale_init = g.add_init_float(prefix + ".obb.scale", scale, {1});
  auto sub = g.node("Sub", {sg, shift_init}, {}, prefix + ".obb.sub");
  auto mul = g.node("Mul", {sub, scale_init}, {}, prefix + ".obb.mul");
  // Squeeze ne dim (axis=1) → [N, A].
  auto sq_axes = g.add_init_int64(prefix + ".obb.sq_ax", {1});
  return g.node("Squeeze", {mul, sq_axes}, {}, out_name);
}

// ─── OBB-aware detect emitters: rotated dist2rbox decode ─────────────────
//
// These mirror emit_detect / emit_detect_v11 / emit_detect_v26 but expect
// an external `angle_in` tensor [N, A] (pre-decoded to [-π/4, 3π/4]) and
// apply Ultralytics' rotated decode:
//   xf = (r - l)/2,  yf = (b - t)/2
//   cx_feat = xf*cos − yf*sin + anchor_x_feat
//   cy_feat = xf*sin + yf*cos + anchor_y_feat
//   w_feat  = l + r,   h_feat = t + b
//   xyxy_pix = (cx*stride − w*stride/2, cy*stride − h*stride/2,
//               cx*stride + w*stride/2, cy*stride + h*stride/2)
// `angle_in` is broadcast as [N, A]; broadcast against [N, *, A] tensors
// happens via explicit Unsqueeze where needed.

// Helper: given DFL distances [N, 4, A] (feature units), an angle [N, A],
// per-anchor feature-unit anchors [1, A] for x and y, and per-anchor
// strides [1, A], emit rotated decode → xyxy [N, 4, A] in pixels.
std::string emit_rbox_decode(GraphBuilder& g,
                              const std::string& dist,
                              const std::string& angle_in,
                              const std::string& anc_x_feat,  // [1, A]
                              const std::string& anc_y_feat,  // [1, A]
                              const std::string& strd_per_a,  // [1, A]
                              int /*total_A*/,
                              const std::string& prefix) {
  // Slice dist → l, t, r, b each [N, 1, A].
  auto axes_ch = g.add_init_int64(prefix + ".rb.axch", {1});
  auto step1   = g.add_init_int64(prefix + ".rb.st1", {1});
  auto sl0 = g.add_init_int64(prefix + ".rb.sl0", {0});
  auto sl1 = g.add_init_int64(prefix + ".rb.sl1", {1});
  auto st0 = g.add_init_int64(prefix + ".rb.st0", {1});
  auto st1n = g.add_init_int64(prefix + ".rb.st1n", {2});
  auto sr0 = g.add_init_int64(prefix + ".rb.sr0", {2});
  auto sr1 = g.add_init_int64(prefix + ".rb.sr1", {3});
  auto sb0 = g.add_init_int64(prefix + ".rb.sb0", {3});
  auto sb1 = g.add_init_int64(prefix + ".rb.sb1", {4});
  auto l = g.node("Slice", {dist, sl0, sl1, axes_ch, step1}, {}, prefix + ".rb.l");
  auto t = g.node("Slice", {dist, st0, st1n, axes_ch, step1}, {}, prefix + ".rb.t");
  auto r = g.node("Slice", {dist, sr0, sr1, axes_ch, step1}, {}, prefix + ".rb.r");
  auto b = g.node("Slice", {dist, sb0, sb1, axes_ch, step1}, {}, prefix + ".rb.b");
  // l/t/r/b are each [N, 1, A]; squeeze to [N, A] for arithmetic with angle.
  auto sq_axes = g.add_init_int64(prefix + ".rb.sq_ax", {1});
  l = g.node("Squeeze", {l, sq_axes}, {}, prefix + ".rb.l2");
  t = g.node("Squeeze", {t, sq_axes}, {}, prefix + ".rb.t2");
  r = g.node("Squeeze", {r, sq_axes}, {}, prefix + ".rb.r2");
  b = g.node("Squeeze", {b, sq_axes}, {}, prefix + ".rb.b2");

  // xf = (r - l) / 2, yf = (b - t) / 2
  std::vector<float> half = {0.5f};
  auto half_init = g.add_init_float(prefix + ".rb.half", half, {1});
  auto rml = g.node("Sub", {r, l}, {}, prefix + ".rb.rml");
  auto bmt = g.node("Sub", {b, t}, {}, prefix + ".rb.bmt");
  auto xf  = g.node("Mul", {rml, half_init}, {}, prefix + ".rb.xf");
  auto yf  = g.node("Mul", {bmt, half_init}, {}, prefix + ".rb.yf");

  auto cos_a = g.node("Cos", {angle_in}, {}, prefix + ".rb.cos");
  auto sin_a = g.node("Sin", {angle_in}, {}, prefix + ".rb.sin");
  auto xc_p = g.node("Mul", {xf, cos_a}, {}, prefix + ".rb.xc");
  auto ys_p = g.node("Mul", {yf, sin_a}, {}, prefix + ".rb.ys");
  auto xs_p = g.node("Mul", {xf, sin_a}, {}, prefix + ".rb.xs");
  auto yc_p = g.node("Mul", {yf, cos_a}, {}, prefix + ".rb.yc");
  auto cx0  = g.node("Sub", {xc_p, ys_p}, {}, prefix + ".rb.cx0");
  auto cy0  = g.node("Add", {xs_p, yc_p}, {}, prefix + ".rb.cy0");
  auto cx_feat = g.node("Add", {cx0, anc_x_feat}, {}, prefix + ".rb.cxf");
  auto cy_feat = g.node("Add", {cy0, anc_y_feat}, {}, prefix + ".rb.cyf");
  auto w_feat = g.node("Add", {l, r}, {}, prefix + ".rb.wf");
  auto h_feat = g.node("Add", {t, b}, {}, prefix + ".rb.hf");
  auto cx_pix = g.node("Mul", {cx_feat, strd_per_a}, {}, prefix + ".rb.cxp");
  auto cy_pix = g.node("Mul", {cy_feat, strd_per_a}, {}, prefix + ".rb.cyp");
  auto w_pix  = g.node("Mul", {w_feat,  strd_per_a}, {}, prefix + ".rb.wp");
  auto h_pix  = g.node("Mul", {h_feat,  strd_per_a}, {}, prefix + ".rb.hp");
  auto wh_off = g.node("Mul", {w_pix, half_init}, {}, prefix + ".rb.wo");
  auto hh_off = g.node("Mul", {h_pix, half_init}, {}, prefix + ".rb.ho");
  auto x1 = g.node("Sub", {cx_pix, wh_off}, {}, prefix + ".rb.x1");
  auto y1 = g.node("Sub", {cy_pix, hh_off}, {}, prefix + ".rb.y1");
  auto x2 = g.node("Add", {cx_pix, wh_off}, {}, prefix + ".rb.x2");
  auto y2 = g.node("Add", {cy_pix, hh_off}, {}, prefix + ".rb.y2");
  // Stack along a new dim=1 → [N, 4, A]. ONNX: Unsqueeze(axis=1) each then Concat.
  auto u_ax = g.add_init_int64(prefix + ".rb.u_ax", {1});
  x1 = g.node("Unsqueeze", {x1, u_ax}, {}, prefix + ".rb.x1u");
  y1 = g.node("Unsqueeze", {y1, u_ax}, {}, prefix + ".rb.y1u");
  x2 = g.node("Unsqueeze", {x2, u_ax}, {}, prefix + ".rb.x2u");
  y2 = g.node("Unsqueeze", {y2, u_ax}, {}, prefix + ".rb.y2u");
  return g.node("Concat", {x1, y1, x2, y2}, {attr_int("axis", 1)},
                prefix + ".rb.xyxy");
}

// v8/v11 OBB detect emitter (DFL form). Mirrors emit_detect / emit_detect_v11
// but with rotated decode using angle_in. v11_cv3 selects DWConvBlock cv3.
std::string emit_detect_obb_dfl(GraphBuilder& g,
                                 const std::vector<std::string>& detect_ins,
                                 const std::vector<int>&           /*detect_in_ch*/,
                                 const std::vector<double>&        strides,
                                 models::DetectImpl* d, int imgsz,
                                 const std::string& angle_in,   // [N, A]
                                 bool v11_cv3,
                                 const std::string& prefix,
                                 const std::string& final_out_name) {
  int nl      = d->nl;
  int nc      = d->nc;
  int reg_max = d->reg_max;
  int no      = nc + 4 * reg_max;

  std::vector<std::string> level_outs;
  std::vector<int>         spatial_per_level;
  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    int hw     = feat_h * feat_w;
    spatial_per_level.push_back(hw);

    auto* reg = d->cv2[i]->as<torch::nn::SequentialImpl>();
    auto* cls_seq = d->cv3[i]->as<torch::nn::SequentialImpl>();
    // cv2 (regression — Conv→Conv→Conv2d, same in v8 and v11).
    auto* r0 = reg->ptr(0)->as<models::ConvImpl>();
    auto* r1 = reg->ptr(1)->as<models::ConvImpl>();
    auto  r2 = reg->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_r = emit_conv_module(g, detect_ins[i],
                                 prefix + ".cv2." + std::to_string(i) + ".0", r0);
    y_r = emit_conv_module(g, y_r,
                            prefix + ".cv2." + std::to_string(i) + ".1", r1);
    auto rw = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.weight",
                         r2->weight);
    auto rb = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.bias",
                         r2->bias);
    y_r = g.node("Conv", {y_r, rw, rb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv2." + std::to_string(i) + ".2.out");

    // cv3 (cls): Conv chain (v8) or DWConvBlock chain (v11).
    std::string y_c;
    if (v11_cv3) {
      auto* db0 = cls_seq->ptr(0)->as<models::DWConvBlockImpl>();
      auto* db1 = cls_seq->ptr(1)->as<models::DWConvBlockImpl>();
      auto  c2  = cls_seq->ptr(2)->as<torch::nn::Conv2dImpl>();
      y_c = emit_dwconv_block(g, detect_ins[i],
                               prefix + ".cv3." + std::to_string(i) + ".0", db0);
      y_c = emit_dwconv_block(g, y_c,
                               prefix + ".cv3." + std::to_string(i) + ".1", db1);
      auto cw = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.weight",
                           c2->weight);
      auto cb = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.bias",
                           c2->bias);
      y_c = g.node("Conv", {y_c, cw, cb},
                   {attr_ints("dilations", {1, 1}),
                    attr_int("group", 1),
                    attr_ints("kernel_shape", {1, 1}),
                    attr_ints("pads", {0, 0, 0, 0}),
                    attr_ints("strides", {1, 1})},
                   prefix + ".cv3." + std::to_string(i) + ".2.out");
    } else {
      auto* c0 = cls_seq->ptr(0)->as<models::ConvImpl>();
      auto* c1 = cls_seq->ptr(1)->as<models::ConvImpl>();
      auto  c2 = cls_seq->ptr(2)->as<torch::nn::Conv2dImpl>();
      y_c = emit_conv_module(g, detect_ins[i],
                              prefix + ".cv3." + std::to_string(i) + ".0", c0);
      y_c = emit_conv_module(g, y_c,
                              prefix + ".cv3." + std::to_string(i) + ".1", c1);
      auto cw = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.weight",
                           c2->weight);
      auto cb = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.bias",
                           c2->bias);
      y_c = g.node("Conv", {y_c, cw, cb},
                   {attr_ints("dilations", {1, 1}),
                    attr_int("group", 1),
                    attr_ints("kernel_shape", {1, 1}),
                    attr_ints("pads", {0, 0, 0, 0}),
                    attr_ints("strides", {1, 1})},
                   prefix + ".cv3." + std::to_string(i) + ".2.out");
    }
    auto cat = g.node("Concat", {y_r, y_c}, {attr_int("axis", 1)},
                      prefix + ".level." + std::to_string(i) + ".cat");
    auto rshape = g.add_init_int64(
        prefix + ".level." + std::to_string(i) + ".rshape",
        {0, no, hw});
    auto flat = g.node("Reshape", {cat, rshape}, {},
                       prefix + ".level." + std::to_string(i) + ".flat");
    level_outs.push_back(flat);
  }

  auto pred = g.node("Concat", level_outs, {attr_int("axis", 2)},
                     prefix + ".pred");
  auto axes_ch  = g.add_init_int64(prefix + ".axch", {1});
  auto starts_b = g.add_init_int64(prefix + ".sb",   {0});
  auto ends_b   = g.add_init_int64(prefix + ".eb",   {(int64_t)(4 * reg_max)});
  auto steps_1  = g.add_init_int64(prefix + ".st1",  {1});
  auto box      = g.node("Slice", {pred, starts_b, ends_b, axes_ch, steps_1}, {},
                          prefix + ".box");
  auto starts_c = g.add_init_int64(prefix + ".sc",   {(int64_t)(4 * reg_max)});
  auto ends_c   = g.add_init_int64(prefix + ".ec",   {(int64_t)no});
  auto cls      = g.node("Slice", {pred, starts_c, ends_c, axes_ch, steps_1}, {},
                          prefix + ".cls");
  auto cls_sig  = g.node("Sigmoid", {cls}, {}, prefix + ".cls.sig");

  int total_A = 0;
  for (int hw : spatial_per_level) total_A += hw;
  // DFL → distances [N, 4, A] in feature units (NO multiplication by stride).
  auto dfl_rshape = g.add_init_int64(prefix + ".dfl.rshape",
                                     {0, 4, (int64_t)reg_max, (int64_t)total_A});
  auto dfl_box    = g.node("Reshape", {box, dfl_rshape}, {},
                            prefix + ".dfl.box");
  auto dfl_soft   = g.node("Softmax", {dfl_box}, {attr_int("axis", 2)},
                            prefix + ".dfl.soft");
  std::vector<float> proj_v(reg_max);
  for (int i = 0; i < reg_max; ++i) proj_v[i] = (float)i;
  auto proj = g.add_init_float(prefix + ".dfl.proj", proj_v,
                               {1, 1, (int64_t)reg_max, 1});
  auto wmul = g.node("Mul", {dfl_soft, proj}, {}, prefix + ".dfl.wmul");
  auto axes_red = g.add_init_int64(prefix + ".dfl.red", {2});
  auto dist = g.node("ReduceSum", {wmul, axes_red},
                     {attr_int("keepdims", 0)}, prefix + ".dfl.dist");

  // Build per-anchor feature-unit anchors and stride [1, A].
  std::vector<float> anc_x(total_A), anc_y(total_A), strd_a(total_A);
  int idx = 0;
  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    for (int y = 0; y < feat_h; ++y) {
      for (int x = 0; x < feat_w; ++x) {
        anc_x[idx] = (float)x + 0.5f;     // feature units (cell+0.5)
        anc_y[idx] = (float)y + 0.5f;
        strd_a[idx] = (float)strides[i];
        ++idx;
      }
    }
  }
  auto anc_x_init = g.add_init_float(prefix + ".rb.anc_x_feat", anc_x,
                                      {1, (int64_t)total_A});
  auto anc_y_init = g.add_init_float(prefix + ".rb.anc_y_feat", anc_y,
                                      {1, (int64_t)total_A});
  auto strd_init  = g.add_init_float(prefix + ".rb.strides", strd_a,
                                      {1, (int64_t)total_A});

  // Rotated decode → xyxy [N, 4, A].
  auto xyxy = emit_rbox_decode(g, dist, angle_in,
                                anc_x_init, anc_y_init, strd_init,
                                total_A, prefix);
  return g.node("Concat", {xyxy, cls_sig}, {attr_int("axis", 1)},
                final_out_name);
}

// v26 OBB detect emitter (DFL-free): cv2 emits 4 channels directly,
// softplus to get distances, then rotated decode.
std::string emit_detect_obb_v26(GraphBuilder& g,
                                 const std::vector<std::string>& detect_ins,
                                 const std::vector<int>&           /*detect_in_ch*/,
                                 const std::vector<double>&        strides,
                                 models::Detect26Impl* d, int imgsz,
                                 const std::string& angle_in,
                                 const std::string& prefix,
                                 const std::string& final_out_name) {
  int nl = d->nl;
  int nc = d->nc;
  int no = 4 + nc;

  std::vector<std::string> level_outs;
  std::vector<int>         spatial_per_level;
  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    int hw     = feat_h * feat_w;
    spatial_per_level.push_back(hw);

    auto* reg = d->cv2[i]->as<torch::nn::SequentialImpl>();
    auto* r0  = reg->ptr(0)->as<models::ConvImpl>();
    auto* r1  = reg->ptr(1)->as<models::ConvImpl>();
    auto  r2  = reg->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_r = emit_conv_module(g, detect_ins[i],
                                 prefix + ".cv2." + std::to_string(i) + ".0", r0);
    y_r = emit_conv_module(g, y_r,
                            prefix + ".cv2." + std::to_string(i) + ".1", r1);
    auto rw = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.weight",
                         r2->weight);
    auto rb = g.add_init(prefix + ".cv2." + std::to_string(i) + ".2.bias",
                         r2->bias);
    y_r = g.node("Conv", {y_r, rw, rb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv2." + std::to_string(i) + ".2.out");
    auto* cls = d->cv3[i]->as<torch::nn::SequentialImpl>();
    auto* db0 = cls->ptr(0)->as<models::DWConvBlockImpl>();
    auto* db1 = cls->ptr(1)->as<models::DWConvBlockImpl>();
    auto  c2  = cls->ptr(2)->as<torch::nn::Conv2dImpl>();
    auto y_c = emit_dwconv_block(g, detect_ins[i],
                                  prefix + ".cv3." + std::to_string(i) + ".0", db0);
    y_c = emit_dwconv_block(g, y_c,
                             prefix + ".cv3." + std::to_string(i) + ".1", db1);
    auto cw = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.weight",
                         c2->weight);
    auto cb = g.add_init(prefix + ".cv3." + std::to_string(i) + ".2.bias",
                         c2->bias);
    y_c = g.node("Conv", {y_c, cw, cb},
                 {attr_ints("dilations", {1, 1}),
                  attr_int("group", 1),
                  attr_ints("kernel_shape", {1, 1}),
                  attr_ints("pads", {0, 0, 0, 0}),
                  attr_ints("strides", {1, 1})},
                 prefix + ".cv3." + std::to_string(i) + ".2.out");

    auto cat = g.node("Concat", {y_r, y_c}, {attr_int("axis", 1)},
                      prefix + ".level." + std::to_string(i) + ".cat");
    auto rshape = g.add_init_int64(
        prefix + ".level." + std::to_string(i) + ".rshape",
        {0, no, hw});
    auto flat = g.node("Reshape", {cat, rshape}, {},
                       prefix + ".level." + std::to_string(i) + ".flat");
    level_outs.push_back(flat);
  }

  auto pred = g.node("Concat", level_outs, {attr_int("axis", 2)},
                     prefix + ".pred");
  auto axes_ch  = g.add_init_int64(prefix + ".axch", {1});
  auto starts_b = g.add_init_int64(prefix + ".sb",   {0});
  auto ends_b   = g.add_init_int64(prefix + ".eb",   {4});
  auto steps_1  = g.add_init_int64(prefix + ".st1",  {1});
  auto box      = g.node("Slice", {pred, starts_b, ends_b, axes_ch, steps_1}, {},
                          prefix + ".box");
  auto starts_c = g.add_init_int64(prefix + ".sc",   {4});
  auto ends_c   = g.add_init_int64(prefix + ".ec",   {(int64_t)no});
  auto cls      = g.node("Slice", {pred, starts_c, ends_c, axes_ch, steps_1}, {},
                          prefix + ".cls");
  auto cls_sig  = g.node("Sigmoid", {cls}, {}, prefix + ".cls.sig");
  auto dist = g.node("Softplus", {box}, {}, prefix + ".dist");

  int total_A = 0;
  for (int hw : spatial_per_level) total_A += hw;
  std::vector<float> anc_x(total_A), anc_y(total_A), strd_a(total_A);
  int idx = 0;
  for (int i = 0; i < nl; ++i) {
    int feat_h = imgsz / (int)strides[i];
    int feat_w = imgsz / (int)strides[i];
    for (int y = 0; y < feat_h; ++y) {
      for (int x = 0; x < feat_w; ++x) {
        anc_x[idx] = (float)x + 0.5f;
        anc_y[idx] = (float)y + 0.5f;
        strd_a[idx] = (float)strides[i];
        ++idx;
      }
    }
  }
  auto anc_x_init = g.add_init_float(prefix + ".rb.anc_x_feat", anc_x,
                                      {1, (int64_t)total_A});
  auto anc_y_init = g.add_init_float(prefix + ".rb.anc_y_feat", anc_y,
                                      {1, (int64_t)total_A});
  auto strd_init  = g.add_init_float(prefix + ".rb.strides", strd_a,
                                      {1, (int64_t)total_A});
  auto xyxy = emit_rbox_decode(g, dist, angle_in,
                                anc_x_init, anc_y_init, strd_init,
                                total_A, prefix);
  return g.node("Concat", {xyxy, cls_sig}, {attr_int("axis", 1)},
                final_out_name);
}

}  // anonymous namespace

void export_yolo8_onnx(models::Yolo8Detect& model,
                        const std::string&    path,
                        const OnnxExportConfig& cfg) {
  // Force eval (BN running stats stable).
  model->eval();

  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1,
              {-1, 3, cfg.imgsz, cfg.imgsz});

  // We replicate the v8 YAML's connectivity by re-reading the model's
  // ModuleList and the same yaml structure used in Yolo8DetectImpl.
  // Layers 0..21 are forward layers; layer 22 is Detect.
  struct Step {
    std::vector<int> from;
    std::string      kind;
  };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1}, "C2f"}, {{-1}, "SPPF"},
      {{-1}, "Up"},   {{-1, 6}, "Cat"}, {{-1}, "C2f"},
      {{-1}, "Up"},   {{-1, 4}, "Cat"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1, 12}, "Cat"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1, 9}, "Cat"},  {{-1}, "C2f"},
      {{15, 18, 21}, "Detect"},
  };

  std::vector<std::string> outs(yaml.size());
  std::string prev = cfg.input_name;
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto idx_or_prev = [&](int f) {
      return (f == -1) ? prev : outs[f];
    };
    auto prefix = "model." + std::to_string(i);

    if (s.kind == "Conv") {
      auto* m = model->model[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), prefix, m);
    } else if (s.kind == "C2f") {
      auto* m = model->model[i]->as<models::C2fImpl>();
      outs[i] = emit_c2f(g, idx_or_prev(s.from[0]), prefix, m);
    } else if (s.kind == "SPPF") {
      auto* m = model->model[i]->as<models::SPPFImpl>();
      outs[i] = emit_sppf(g, idx_or_prev(s.from[0]), prefix, m);
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), prefix);
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back((f == -1) ? prev : outs[f]);
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, prefix + ".cat");
    } else if (s.kind == "Detect") {
      auto* d = model->model[i]->as<models::DetectImpl>();
      std::vector<std::string> det_in;
      std::vector<int>          det_ch;
      for (size_t k = 0; k < s.from.size(); ++k) {
        int f = s.from[k];
        det_in.push_back(outs[f]);
        det_ch.push_back(d->ch[k]);
      }
      outs[i] = emit_detect(g, det_in, det_ch, model->stride, d,
                            cfg.imgsz, prefix, cfg.output_name);
    }
    prev = outs[i];
  }

  int nc = model->nc;
  g.set_output(cfg.output_name, /*FLOAT=*/1,
               {-1, 4 + nc, /*A*/ -1});

  // Build ModelProto.
  Pb model_pb;
  model_pb.write_int64_field(1, /*ir_version=*/8);  // ONNX 1.13+
  model_pb.write_string_field(2, cfg.producer_name);
  model_pb.write_string_field(3, cfg.producer_version);
  model_pb.write_string_field(4, "");      // domain
  model_pb.write_int64_field(5, 1);        // model_version
  model_pb.write_string_field(6, "");      // doc_string
  model_pb.write_bytes_field(7, g.build_graph_bytes("yolo8"));
  model_pb.write_bytes_field(8, opset_id("", cfg.opset_version));

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(model_pb.bytes().data(),
          (std::streamsize)model_pb.bytes().size());
}

void export_yolo11_onnx(models::Yolo11Detect& model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1,
              {-1, 3, cfg.imgsz, cfg.imgsz});

  // v11 yaml connectivity (24 layers; same FPN topology as v8 but with
  // C3k2/C2PSA at known indices).
  struct Step {
    std::vector<int> from;
    std::string      kind;
  };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C3k2"},   // 0  1  2
      {{-1}, "Conv"}, {{-1}, "C3k2"},                    // 3  4
      {{-1}, "Conv"}, {{-1}, "C3k2"},                    // 5  6
      {{-1}, "Conv"}, {{-1}, "C3k2"}, {{-1}, "SPPF"},    // 7  8  9
      {{-1}, "C2PSA"},                                    // 10
      {{-1}, "Up"}, {{-1, 6},  "Cat"}, {{-1}, "C3k2"},   // 11 12 13
      {{-1}, "Up"}, {{-1, 4},  "Cat"}, {{-1}, "C3k2"},   // 14 15 16
      {{-1}, "Conv"}, {{-1, 13}, "Cat"}, {{-1}, "C3k2"}, // 17 18 19
      {{-1}, "Conv"}, {{-1, 10}, "Cat"}, {{-1}, "C3k2"}, // 20 21 22
      {{16, 19, 22}, "Detect"},                          // 23
  };

  std::vector<std::string> outs(yaml.size());
  std::vector<std::pair<int, int>> spatial(yaml.size(), {0, 0});  // (H, W)
  std::string prev = cfg.input_name;
  // Track input HxW through the graph for C2PSA (needs spatial dims).
  int H_cur = cfg.imgsz, W_cur = cfg.imgsz;
  std::vector<std::pair<int, int>> outs_hw(yaml.size());

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
    auto hw_at = [&](int f) -> std::pair<int, int> {
      if (f == -1) return (i == 0) ? std::pair<int, int>{cfg.imgsz, cfg.imgsz}
                                   : outs_hw[i - 1];
      return outs_hw[f];
    };
    auto prefix = "model." + std::to_string(i);

    if (s.kind == "Conv") {
      auto* m = model->model[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), prefix, m);
      auto in_hw = hw_at(s.from[0]);
      int stride = m->conv->options.stride()->at(0);
      outs_hw[i] = {in_hw.first / stride, in_hw.second / stride};
    } else if (s.kind == "C3k2") {
      auto* m = model->model[i]->as<models::C3k2Impl>();
      outs[i] = emit_c3k2(g, idx_or_prev(s.from[0]), prefix, m);
      outs_hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "SPPF") {
      auto* m = model->model[i]->as<models::SPPFImpl>();
      outs[i] = emit_sppf(g, idx_or_prev(s.from[0]), prefix, m);
      outs_hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "C2PSA") {
      auto* m = model->model[i]->as<models::C2PSAImpl>();
      auto in_hw = hw_at(s.from[0]);
      outs[i] = emit_c2psa(g, idx_or_prev(s.from[0]),
                           in_hw.first, in_hw.second, prefix, m);
      outs_hw[i] = in_hw;
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), prefix);
      auto in_hw = hw_at(s.from[0]);
      outs_hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back((f == -1) ? prev : outs[f]);
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, prefix + ".cat");
      outs_hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "Detect") {
      auto* d = model->model[i]->as<models::DetectImpl>();
      std::vector<std::string> det_in;
      std::vector<int>          det_ch;
      // Iterate by index — `&f - s.from.data()` is UB because `f` is a
      // copy from the range-based for loop, not an element of s.from.
      for (size_t k = 0; k < s.from.size(); ++k) {
        det_in.push_back(outs[s.from[k]]);
        det_ch.push_back(d->ch[k]);
      }
      outs[i] = emit_detect_v11(g, det_in, det_ch, model->stride, d,
                                 cfg.imgsz, prefix, cfg.output_name);
      outs_hw[i] = {0, 0};
    }
    prev = outs[i];
  }

  int nc = model->nc;
  g.set_output(cfg.output_name, /*FLOAT=*/1,
               {-1, 4 + nc, /*A*/ -1});

  Pb model_pb;
  model_pb.write_int64_field(1, /*ir_version=*/8);
  model_pb.write_string_field(2, cfg.producer_name);
  model_pb.write_string_field(3, cfg.producer_version);
  model_pb.write_string_field(4, "");
  model_pb.write_int64_field(5, 1);
  model_pb.write_string_field(6, "");
  model_pb.write_bytes_field(7, g.build_graph_bytes("yolo11"));
  model_pb.write_bytes_field(8, opset_id("", cfg.opset_version));

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(model_pb.bytes().data(),
          (std::streamsize)model_pb.bytes().size());
}

void export_yolo26_onnx(models::Yolo26Detect&  model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1,
              {-1, 3, cfg.imgsz, cfg.imgsz});

  // v26 yaml connectivity (24 layers — same as v11; only the head differs).
  struct Step {
    std::vector<int> from;
    std::string      kind;
  };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"}, {{-1}, "SPPF"},
      {{-1}, "C2PSA"},
      {{-1}, "Up"}, {{-1, 6},  "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Up"}, {{-1, 4},  "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1, 13}, "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1, 10}, "Cat"}, {{-1}, "C2PSAf"},   // 22 — DIFFERS from v11
      {{16, 19, 22}, "Detect"},
  };

  std::vector<std::string> outs(yaml.size());
  std::string prev = cfg.input_name;
  std::vector<std::pair<int, int>> outs_hw(yaml.size());

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
    auto hw_at = [&](int f) -> std::pair<int, int> {
      if (f == -1) return (i == 0) ? std::pair<int, int>{cfg.imgsz, cfg.imgsz}
                                   : outs_hw[i - 1];
      return outs_hw[f];
    };
    auto prefix = "model." + std::to_string(i);

    if (s.kind == "Conv") {
      auto* m = model->model[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), prefix, m);
      auto in_hw = hw_at(s.from[0]);
      int stride = m->conv->options.stride()->at(0);
      outs_hw[i] = {in_hw.first / stride, in_hw.second / stride};
    } else if (s.kind == "C3k2") {
      auto* m = model->model[i]->as<models::C3k2Impl>();
      outs[i] = emit_c3k2(g, idx_or_prev(s.from[0]), prefix, m);
      outs_hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "SPPF") {
      auto* m = model->model[i]->as<models::SPPFImpl>();
      outs[i] = emit_sppf(g, idx_or_prev(s.from[0]), prefix, m);
      outs_hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "C2PSA") {
      auto* m = model->model[i]->as<models::C2PSAImpl>();
      auto in_hw = hw_at(s.from[0]);
      outs[i] = emit_c2psa(g, idx_or_prev(s.from[0]),
                           in_hw.first, in_hw.second, prefix, m);
      outs_hw[i] = in_hw;
    } else if (s.kind == "C2PSAf") {
      auto* m = model->model[i]->as<models::C2PSAfImpl>();
      auto in_hw = hw_at(s.from[0]);
      outs[i] = emit_c2psaf(g, idx_or_prev(s.from[0]),
                             in_hw.first, in_hw.second, prefix, m);
      outs_hw[i] = in_hw;
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), prefix);
      auto in_hw = hw_at(s.from[0]);
      outs_hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back((f == -1) ? prev : outs[f]);
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, prefix + ".cat");
      outs_hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "Detect") {
      auto* d = model->model[i]->as<models::Detect26Impl>();
      std::vector<std::string> det_in;
      std::vector<int>          det_ch;
      for (size_t k = 0; k < s.from.size(); ++k) {
        det_in.push_back(outs[s.from[k]]);
        det_ch.push_back(d->ch[k]);
      }
      outs[i] = emit_detect_v26(g, det_in, det_ch, model->stride, d,
                                 cfg.imgsz, prefix, cfg.output_name);
      outs_hw[i] = {0, 0};
    }
    prev = outs[i];
  }

  int nc = model->nc;
  g.set_output(cfg.output_name, /*FLOAT=*/1,
               {-1, 4 + nc, /*A*/ -1});

  Pb model_pb;
  model_pb.write_int64_field(1, /*ir_version=*/8);
  model_pb.write_string_field(2, cfg.producer_name);
  model_pb.write_string_field(3, cfg.producer_version);
  model_pb.write_string_field(4, "");
  model_pb.write_int64_field(5, 1);
  model_pb.write_string_field(6, "");
  model_pb.write_bytes_field(7, g.build_graph_bytes("yolo26"));
  model_pb.write_bytes_field(8, opset_id("", cfg.opset_version));

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(model_pb.bytes().data(),
          (std::streamsize)model_pb.bytes().size());
}

// ─── Helpers for task-head exporters ──────────────────────────────────────
//
// Each version's task models share the same backbone+neck topology as the
// version's detect model. The walkers below mirror the in-tree yaml in
// yolo8.cpp / yolo11.cpp / yolo26.cpp respectively. Returns the per-level
// detect-input tensor names and (H, W).

namespace {

struct TrunkOut {
  std::vector<std::string>           det_ins;
  std::vector<std::pair<int, int>>   det_hw;
  std::vector<std::string>           all_outs;     // every layer (for proto access)
  std::vector<std::pair<int, int>>   all_hw;
};

// v8 yaml topology: layers 0..21 backbone+neck, then Detect/Segment/etc.
TrunkOut walk_v8_bb_neck(GraphBuilder& g, const std::string& input_name,
                         torch::nn::ModuleList& mlist, int imgsz) {
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1}, "C2f"}, {{-1}, "SPPF"},
      {{-1}, "Up"},   {{-1, 6},  "Cat"}, {{-1}, "C2f"},
      {{-1}, "Up"},   {{-1, 4},  "Cat"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1, 12}, "Cat"}, {{-1}, "C2f"},
      {{-1}, "Conv"}, {{-1, 9},  "Cat"}, {{-1}, "C2f"},
  };
  std::vector<std::string> outs(yaml.size());
  std::vector<std::pair<int, int>> hw(yaml.size());
  std::string prev = input_name;
  int H = imgsz, W = imgsz;
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
    auto hw_at = [&](int f) {
      return (f == -1) ? (i == 0 ? std::pair<int, int>{H, W} : hw[i - 1])
                       : hw[f];
    };
    auto prefix = "model." + std::to_string(i);
    if (s.kind == "Conv") {
      auto* m = mlist[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), prefix, m);
      auto in_hw = hw_at(s.from[0]);
      int st = m->conv->options.stride()->at(0);
      hw[i] = {in_hw.first / st, in_hw.second / st};
    } else if (s.kind == "C2f") {
      auto* m = mlist[i]->as<models::C2fImpl>();
      outs[i] = emit_c2f(g, idx_or_prev(s.from[0]), prefix, m);
      hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "SPPF") {
      auto* m = mlist[i]->as<models::SPPFImpl>();
      outs[i] = emit_sppf(g, idx_or_prev(s.from[0]), prefix, m);
      hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), prefix);
      auto in_hw = hw_at(s.from[0]);
      hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back((f == -1) ? prev : outs[f]);
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)},
                       prefix + ".cat");
      hw[i] = hw_at(s.from[0]);
    }
    prev = outs[i];
  }
  TrunkOut t;
  t.all_outs = outs;
  t.all_hw   = hw;
  // Detect-input layers for v8: 15, 18, 21.
  t.det_ins  = {outs[15], outs[18], outs[21]};
  t.det_hw   = {hw[15],   hw[18],   hw[21]};
  return t;
}

// v11 yaml topology: 23 backbone+neck layers (0..22).
TrunkOut walk_v11_bb_neck(GraphBuilder& g, const std::string& input_name,
                           torch::nn::ModuleList& mlist, int imgsz) {
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"}, {{-1}, "SPPF"},
      {{-1}, "C2PSA"},
      {{-1}, "Up"}, {{-1, 6},  "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Up"}, {{-1, 4},  "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1, 13}, "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1, 10}, "Cat"}, {{-1}, "C3k2"},
  };
  std::vector<std::string> outs(yaml.size());
  std::vector<std::pair<int, int>> hw(yaml.size());
  std::string prev = input_name;
  int H = imgsz, W = imgsz;
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
    auto hw_at = [&](int f) {
      return (f == -1) ? (i == 0 ? std::pair<int, int>{H, W} : hw[i - 1])
                       : hw[f];
    };
    auto prefix = "model." + std::to_string(i);
    if (s.kind == "Conv") {
      auto* m = mlist[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), prefix, m);
      auto in_hw = hw_at(s.from[0]);
      int st = m->conv->options.stride()->at(0);
      hw[i] = {in_hw.first / st, in_hw.second / st};
    } else if (s.kind == "C3k2") {
      auto* m = mlist[i]->as<models::C3k2Impl>();
      outs[i] = emit_c3k2(g, idx_or_prev(s.from[0]), prefix, m);
      hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "SPPF") {
      auto* m = mlist[i]->as<models::SPPFImpl>();
      outs[i] = emit_sppf(g, idx_or_prev(s.from[0]), prefix, m);
      hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "C2PSA") {
      auto* m = mlist[i]->as<models::C2PSAImpl>();
      auto in_hw = hw_at(s.from[0]);
      outs[i] = emit_c2psa(g, idx_or_prev(s.from[0]),
                           in_hw.first, in_hw.second, prefix, m);
      hw[i] = in_hw;
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), prefix);
      auto in_hw = hw_at(s.from[0]);
      hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back((f == -1) ? prev : outs[f]);
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, prefix + ".cat");
      hw[i] = hw_at(s.from[0]);
    }
    prev = outs[i];
  }
  TrunkOut t;
  t.all_outs = outs;
  t.all_hw   = hw;
  t.det_ins  = {outs[16], outs[19], outs[22]};
  t.det_hw   = {hw[16],   hw[19],   hw[22]};
  return t;
}

// v26 yaml: same as v11 but layer 22 is C2PSAf instead of C3k2.
TrunkOut walk_v26_bb_neck(GraphBuilder& g, const std::string& input_name,
                           torch::nn::ModuleList& mlist, int imgsz) {
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1}, "C3k2"}, {{-1}, "SPPF"},
      {{-1}, "C2PSA"},
      {{-1}, "Up"}, {{-1, 6},  "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Up"}, {{-1, 4},  "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1, 13}, "Cat"}, {{-1}, "C3k2"},
      {{-1}, "Conv"}, {{-1, 10}, "Cat"}, {{-1}, "C2PSAf"},
  };
  std::vector<std::string> outs(yaml.size());
  std::vector<std::pair<int, int>> hw(yaml.size());
  std::string prev = input_name;
  int H = imgsz, W = imgsz;
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
    auto hw_at = [&](int f) {
      return (f == -1) ? (i == 0 ? std::pair<int, int>{H, W} : hw[i - 1])
                       : hw[f];
    };
    auto prefix = "model." + std::to_string(i);
    if (s.kind == "Conv") {
      auto* m = mlist[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), prefix, m);
      auto in_hw = hw_at(s.from[0]);
      int st = m->conv->options.stride()->at(0);
      hw[i] = {in_hw.first / st, in_hw.second / st};
    } else if (s.kind == "C3k2") {
      auto* m = mlist[i]->as<models::C3k2Impl>();
      outs[i] = emit_c3k2(g, idx_or_prev(s.from[0]), prefix, m);
      hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "SPPF") {
      auto* m = mlist[i]->as<models::SPPFImpl>();
      outs[i] = emit_sppf(g, idx_or_prev(s.from[0]), prefix, m);
      hw[i] = hw_at(s.from[0]);
    } else if (s.kind == "C2PSA") {
      auto* m = mlist[i]->as<models::C2PSAImpl>();
      auto in_hw = hw_at(s.from[0]);
      outs[i] = emit_c2psa(g, idx_or_prev(s.from[0]),
                           in_hw.first, in_hw.second, prefix, m);
      hw[i] = in_hw;
    } else if (s.kind == "C2PSAf") {
      auto* m = mlist[i]->as<models::C2PSAfImpl>();
      auto in_hw = hw_at(s.from[0]);
      outs[i] = emit_c2psaf(g, idx_or_prev(s.from[0]),
                             in_hw.first, in_hw.second, prefix, m);
      hw[i] = in_hw;
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), prefix);
      auto in_hw = hw_at(s.from[0]);
      hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back((f == -1) ? prev : outs[f]);
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, prefix + ".cat");
      hw[i] = hw_at(s.from[0]);
    }
    prev = outs[i];
  }
  TrunkOut t;
  t.all_outs = outs;
  t.all_hw   = hw;
  t.det_ins  = {outs[16], outs[19], outs[22]};
  t.det_hw   = {hw[16],   hw[19],   hw[22]};
  return t;
}

// Common ModelProto write — used by every export below.
void write_model_proto(const GraphBuilder& g, const OnnxExportConfig& cfg,
                        const std::string& graph_name,
                        const std::string& path) {
  Pb model_pb;
  model_pb.write_int64_field(1, /*ir_version=*/8);
  model_pb.write_string_field(2, cfg.producer_name);
  model_pb.write_string_field(3, cfg.producer_version);
  model_pb.write_string_field(4, "");
  model_pb.write_int64_field(5, 1);
  model_pb.write_string_field(6, "");
  model_pb.write_bytes_field(7, g.build_graph_bytes(graph_name));
  model_pb.write_bytes_field(8, opset_id("", cfg.opset_version));
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(model_pb.bytes().data(),
          (std::streamsize)model_pb.bytes().size());
}

// ─── v12 emitters (A2C2f / AAttn / ABlock) ────────────────────────────
//
// AAttn forward (mirror of yolo12.cpp::AAttnImpl::forward):
//   1) qkv = Conv(x)                                  [B, 3C, H, W]
//   2) flatten + transpose                            [B, N, 3C]
//   3) optional reshape for area-windowing            [B*area, N/area, 3C]
//   4) view → permute(0,2,3,1)                        [Bg, num_heads, 3hd, Ng]
//   5) slice 3 → q, k, v                              [Bg, nh, hd, Ng] each
//   6) attn = softmax(qᵀ@k * scale, dim=-1)
//   7) out = v @ attnᵀ                                [Bg, nh, hd, Ng]
//   8) permute(0,3,1,2)                               [Bg, Ng, nh, hd]
//   9) reshape to [B, N, C] then [B, H, W, C] then permute(0,3,1,2)
//  10) v_p = v.permute(0,3,1,2) → [B, C, H, W] (same path)
//  11) out = out + pe(v_p);  return proj(out)
//
// pe is depthwise k=7 with conv.bias=True (v12 unique). emit_conv_module
// handles bias=False; we expand it inline for pe.
//
// ABlock: x = x + attn(x); x = x + mlp(x). mlp = Sequential(
//   Conv(dim, mlp_hidden, 1, act=true),
//   Conv(mlp_hidden, dim, 1, act=false))
//
// A2C2f: cv1 → c_inner. Then for each m[i]:
//   if a2: m[i] = Sequential(ABlock × 2) — call forward
//   else:  m[i] = C3k(c_inner, c_inner, n=2) — emit_c3k
// cv2 ← (1+n)*c_inner. Optional gamma residual gate at l/x.
std::string emit_aattn_v12(GraphBuilder& g, const std::string& in,
                           int B, int C, int H, int W,
                           const std::string& prefix,
                           models::AAttnImpl* a) {
  int nh = a->num_heads;
  int hd = a->head_dim;
  int N  = H * W;
  int area = a->area;
  int Bg = (area > 1) ? B * area : B;
  int Ng = (area > 1) ? N / area : N;

  // 1) qkv conv (act=False; v12 qkv has bias=False).
  auto qkv = emit_conv_module(g, in, prefix + ".qkv", a->qkv.get());
  // 2) Reshape [B, 3C, H, W] → [B, 3C, N]
  auto rshape0 = g.add_init_int64(prefix + ".rshape0",
                                   {(int64_t)B, (int64_t)(3*C), (int64_t)N});
  auto y3 = g.node("Reshape", {qkv, rshape0}, {}, prefix + ".y3");
  // Transpose perm=(0,2,1) → [B, N, 3C]
  auto y_flat = g.node("Transpose", {y3},
                        {attr_ints("perm", {0, 2, 1})}, prefix + ".y_flat");
  // 3) Optional area-window reshape → [Bg, Ng, 3C]
  std::string after_window = y_flat;
  if (area > 1) {
    auto ws = g.add_init_int64(prefix + ".win_shape",
                                {(int64_t)Bg, (int64_t)Ng, (int64_t)(3*C)});
    after_window = g.node("Reshape", {y_flat, ws}, {}, prefix + ".windowed");
  }
  // 4) View [Bg, Ng, nh, 3hd] → permute (0,2,3,1) → [Bg, nh, 3hd, Ng]
  auto vshape = g.add_init_int64(prefix + ".v.shape",
                                  {(int64_t)Bg, (int64_t)Ng, (int64_t)nh,
                                   (int64_t)(3*hd)});
  auto y4 = g.node("Reshape", {after_window, vshape}, {}, prefix + ".y4");
  auto qkv_h = g.node("Transpose", {y4},
                       {attr_ints("perm", {0, 2, 3, 1})}, prefix + ".qkv_h");
  // 5) Slice into q, k, v along dim=2 with size hd each.
  auto axes2 = g.add_init_int64(prefix + ".ax2", {2});
  auto step1 = g.add_init_int64(prefix + ".st1", {1});
  auto sq0 = g.add_init_int64(prefix + ".sq0", {0});
  auto sq1 = g.add_init_int64(prefix + ".sq1", {(int64_t)hd});
  auto q   = g.node("Slice", {qkv_h, sq0, sq1, axes2, step1}, {}, prefix + ".q");
  auto sk0 = g.add_init_int64(prefix + ".sk0", {(int64_t)hd});
  auto sk1 = g.add_init_int64(prefix + ".sk1", {(int64_t)(2*hd)});
  auto k   = g.node("Slice", {qkv_h, sk0, sk1, axes2, step1}, {}, prefix + ".k");
  auto sv0 = g.add_init_int64(prefix + ".sv0", {(int64_t)(2*hd)});
  auto sv1 = g.add_init_int64(prefix + ".sv1", {(int64_t)(3*hd)});
  auto v   = g.node("Slice", {qkv_h, sv0, sv1, axes2, step1}, {}, prefix + ".v");
  // 6) attn = softmax((qᵀ @ k) * scale, dim=-1)
  auto qT  = g.node("Transpose", {q},
                     {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".qT");
  auto attn0 = g.node("MatMul", {qT, k}, {}, prefix + ".attn0");
  std::vector<float> sv = {(float)a->scale};
  auto scale_init = g.add_init_float(prefix + ".scale", sv, {1});
  auto attn = g.node("Mul", {attn0, scale_init}, {}, prefix + ".attn.s");
  attn = g.node("Softmax", {attn}, {attr_int("axis", -1)}, prefix + ".attn.sm");
  // 7) out = v @ attnᵀ → [Bg, nh, hd, Ng]
  auto attnT = g.node("Transpose", {attn},
                       {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".attnT");
  auto out0 = g.node("MatMul", {v, attnT}, {}, prefix + ".out0");
  // 8) permute(0,3,1,2) → [Bg, Ng, nh, hd]
  auto out1 = g.node("Transpose", {out0},
                      {attr_ints("perm", {0, 3, 1, 2})}, prefix + ".out1");
  auto v_p1 = g.node("Transpose", {v},
                      {attr_ints("perm", {0, 3, 1, 2})}, prefix + ".v_p1");
  // 9) reshape to [B, N, C] (un-windows if area > 1) then to [B, H, W, C]
  //    then permute to [B, C, H, W].
  auto bnc_shape = g.add_init_int64(prefix + ".bnc",
                                     {(int64_t)B, (int64_t)N, (int64_t)C});
  auto out_bnc = g.node("Reshape", {out1, bnc_shape}, {}, prefix + ".out.bnc");
  auto v_p_bnc = g.node("Reshape", {v_p1, bnc_shape}, {}, prefix + ".v_p.bnc");
  auto bhwc_shape = g.add_init_int64(prefix + ".bhwc",
                                      {(int64_t)B, (int64_t)H, (int64_t)W,
                                       (int64_t)C});
  auto out_bhwc = g.node("Reshape", {out_bnc, bhwc_shape}, {}, prefix + ".out.bhwc");
  auto v_p_bhwc = g.node("Reshape", {v_p_bnc, bhwc_shape}, {}, prefix + ".v_p.bhwc");
  auto out_sp = g.node("Transpose", {out_bhwc},
                        {attr_ints("perm", {0, 3, 1, 2})}, prefix + ".out.sp");
  auto v_p_sp = g.node("Transpose", {v_p_bhwc},
                        {attr_ints("perm", {0, 3, 1, 2})}, prefix + ".v_p.sp");
  // 10) pe(v_p_sp). v12's pe has conv.bias=True; emit_conv_module honours
  //     the conv's stored bias when fusing — but ConvImpl with conv_bias=true
  //     stores bias on conv, and BN absorbs (Wx + b_conv - rm)*γ/sqrt(rv+ε)+β.
  //     fuse_conv_bn here doesn't have a b_conv path; v12 pe.conv.bias is
  //     stored separately. We fold it manually:
  //         scale = γ / sqrt(rv + ε)
  //         fused_w = cw * scale.view([-1,1,1,1])
  //         fused_b = (b_conv - rm) * scale + β
  auto* pe = a->pe.get();
  at::Tensor cw  = pe->conv->weight;
  at::Tensor cb  = pe->conv->bias.defined() ? pe->conv->bias
                                            : torch::zeros({cw.size(0)});
  at::Tensor bw  = pe->bn->weight;
  at::Tensor bb  = pe->bn->bias;
  at::Tensor brm = pe->bn->running_mean;
  at::Tensor brv = pe->bn->running_var;
  double eps     = pe->bn->options.eps();
  auto bn_scale  = bw / torch::sqrt(brv + eps);
  auto fused_w   = cw * bn_scale.view({-1, 1, 1, 1});
  auto fused_b   = (cb - brm) * bn_scale + bb;
  auto pe_w      = g.add_init(prefix + ".pe.weight", fused_w.to(torch::kFloat32));
  auto pe_b      = g.add_init(prefix + ".pe.bias",   fused_b.to(torch::kFloat32));
  int pe_k    = pe->conv->options.kernel_size()->at(0);
  int pe_pad  = std::get<torch::ExpandingArray<2>>(pe->conv->options.padding())->at(0);
  int pe_g    = (int)pe->conv->options.groups();
  auto pe_out = g.node("Conv", {v_p_sp, pe_w, pe_b},
                        {attr_ints("dilations", {1, 1}),
                         attr_int("group", pe_g),
                         attr_ints("kernel_shape", {pe_k, pe_k}),
                         attr_ints("pads", {pe_pad, pe_pad, pe_pad, pe_pad}),
                         attr_ints("strides", {1, 1})},
                        prefix + ".pe.out");
  auto added = g.node("Add", {out_sp, pe_out}, {}, prefix + ".add");
  return emit_conv_module(g, added, prefix + ".proj", a->proj.get());
}

std::string emit_ablock_v12(GraphBuilder& g, const std::string& in,
                            int B, int C, int H, int W,
                            const std::string& prefix,
                            models::ABlockImpl* b) {
  auto attn_out = emit_aattn_v12(g, in, B, C, H, W, prefix + ".attn",
                                  b->attn.get());
  auto y1 = g.node("Add", {in, attn_out}, {}, prefix + ".add1");
  auto* mlp0 = b->mlp->ptr(0)->as<models::ConvImpl>();
  auto* mlp1 = b->mlp->ptr(1)->as<models::ConvImpl>();
  auto m0 = emit_conv_module(g, y1, prefix + ".mlp.0", mlp0);
  auto m1 = emit_conv_module(g, m0, prefix + ".mlp.1", mlp1);
  return g.node("Add", {y1, m1}, {}, prefix + ".add2");
}

std::string emit_a2c2f_v12(GraphBuilder& g, const std::string& in,
                            int B, int H, int W,
                            const std::string& prefix,
                            models::A2C2fImpl* m) {
  auto y0 = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  std::vector<std::string> outs = {y0};
  std::string last = y0;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto sub_prefix = prefix + ".m." + std::to_string(i);
    if (m->a2) {
      auto* seq = m->m[i]->as<torch::nn::SequentialImpl>();
      // Sequential of 2 ABlocks.
      auto* ab0 = seq->ptr(0)->as<models::ABlockImpl>();
      auto* ab1 = seq->ptr(1)->as<models::ABlockImpl>();
      last = emit_ablock_v12(g, last, B, m->c_inner, H, W,
                              sub_prefix + ".0", ab0);
      last = emit_ablock_v12(g, last, B, m->c_inner, H, W,
                              sub_prefix + ".1", ab1);
    } else {
      auto* c3k = m->m[i]->as<models::C3kImpl>();
      last = emit_c3k(g, last, sub_prefix, c3k);
    }
    outs.push_back(last);
  }
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)}, prefix + ".cat");
  auto y = emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
  // Mirror v12 A2C2f forward: residual fires only when c1 == c2. has_gamma
  // is only registered when a2 && residual && c1 == c2 (yolo12.cpp ctor),
  // so it implies the channel match too.
  int c1 = m->cv1->conv->options.in_channels();
  int c2 = m->cv2->conv->options.out_channels();
  if (m->residual && m->has_gamma) {
    auto gw = g.add_init(prefix + ".gamma", m->gamma.detach());
    auto gw_r = g.node("Reshape", {gw,
                       g.add_init_int64(prefix + ".gamma.r",
                                         {1, (int64_t)m->gamma.size(0), 1, 1})},
                       {}, prefix + ".gamma.r.out");
    auto gy = g.node("Mul", {gw_r, y}, {}, prefix + ".gamma.mul");
    return g.node("Add", {in, gy}, {}, prefix + ".gamma.add");
  } else if (m->residual && c1 == c2) {
    return g.node("Add", {in, y}, {}, prefix + ".res.add");
  }
  return y;
}

// ─── v13 emitters (DSConv / DSC3k2 / V13AAttn / HyperACE / FullPAD) ─────

// DSConv: depthwise k×k → pointwise 1×1 → BN → SiLU.
// State-dict: dw.weight, pw.weight, bn.{weight,bias,running_mean,running_var}.
// dw and pw both have bias=False; BN absorbs bias as usual.
std::string emit_dsconv(GraphBuilder& g, const std::string& in,
                        const std::string& prefix,
                        models::DSConvImpl* m) {
  // Depthwise conv.
  auto* dwc = m->dw.get();
  auto dw_w = g.add_init(prefix + ".dw.weight", dwc->weight);
  int dw_k  = dwc->options.kernel_size()->at(0);
  int dw_s  = dwc->options.stride()->at(0);
  int dw_p  = std::get<torch::ExpandingArray<2>>(dwc->options.padding())->at(0);
  int dw_g  = (int)dwc->options.groups();
  auto dw_out = g.node("Conv", {in, dw_w},
                        {attr_ints("dilations", {1, 1}),
                         attr_int("group", dw_g),
                         attr_ints("kernel_shape", {dw_k, dw_k}),
                         attr_ints("pads", {dw_p, dw_p, dw_p, dw_p}),
                         attr_ints("strides", {dw_s, dw_s})},
                        prefix + ".dw.out");
  // Pointwise conv with BN-fold and SiLU.
  auto* pwc = m->pw.get();
  auto fused = fuse_conv_bn(pwc->weight, m->bn->weight, m->bn->bias,
                             m->bn->running_mean, m->bn->running_var,
                             m->bn->options.eps());
  auto pw_w = g.add_init(prefix + ".pw.weight", fused.weight);
  auto pw_b = g.add_init(prefix + ".pw.bias",   fused.bias);
  auto pw_out = g.node("Conv", {dw_out, pw_w, pw_b},
                        {attr_ints("dilations", {1, 1}),
                         attr_int("group", 1),
                         attr_ints("kernel_shape", {1, 1}),
                         attr_ints("pads", {0, 0, 0, 0}),
                         attr_ints("strides", {1, 1})},
                        prefix + ".pw.out");
  if (!m->act_silu) return pw_out;
  auto sig = g.node("Sigmoid", {pw_out}, {}, prefix + ".sig");
  return g.node("Mul", {pw_out, sig}, {}, prefix + ".silu");
}

std::string emit_dsbottleneck(GraphBuilder& g, const std::string& in,
                              const std::string& prefix,
                              models::DSBottleneckImpl* m) {
  auto y = emit_dsconv(g, in, prefix + ".cv1", m->cv1.get());
  y = emit_dsconv(g, y, prefix + ".cv2", m->cv2.get());
  return m->add ? g.node("Add", {in, y}, {}, prefix + ".add") : y;
}

std::string emit_dsc3k(GraphBuilder& g, const std::string& in,
                       const std::string& prefix,
                       models::DSC3kImpl* m) {
  auto a = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto* bn = m->m->ptr(i)->as<models::DSBottleneckImpl>();
    a = emit_dsbottleneck(g, a, prefix + ".m." + std::to_string(i), bn);
  }
  auto b = emit_conv_module(g, in, prefix + ".cv2", m->cv2.get());
  auto cat = g.node("Concat", {a, b}, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv3", m->cv3.get());
}

std::string emit_dsc3k2(GraphBuilder& g, const std::string& in,
                        const std::string& prefix,
                        models::DSC3k2Impl* m) {
  auto y = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  emit_split2(g, y, prefix + ".split", 2 * m->c_inner);
  std::string a = prefix + ".split.a";
  std::string b = prefix + ".split.b";
  std::vector<std::string> outs = {a, b};
  std::string last = b;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto sub_prefix = prefix + ".m." + std::to_string(i);
    if (m->dsc3k) {
      auto* sub = m->m[i]->as<models::DSC3kImpl>();
      last = emit_dsc3k(g, last, sub_prefix, sub);
    } else {
      auto* sub = m->m[i]->as<models::DSBottleneckImpl>();
      last = emit_dsbottleneck(g, last, sub_prefix, sub);
    }
    outs.push_back(last);
  }
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

// V13AAttn: distinct from v12 — separate qk (out=2C) and v (out=C) convs;
// pe is k=5 depthwise applied to v; final = proj(attn_out + pe(v_4d)).
std::string emit_v13_aattn(GraphBuilder& g, const std::string& in,
                           int B, int C, int H, int W,
                           const std::string& prefix,
                           models::V13AAttnImpl* a) {
  int nh = a->num_heads;
  int hd = a->head_dim;
  int N  = H * W;
  int area = a->area;
  int Bg = (area > 1) ? B * area : B;
  int Ng = (area > 1) ? N / area : N;

  auto qk = emit_conv_module(g, in, prefix + ".qk", a->qk.get());
  auto v_4d = emit_conv_module(g, in, prefix + ".v",  a->v.get());

  // pe(v_4d): bias=False here (v13 pe doesn't ship with conv bias).
  auto* pe = a->pe.get();
  auto pe_fused = fuse_conv_bn(pe->conv->weight, pe->bn->weight, pe->bn->bias,
                                pe->bn->running_mean, pe->bn->running_var,
                                pe->bn->options.eps());
  auto pe_w = g.add_init(prefix + ".pe.weight", pe_fused.weight);
  auto pe_b = g.add_init(prefix + ".pe.bias",   pe_fused.bias);
  int pe_k  = pe->conv->options.kernel_size()->at(0);
  int pe_pad = std::get<torch::ExpandingArray<2>>(pe->conv->options.padding())->at(0);
  int pe_g   = (int)pe->conv->options.groups();
  auto pe_out = g.node("Conv", {v_4d, pe_w, pe_b},
                        {attr_ints("dilations", {1, 1}),
                         attr_int("group", pe_g),
                         attr_ints("kernel_shape", {pe_k, pe_k}),
                         attr_ints("pads", {pe_pad, pe_pad, pe_pad, pe_pad}),
                         attr_ints("strides", {1, 1})},
                        prefix + ".pe.out");

  // qk_flat: [B, 2C, H, W] → [B, 2C, N] → transpose → [B, N, 2C]
  auto qk_n2c = g.add_init_int64(prefix + ".qk.n2c",
                                  {(int64_t)B, (int64_t)(2*C), (int64_t)N});
  auto qk_r = g.node("Reshape", {qk, qk_n2c}, {}, prefix + ".qk.r");
  auto qk_flat = g.node("Transpose", {qk_r},
                         {attr_ints("perm", {0, 2, 1})}, prefix + ".qk.flat");
  // v_flat: [B, N, C]
  auto v_nc = g.add_init_int64(prefix + ".v.n1c",
                                {(int64_t)B, (int64_t)C, (int64_t)N});
  auto v_r = g.node("Reshape", {v_4d, v_nc}, {}, prefix + ".v.r");
  auto v_flat = g.node("Transpose", {v_r},
                        {attr_ints("perm", {0, 2, 1})}, prefix + ".v.flat");
  // Optional area-window reshape.
  std::string qk_w = qk_flat;
  std::string v_w  = v_flat;
  if (area > 1) {
    auto qkw = g.add_init_int64(prefix + ".qk.win",
                                 {(int64_t)Bg, (int64_t)Ng, (int64_t)(2*C)});
    qk_w = g.node("Reshape", {qk_flat, qkw}, {}, prefix + ".qk.windowed");
    auto vw  = g.add_init_int64(prefix + ".v.win",
                                 {(int64_t)Bg, (int64_t)Ng, (int64_t)C});
    v_w  = g.node("Reshape", {v_flat, vw}, {}, prefix + ".v.windowed");
  }
  // Split qk → q, k along last dim (size C each).
  auto axes_last = g.add_init_int64(prefix + ".ax2", {2});
  auto step1     = g.add_init_int64(prefix + ".st1", {1});
  auto sq0 = g.add_init_int64(prefix + ".sq0", {0});
  auto sq1 = g.add_init_int64(prefix + ".sq1", {(int64_t)C});
  auto qf  = g.node("Slice", {qk_w, sq0, sq1, axes_last, step1}, {}, prefix + ".qf");
  auto sk0 = g.add_init_int64(prefix + ".sk0", {(int64_t)C});
  auto sk1 = g.add_init_int64(prefix + ".sk1", {(int64_t)(2*C)});
  auto kf  = g.node("Slice", {qk_w, sk0, sk1, axes_last, step1}, {}, prefix + ".kf");

  // Reshape q/k/v to (Bg, Ng, nh, hd) → transpose to (Bg, nh, hd, Ng).
  auto qkv_shape = g.add_init_int64(prefix + ".qkv.shape",
                                     {(int64_t)Bg, (int64_t)Ng,
                                      (int64_t)nh, (int64_t)hd});
  auto q_r = g.node("Reshape", {qf, qkv_shape}, {}, prefix + ".q.r");
  auto k_r = g.node("Reshape", {kf, qkv_shape}, {}, prefix + ".k.r");
  auto v_r2 = g.node("Reshape", {v_w, qkv_shape}, {}, prefix + ".v.r2");
  auto q_h = g.node("Transpose", {q_r},
                     {attr_ints("perm", {0, 2, 3, 1})}, prefix + ".q.h");
  auto k_h = g.node("Transpose", {k_r},
                     {attr_ints("perm", {0, 2, 3, 1})}, prefix + ".k.h");
  auto v_h = g.node("Transpose", {v_r2},
                     {attr_ints("perm", {0, 2, 3, 1})}, prefix + ".v.h");
  // attn = softmax(qᵀ @ k * scale, dim=-1)
  auto qT = g.node("Transpose", {q_h},
                    {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".qT");
  auto attn0 = g.node("MatMul", {qT, k_h}, {}, prefix + ".attn0");
  std::vector<float> sv = {(float)(1.0 / std::sqrt((double)hd))};
  auto scale_init = g.add_init_float(prefix + ".scale", sv, {1});
  auto attn = g.node("Mul", {attn0, scale_init}, {}, prefix + ".attn.s");
  attn = g.node("Softmax", {attn}, {attr_int("axis", -1)}, prefix + ".attn.sm");
  // out = v @ attnᵀ → (Bg, nh, hd, Ng) → permute (0,3,1,2) → (Bg, Ng, nh, hd)
  auto attnT = g.node("Transpose", {attn},
                       {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".attnT");
  auto out0 = g.node("MatMul", {v_h, attnT}, {}, prefix + ".out0");
  auto out1 = g.node("Transpose", {out0},
                      {attr_ints("perm", {0, 3, 1, 2})}, prefix + ".out1");
  // → reshape to (B, N, C) → (B, H, W, C) → permute to (B, C, H, W)
  auto bnc = g.add_init_int64(prefix + ".bnc",
                               {(int64_t)B, (int64_t)N, (int64_t)C});
  auto out_bnc = g.node("Reshape", {out1, bnc}, {}, prefix + ".out.bnc");
  auto bhwc = g.add_init_int64(prefix + ".bhwc",
                                {(int64_t)B, (int64_t)H, (int64_t)W,
                                 (int64_t)C});
  auto out_bhwc = g.node("Reshape", {out_bnc, bhwc}, {}, prefix + ".out.bhwc");
  auto out_sp = g.node("Transpose", {out_bhwc},
                        {attr_ints("perm", {0, 3, 1, 2})}, prefix + ".out.sp");
  auto added = g.node("Add", {out_sp, pe_out}, {}, prefix + ".add");
  return emit_conv_module(g, added, prefix + ".proj", a->proj.get());
}

std::string emit_v13_ablock(GraphBuilder& g, const std::string& in,
                            int B, int C, int H, int W,
                            const std::string& prefix,
                            models::V13ABlockImpl* b) {
  auto attn_out = emit_v13_aattn(g, in, B, C, H, W, prefix + ".attn",
                                  b->attn.get());
  auto y1 = g.node("Add", {in, attn_out}, {}, prefix + ".add1");
  auto* mlp0 = b->mlp->ptr(0)->as<models::ConvImpl>();
  auto* mlp1 = b->mlp->ptr(1)->as<models::ConvImpl>();
  auto m0 = emit_conv_module(g, y1, prefix + ".mlp.0", mlp0);
  auto m1 = emit_conv_module(g, m0, prefix + ".mlp.1", mlp1);
  return g.node("Add", {y1, m1}, {}, prefix + ".add2");
}

std::string emit_v13_a2c2f(GraphBuilder& g, const std::string& in,
                           int B, int H, int W,
                           const std::string& prefix,
                           models::V13A2C2fImpl* m) {
  auto y0 = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  std::vector<std::string> outs = {y0};
  std::string last = y0;
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto sp = prefix + ".m." + std::to_string(i);
    if (m->a2) {
      auto* seq = m->m[i]->as<torch::nn::SequentialImpl>();
      auto* ab0 = seq->ptr(0)->as<models::V13ABlockImpl>();
      auto* ab1 = seq->ptr(1)->as<models::V13ABlockImpl>();
      last = emit_v13_ablock(g, last, B, m->c_inner, H, W, sp + ".0", ab0);
      last = emit_v13_ablock(g, last, B, m->c_inner, H, W, sp + ".1", ab1);
    } else {
      auto* c3k = m->m[i]->as<models::C3kImpl>();
      last = emit_c3k(g, last, sp, c3k);
    }
    outs.push_back(last);
  }
  auto cat = g.node("Concat", outs, {attr_int("axis", 1)}, prefix + ".cat");
  auto y = emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
  if (m->has_gamma) {
    auto gw = g.add_init(prefix + ".gamma", m->gamma.detach());
    auto gw_r_init = g.add_init_int64(prefix + ".gamma.r",
                                       {1, (int64_t)m->gamma.size(0), 1, 1});
    auto gw_r = g.node("Reshape", {gw, gw_r_init}, {}, prefix + ".gamma.r.out");
    auto gy = g.node("Mul", {gw_r, y}, {}, prefix + ".gamma.mul");
    return g.node("Add", {in, gy}, {}, prefix + ".gamma.add");
  }
  return y;
}

std::string emit_downsample_conv_v13(GraphBuilder& g, const std::string& in,
                                      const std::string& prefix,
                                      models::DownsampleConvImpl* m) {
  auto pooled = g.node("AveragePool", {in},
                        {attr_ints("kernel_shape", {2, 2}),
                         attr_ints("strides",      {2, 2}),
                         attr_ints("pads",         {0, 0, 0, 0})},
                        prefix + ".pool");
  if (!m->channel_adjust) return pooled;
  return emit_conv_module(g, pooled, prefix + ".channel_adjust",
                           m->channel_adjust_conv.get());
}

std::string emit_full_pad_tunnel(GraphBuilder& g,
                                  const std::string& a,
                                  const std::string& b,
                                  const std::string& prefix,
                                  models::FullPADTunnelImpl* m) {
  auto gw = g.add_init(prefix + ".gate", m->gate.detach().view({1}));
  auto mul = g.node("Mul", {gw, b}, {}, prefix + ".mul");
  return g.node("Add", {a, mul}, {}, prefix + ".add");
}

std::string emit_fuse_module(GraphBuilder& g,
                              const std::vector<std::string>& xs,
                              const std::string& prefix,
                              models::FuseModuleImpl* m) {
  auto x0_ds = g.node("AveragePool", {xs[0]},
                       {attr_ints("kernel_shape", {2, 2}),
                        attr_ints("strides",      {2, 2}),
                        attr_ints("pads",         {0, 0, 0, 0})},
                       prefix + ".x0.ds");
  auto x2_up = emit_upsample_2x(g, xs[2], prefix + ".x2.up");
  auto cat = g.node("Concat", {x0_ds, xs[1], x2_up},
                     {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".conv_out", m->conv_out.get());
}

// AdaHyperedgeGen forward (operates on tokens [B, N, D] → [B, N, M]).
std::string emit_ada_hyperedge_gen(GraphBuilder& g, const std::string& in,
                                    int B, int N, int D,
                                    const std::string& prefix,
                                    models::AdaHyperedgeGenImpl* a) {
  int M  = a->num_hyperedges;
  int nh = a->num_heads;
  int hd = a->head_dim;

  // context = cat(mean(in, dim=1), max(in, dim=1)) for context="both"
  // shape (B, 2D). ReduceMean/ReduceMax use axes-as-attribute through
  // opset 17; opset 18+ moves axes to input. We target opset 17.
  auto mean = g.node("ReduceMean", {in},
                      {attr_int("keepdims", 0),
                       attr_ints("axes", {1})}, prefix + ".mean");
  auto maxv = g.node("ReduceMax", {in},
                      {attr_int("keepdims", 0),
                       attr_ints("axes", {1})}, prefix + ".maxv");
  auto ctx = g.node("Concat", {mean, maxv},
                     {attr_int("axis", -1)}, prefix + ".ctx");
  // context_net is a Linear: weight [M*D, 2D], bias [M*D]. y = ctx @ Wᵀ + b.
  auto wt = a->context_net->weight;             // [M*D, 2D]
  auto bt = a->context_net->bias;               // [M*D]
  // ONNX MatMul expects [B, K] @ [K, OUT] so we Transpose wt → [2D, M*D]
  auto cw = g.add_init(prefix + ".context_net.weight",
                        wt.transpose(0, 1).contiguous());
  auto cb = g.add_init(prefix + ".context_net.bias", bt);
  auto offsets_flat = g.node("MatMul", {ctx, cw}, {}, prefix + ".offsets.mm");
  offsets_flat = g.node("Add", {offsets_flat, cb}, {},
                         prefix + ".offsets.bias");
  // Reshape offsets to [B, M, D]
  auto off_shape = g.add_init_int64(prefix + ".off.shape",
                                     {(int64_t)B, (int64_t)M, (int64_t)D});
  auto offsets = g.node("Reshape", {offsets_flat, off_shape}, {},
                         prefix + ".offsets");
  // prototypes = prototype_base.unsqueeze(0) + offsets  → [B, M, D]
  auto proto_base_t = g.add_init(prefix + ".prototype_base",
                                  a->prototype_base.unsqueeze(0));
  auto prototypes = g.node("Add", {proto_base_t, offsets}, {},
                            prefix + ".prototypes");

  // pre_head_proj: Linear D → D. Apply elementwise on tokens.
  auto pw = a->pre_head_proj->weight;          // [D, D]
  auto pb = a->pre_head_proj->bias;            // [D]
  auto pwT = g.add_init(prefix + ".pre_head_proj.weight",
                         pw.transpose(0, 1).contiguous());
  auto pbT = g.add_init(prefix + ".pre_head_proj.bias", pb);
  auto x_proj = g.node("MatMul", {in, pwT}, {}, prefix + ".x_proj.mm");
  x_proj = g.node("Add", {x_proj, pbT}, {}, prefix + ".x_proj.bias");

  // Reshape & permute X_proj to (B, num_heads, N, head_dim)
  auto xshape = g.add_init_int64(prefix + ".x.shape",
                                  {(int64_t)B, (int64_t)N, (int64_t)nh,
                                   (int64_t)hd});
  auto x_r = g.node("Reshape", {x_proj, xshape}, {}, prefix + ".x.r");
  auto X_heads = g.node("Transpose", {x_r},
                         {attr_ints("perm", {0, 2, 1, 3})}, prefix + ".X_heads");
  // proto_heads: (B, M, num_heads, head_dim) → permute (0,2,1,3) → (B, nh, M, hd)
  auto pshape = g.add_init_int64(prefix + ".proto.shape",
                                  {(int64_t)B, (int64_t)M, (int64_t)nh,
                                   (int64_t)hd});
  auto p_r = g.node("Reshape", {prototypes, pshape}, {}, prefix + ".p.r");
  auto proto_heads = g.node("Transpose", {p_r},
                             {attr_ints("perm", {0, 2, 1, 3})},
                             prefix + ".proto_heads");
  // logits = X_heads @ proto_headsᵀ / sqrt(hd) → (B, nh, N, M)
  auto pT = g.node("Transpose", {proto_heads},
                    {attr_ints("perm", {0, 1, 3, 2})}, prefix + ".pT");
  auto logits_h = g.node("MatMul", {X_heads, pT}, {}, prefix + ".logits_h");
  std::vector<float> sv = {(float)(1.0 / a->scaling)};
  auto si = g.add_init_float(prefix + ".scale", sv, {1});
  logits_h = g.node("Mul", {logits_h, si}, {}, prefix + ".logits_h.s");
  // mean over num_heads (dim=1) → (B, N, M)
  auto logits = g.node("ReduceMean", {logits_h},
                        {attr_int("keepdims", 0),
                         attr_ints("axes", {1})}, prefix + ".logits");
  // softmax over N (dim=1)
  return g.node("Softmax", {logits}, {attr_int("axis", 1)}, prefix + ".A");
}

std::string emit_ada_hg_conv(GraphBuilder& g, const std::string& in,
                              int B, int N, int D,
                              const std::string& prefix,
                              models::AdaHGConvImpl* a) {
  // A = edge_generator(in)              [B, N, M]
  // He = bmm(Aᵀ, in)                    [B, M, D]
  // He = GELU(edge_proj.0(He))
  // X' = bmm(A, He)                     [B, N, D]
  // X' = GELU(node_proj.0(X'))
  // out = X' + in
  auto A = emit_ada_hyperedge_gen(g, in, B, N, D,
                                   prefix + ".edge_generator",
                                   a->edge_generator.get());
  auto AT = g.node("Transpose", {A},
                    {attr_ints("perm", {0, 2, 1})}, prefix + ".AT");
  auto He = g.node("MatMul", {AT, in}, {}, prefix + ".He");
  // edge_proj.0 Linear D→D:
  auto* ep0 = a->edge_proj->ptr(0)->as<torch::nn::LinearImpl>();
  auto epw = g.add_init(prefix + ".edge_proj.0.weight",
                         ep0->weight.transpose(0, 1).contiguous());
  auto epb = g.add_init(prefix + ".edge_proj.0.bias", ep0->bias);
  auto He2 = g.node("MatMul", {He, epw}, {}, prefix + ".ep0.mm");
  He2 = g.node("Add", {He2, epb}, {}, prefix + ".ep0.bias");
  // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))). Opset 17 has no Gelu op
  // (added in 20); decompose explicitly.
  {
    std::vector<float> ones = {1.0f};
    std::vector<float> half = {0.5f};
    std::vector<float> isqrt2 = {(float)(1.0 / std::sqrt(2.0))};
    auto one_t = g.add_init_float(prefix + ".ep0.gelu.one", ones, {1});
    auto half_t = g.add_init_float(prefix + ".ep0.gelu.half", half, {1});
    auto rs2 = g.add_init_float(prefix + ".ep0.gelu.rs2", isqrt2, {1});
    auto u = g.node("Mul", {He2, rs2}, {}, prefix + ".ep0.gelu.u");
    auto e = g.node("Erf", {u}, {}, prefix + ".ep0.gelu.erf");
    auto p1 = g.node("Add", {e, one_t}, {}, prefix + ".ep0.gelu.p1");
    auto hx = g.node("Mul", {He2, half_t}, {}, prefix + ".ep0.gelu.hx");
    He2 = g.node("Mul", {hx, p1}, {}, prefix + ".ep0.gelu");
  }
  auto X_new = g.node("MatMul", {A, He2}, {}, prefix + ".X_new");
  auto* np0 = a->node_proj->ptr(0)->as<torch::nn::LinearImpl>();
  auto npw = g.add_init(prefix + ".node_proj.0.weight",
                         np0->weight.transpose(0, 1).contiguous());
  auto npb = g.add_init(prefix + ".node_proj.0.bias", np0->bias);
  auto X2 = g.node("MatMul", {X_new, npw}, {}, prefix + ".np0.mm");
  X2 = g.node("Add", {X2, npb}, {}, prefix + ".np0.bias");
  {
    std::vector<float> ones = {1.0f};
    std::vector<float> half = {0.5f};
    std::vector<float> isqrt2 = {(float)(1.0 / std::sqrt(2.0))};
    auto one_t = g.add_init_float(prefix + ".np0.gelu.one", ones, {1});
    auto half_t = g.add_init_float(prefix + ".np0.gelu.half", half, {1});
    auto rs2 = g.add_init_float(prefix + ".np0.gelu.rs2", isqrt2, {1});
    auto u = g.node("Mul", {X2, rs2}, {}, prefix + ".np0.gelu.u");
    auto e = g.node("Erf", {u}, {}, prefix + ".np0.gelu.erf");
    auto p1 = g.node("Add", {e, one_t}, {}, prefix + ".np0.gelu.p1");
    auto hx = g.node("Mul", {X2, half_t}, {}, prefix + ".np0.gelu.hx");
    X2 = g.node("Mul", {hx, p1}, {}, prefix + ".np0.gelu");
  }
  return g.node("Add", {X2, in}, {}, prefix + ".res.add");
}

std::string emit_ada_hg_computation(GraphBuilder& g, const std::string& in,
                                     int B, int C, int H, int W,
                                     const std::string& prefix,
                                     models::AdaHGComputationImpl* m) {
  // tokens = in.flatten(2).transpose(1, 2)  → (B, N, C)
  int N = H * W;
  auto bcn = g.add_init_int64(prefix + ".bcn",
                               {(int64_t)B, (int64_t)C, (int64_t)N});
  auto x_r = g.node("Reshape", {in, bcn}, {}, prefix + ".x.r");
  auto tokens = g.node("Transpose", {x_r},
                        {attr_ints("perm", {0, 2, 1})}, prefix + ".tokens");
  auto y = emit_ada_hg_conv(g, tokens, B, N, C, prefix + ".hgnn",
                              m->hgnn.get());
  // back to (B, C, H, W)
  auto y_t = g.node("Transpose", {y},
                     {attr_ints("perm", {0, 2, 1})}, prefix + ".y.t");
  auto bchw = g.add_init_int64(prefix + ".bchw",
                                {(int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W});
  return g.node("Reshape", {y_t, bchw}, {}, prefix + ".out");
}

std::string emit_c3ah(GraphBuilder& g, const std::string& in,
                       int B, int H, int W,
                       const std::string& prefix,
                       models::C3AHImpl* m) {
  auto a = emit_conv_module(g, in, prefix + ".cv1", m->cv1.get());
  // Determine c_ from cv1's output channels.
  int c_ = m->cv1->conv->options.out_channels();
  a = emit_ada_hg_computation(g, a, B, c_, H, W, prefix + ".m", m->m.get());
  auto b = emit_conv_module(g, in, prefix + ".cv2", m->cv2.get());
  auto cat = g.node("Concat", {a, b}, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv3", m->cv3.get());
}

std::string emit_hyperace(GraphBuilder& g,
                           const std::vector<std::string>& xs,
                           int B, int H, int W,
                           const std::string& prefix,
                           models::HyperACEImpl* m) {
  auto x = emit_fuse_module(g, xs, prefix + ".fuse", m->fuse.get());
  auto cv1_out = emit_conv_module(g, x, prefix + ".cv1", m->cv1.get());
  // Split into 3 chunks of c_inner along channel.
  int c_in = m->c_inner;
  auto axes = g.add_init_int64(prefix + ".ax", {1});
  auto step = g.add_init_int64(prefix + ".st", {1});
  std::vector<std::string> y;
  for (int i = 0; i < 3; ++i) {
    auto s0 = g.add_init_int64(prefix + ".s0." + std::to_string(i),
                                {(int64_t)(i * c_in)});
    auto s1 = g.add_init_int64(prefix + ".s1." + std::to_string(i),
                                {(int64_t)((i + 1) * c_in)});
    y.push_back(g.node("Slice", {cv1_out, s0, s1, axes, step}, {},
                        prefix + ".chunk." + std::to_string(i)));
  }
  auto out1 = emit_c3ah(g, y[1], B, H, W,
                         prefix + ".branch1", m->branch1.get());
  auto out2 = emit_c3ah(g, y[1], B, H, W,
                         prefix + ".branch2", m->branch2.get());
  // m chain: y.append(m[i](y.back())). After loop y[1] = out1; y.append(out2).
  std::string last = y[2];
  for (size_t i = 0; i < m->m->size(); ++i) {
    auto sp = prefix + ".m." + std::to_string(i);
    auto mi = m->m[i];
    if (auto* d = mi->as<models::DSC3kImpl>()) {
      last = emit_dsc3k(g, last, sp, d);
    } else if (auto* d = mi->as<models::DSBottleneckImpl>()) {
      last = emit_dsbottleneck(g, last, sp, d);
    } else {
      throw std::runtime_error(
          "emit_hyperace: unexpected inner module type");
    }
    y.push_back(last);
  }
  y[1] = out1;
  y.push_back(out2);
  auto cat = g.node("Concat", y, {attr_int("axis", 1)}, prefix + ".cat");
  return emit_conv_module(g, cat, prefix + ".cv2", m->cv2.get());
}

// Spatial table for the 3 detect levels (h*w each).
std::vector<int> spatial_table(const std::vector<std::pair<int,int>>& hw) {
  std::vector<int> s;
  s.reserve(hw.size());
  for (auto& p : hw) s.push_back(p.first * p.second);
  return s;
}

}  // anonymous namespace

// ─── v12 detect exporter ──────────────────────────────────────────────────

void export_yolo12_onnx(models::Yolo12Detect& model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1, {1, 3, cfg.imgsz, cfg.imgsz});

  // v12 yaml: 22 layers ending in Detect (legacy=false).
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step> yaml = {
      {{-1}, "Conv"},   {{-1}, "Conv"},   {{-1}, "C3k2"},
      {{-1}, "Conv"},   {{-1}, "C3k2"},
      {{-1}, "Conv"},   {{-1}, "A2C2f"},
      {{-1}, "Conv"},   {{-1}, "A2C2f"},
      {{-1}, "Up"},     {{-1, 6}, "Cat"},  {{-1}, "A2C2f"},
      {{-1}, "Up"},     {{-1, 4}, "Cat"},  {{-1}, "A2C2f"},
      {{-1}, "Conv"},   {{-1, 11}, "Cat"}, {{-1}, "A2C2f"},
      {{-1}, "Conv"},   {{-1, 8},  "Cat"}, {{-1}, "C3k2"},
      {{14, 17, 20}, "Detect"},
  };

  std::vector<std::string> outs(yaml.size());
  std::vector<std::pair<int, int>> outs_hw(yaml.size());
  std::string prev = cfg.input_name;
  int H_in = cfg.imgsz, W_in = cfg.imgsz;
  int B_in = 1;

  auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
  auto hw_at = [&](int i, int f) -> std::pair<int, int> {
    if (f == -1) return (i == 0) ? std::pair<int, int>{H_in, W_in}
                                 : outs_hw[i - 1];
    return outs_hw[f];
  };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto pfx = "model." + std::to_string(i);
    if (s.kind == "Conv") {
      auto* m = model->model[i]->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), pfx, m);
      auto in_hw = hw_at(i, s.from[0]);
      int st = m->conv->options.stride()->at(0);
      outs_hw[i] = {in_hw.first / st, in_hw.second / st};
    } else if (s.kind == "C3k2") {
      auto* m = model->model[i]->as<models::C3k2Impl>();
      outs[i] = emit_c3k2(g, idx_or_prev(s.from[0]), pfx, m);
      outs_hw[i] = hw_at(i, s.from[0]);
    } else if (s.kind == "A2C2f") {
      auto* m = model->model[i]->as<models::A2C2fImpl>();
      auto in_hw = hw_at(i, s.from[0]);
      outs[i] = emit_a2c2f_v12(g, idx_or_prev(s.from[0]),
                                 B_in, in_hw.first, in_hw.second, pfx, m);
      outs_hw[i] = in_hw;
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), pfx);
      auto in_hw = hw_at(i, s.from[0]);
      outs_hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back(idx_or_prev(f));
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, pfx + ".cat");
      outs_hw[i] = hw_at(i, s.from[0]);
    } else if (s.kind == "Detect") {
      auto* d = model->model[i]->as<models::DetectImpl>();
      std::vector<std::string> det_in;
      std::vector<int>          det_ch;
      for (size_t k = 0; k < s.from.size(); ++k) {
        det_in.push_back(outs[s.from[k]]);
        det_ch.push_back(d->ch[k]);
      }
      outs[i] = emit_detect_v11(g, det_in, det_ch, model->stride, d,
                                  cfg.imgsz, pfx, cfg.output_name);
      outs_hw[i] = {0, 0};
    }
    prev = outs[i];
  }

  int nc = model->nc;
  g.set_output(cfg.output_name, /*FLOAT=*/1, {-1, 4 + nc, /*A*/ -1});
  write_model_proto(g, cfg, "yolo12", path);
}

// ─── v13 detect exporter ──────────────────────────────────────────────────

void export_yolo13_onnx(models::Yolo13Detect& model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1, {1, 3, cfg.imgsz, cfg.imgsz});

  // 33-layer v13 yaml; same kV13Yaml table as Yolo13DetectImpl.
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step> yaml = {
      {{-1},      "Conv"},   {{-1},      "Conv"},   {{-1}, "DSC3k2"},
      {{-1},      "Conv"},   {{-1},      "DSC3k2"},
      {{-1},      "DSConv"}, {{-1},      "A2C2f"},
      {{-1},      "DSConv"}, {{-1},      "A2C2f"},
      {{4,6,8},   "HyperACE"},
      {{-1},      "Up"},     {{9},       "DownsampleConv"},
      {{6,9},     "FullPADTunnel"}, {{4,10}, "FullPADTunnel"},
      {{8,11},    "FullPADTunnel"},
      {{-1},      "Up"},     {{-1,12},   "Cat"},    {{-1}, "DSC3k2"},
      {{-1,9},    "FullPADTunnel"},
      {{17},      "Up"},     {{-1,13},   "Cat"},    {{-1}, "DSC3k2"},
      {{10},      "Conv"},   {{21,22},   "FullPADTunnel"},
      {{-1},      "Conv"},   {{-1,18},   "Cat"},    {{-1}, "DSC3k2"},
      {{-1,9},    "FullPADTunnel"},
      {{26},      "Conv"},   {{-1,14},   "Cat"},    {{-1}, "DSC3k2"},
      {{-1,11},   "FullPADTunnel"},
      {{23,27,31},"Detect"},
  };

  std::vector<std::string> outs(yaml.size());
  std::vector<std::pair<int, int>> outs_hw(yaml.size());
  std::string prev = cfg.input_name;
  int H_in = cfg.imgsz, W_in = cfg.imgsz;
  int B_in = 1;

  auto idx_or_prev = [&](int f) { return (f == -1) ? prev : outs[f]; };
  auto hw_at = [&](int i, int f) -> std::pair<int, int> {
    if (f == -1) return (i == 0) ? std::pair<int, int>{H_in, W_in}
                                 : outs_hw[i - 1];
    return outs_hw[f];
  };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    auto pfx = "model." + std::to_string(i);
    auto mod = model->model[i];
    if (s.kind == "Conv") {
      auto* m = mod->as<models::ConvImpl>();
      outs[i] = emit_conv_module(g, idx_or_prev(s.from[0]), pfx, m);
      auto in_hw = hw_at(i, s.from[0]);
      int st = m->conv->options.stride()->at(0);
      outs_hw[i] = {in_hw.first / st, in_hw.second / st};
    } else if (s.kind == "DSConv") {
      auto* m = mod->as<models::DSConvImpl>();
      outs[i] = emit_dsconv(g, idx_or_prev(s.from[0]), pfx, m);
      auto in_hw = hw_at(i, s.from[0]);
      int st = m->dw->options.stride()->at(0);
      outs_hw[i] = {in_hw.first / st, in_hw.second / st};
    } else if (s.kind == "DSC3k2") {
      auto* m = mod->as<models::DSC3k2Impl>();
      outs[i] = emit_dsc3k2(g, idx_or_prev(s.from[0]), pfx, m);
      outs_hw[i] = hw_at(i, s.from[0]);
    } else if (s.kind == "A2C2f") {
      auto* m = mod->as<models::V13A2C2fImpl>();
      auto in_hw = hw_at(i, s.from[0]);
      outs[i] = emit_v13_a2c2f(g, idx_or_prev(s.from[0]),
                                 B_in, in_hw.first, in_hw.second, pfx, m);
      outs_hw[i] = in_hw;
    } else if (s.kind == "Up") {
      outs[i] = emit_upsample_2x(g, idx_or_prev(s.from[0]), pfx);
      auto in_hw = hw_at(i, s.from[0]);
      outs_hw[i] = {in_hw.first * 2, in_hw.second * 2};
    } else if (s.kind == "DownsampleConv") {
      auto* m = mod->as<models::DownsampleConvImpl>();
      outs[i] = emit_downsample_conv_v13(g, idx_or_prev(s.from[0]), pfx, m);
      auto in_hw = hw_at(i, s.from[0]);
      outs_hw[i] = {in_hw.first / 2, in_hw.second / 2};
    } else if (s.kind == "Cat") {
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back(idx_or_prev(f));
      outs[i] = g.node("Concat", ins, {attr_int("axis", 1)}, pfx + ".cat");
      outs_hw[i] = hw_at(i, s.from[0]);
    } else if (s.kind == "FullPADTunnel") {
      auto* m = mod->as<models::FullPADTunnelImpl>();
      outs[i] = emit_full_pad_tunnel(g, idx_or_prev(s.from[0]),
                                       idx_or_prev(s.from[1]), pfx, m);
      outs_hw[i] = hw_at(i, s.from[0]);
    } else if (s.kind == "HyperACE") {
      auto* m = mod->as<models::HyperACEImpl>();
      // HyperACE output H,W = the spatial of fuse's middle input (xs[1]).
      auto mid_hw = hw_at(i, s.from[1]);
      std::vector<std::string> ins;
      for (int f : s.from) ins.push_back(outs[f]);
      outs[i] = emit_hyperace(g, ins, B_in, mid_hw.first, mid_hw.second,
                                pfx, m);
      outs_hw[i] = mid_hw;
    } else if (s.kind == "Detect") {
      auto* d = mod->as<models::DetectImpl>();
      std::vector<std::string> det_in;
      std::vector<int>          det_ch;
      for (size_t k = 0; k < s.from.size(); ++k) {
        det_in.push_back(outs[s.from[k]]);
        det_ch.push_back(d->ch[k]);
      }
      outs[i] = emit_detect_v11(g, det_in, det_ch, model->stride, d,
                                  cfg.imgsz, pfx, cfg.output_name);
      outs_hw[i] = {0, 0};
    }
    prev = outs[i];
  }

  int nc = model->nc;
  g.set_output(cfg.output_name, /*FLOAT=*/1, {-1, 4 + nc, /*A*/ -1});
  write_model_proto(g, cfg, "yolo13", path);
}

// ─── Segment exporters ────────────────────────────────────────────────────
//
// Three outputs: "output" (decoded [N,4+nc,A]), "coefs" [N,nm,A],
// "protos" [N,nm,h_p,w_p]. The Proto module takes the P3 feature
// (smallest stride, largest spatial) and 2× upsamples it.

void export_yolo8_segment_onnx(models::Yolo8Segment& model,
                                const std::string& path,
                                const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1,
              {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v8_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 22;
  auto* seg = mlist[HEAD_IDX]->as<models::SegmentImpl>();
  auto* d   = seg->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  // Decoded box+cls.
  emit_detect(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
              head_prefix + ".detect", cfg.output_name);
  // Mask coefs cv4 chain.
  auto coefs = emit_task_cv4_concat(g, trunk.det_ins,
                                     spatial_table(trunk.det_hw),
                                     seg->cv4, seg->nm,
                                     head_prefix);
  g.rename_output(coefs, "coefs");
  // Proto from P3 (det_ins[0]).
  auto protos = emit_proto(g, trunk.det_ins[0], head_prefix + ".proto",
                            seg->proto.get());
  g.rename_output(protos, "protos");

  int nc = model->nc;
  int nm = seg->nm;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("coefs",        1, {-1, nm, -1});
  g.set_output("protos",       1, {-1, nm, -1, -1});
  write_model_proto(g, cfg, "yolo8_seg", path);
}

void export_yolo11_segment_onnx(models::Yolo11Segment& model,
                                 const std::string& path,
                                 const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v11_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 23;
  auto* seg = mlist[HEAD_IDX]->as<models::SegmentImpl>();
  auto* d   = seg->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  emit_detect_v11(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                   head_prefix + ".detect", cfg.output_name);
  auto coefs  = emit_task_cv4_concat(g, trunk.det_ins,
                                      spatial_table(trunk.det_hw),
                                      seg->cv4, seg->nm, head_prefix);
  g.rename_output(coefs, "coefs");
  auto protos = emit_proto(g, trunk.det_ins[0], head_prefix + ".proto",
                            seg->proto.get());
  g.rename_output(protos, "protos");

  int nc = model->nc;
  int nm = seg->nm;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("coefs",        1, {-1, nm, -1});
  g.set_output("protos",       1, {-1, nm, -1, -1});
  write_model_proto(g, cfg, "yolo11_seg", path);
}

void export_yolo26_segment_onnx(models::Yolo26Segment& model,
                                 const std::string& path,
                                 const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v26_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 23;
  auto* seg = mlist[HEAD_IDX]->as<models::Segment26Impl>();
  auto* d   = seg->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  emit_detect_v26(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                   head_prefix + ".detect", cfg.output_name);
  auto coefs  = emit_task_cv4_concat(g, trunk.det_ins,
                                      spatial_table(trunk.det_hw),
                                      seg->cv4, seg->nm, head_prefix);
  g.rename_output(coefs, "coefs");
  auto protos = emit_proto(g, trunk.det_ins[0], head_prefix + ".proto",
                            seg->proto.get());
  g.rename_output(protos, "protos");

  int nc = model->nc;
  int nm = seg->nm;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("coefs",        1, {-1, nm, -1});
  g.set_output("protos",       1, {-1, nm, -1, -1});
  write_model_proto(g, cfg, "yolo26_seg", path);
}

// ─── Pose exporters ───────────────────────────────────────────────────────

namespace {

// Build the flat anchor / stride tables to pass to the kpt decoder for
// the three detect levels (uses the strides field on the model).
AnchorTables build_pose_anchor_tables(const std::vector<std::pair<int,int>>& det_hw,
                                       const std::vector<double>& strides) {
  std::vector<int> spatial = spatial_table(det_hw);
  return build_anchor_tables(spatial, strides, det_hw);
}

}  // anonymous namespace

void export_yolo8_pose_onnx(models::Yolo8Pose& model,
                             const std::string& path,
                             const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v8_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 22;
  auto* pose = mlist[HEAD_IDX]->as<models::PoseImpl>();
  auto* d    = pose->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  emit_detect(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
              head_prefix + ".detect", cfg.output_name);

  auto raw = emit_task_cv4_concat(g, trunk.det_ins,
                                   spatial_table(trunk.det_hw),
                                   pose->cv4, pose->nk, head_prefix);
  auto tab = build_pose_anchor_tables(trunk.det_hw, model->stride);
  emit_kpt_decode(g, raw, pose->num_kpts, pose->kpt_dim, tab.total_A,
                   tab.anc_xy, tab.strides_per_a, head_prefix, "keypoints");

  int nc = model->nc;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("keypoints",     1, {-1, pose->nk, -1});
  write_model_proto(g, cfg, "yolo8_pose", path);
}

void export_yolo11_pose_onnx(models::Yolo11Pose& model,
                              const std::string& path,
                              const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v11_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 23;
  auto* pose = mlist[HEAD_IDX]->as<models::PoseImpl>();
  auto* d    = pose->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  emit_detect_v11(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                   head_prefix + ".detect", cfg.output_name);

  auto raw = emit_task_cv4_concat(g, trunk.det_ins,
                                   spatial_table(trunk.det_hw),
                                   pose->cv4, pose->nk, head_prefix);
  auto tab = build_pose_anchor_tables(trunk.det_hw, model->stride);
  emit_kpt_decode(g, raw, pose->num_kpts, pose->kpt_dim, tab.total_A,
                   tab.anc_xy, tab.strides_per_a, head_prefix, "keypoints");

  int nc = model->nc;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("keypoints",     1, {-1, pose->nk, -1});
  write_model_proto(g, cfg, "yolo11_pose", path);
}

void export_yolo26_pose_onnx(models::Yolo26Pose& model,
                              const std::string& path,
                              const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v26_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 23;
  auto* pose = mlist[HEAD_IDX]->as<models::Pose26Impl>();
  auto* d    = pose->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  emit_detect_v26(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                   head_prefix + ".detect", cfg.output_name);

  // v26 cv4 emits nk + nk_sigma channels per anchor; slice off the σ
  // channels before decoding (they're a training-only signal).
  int total = pose->nk + pose->nk_sigma;
  auto raw_full = emit_task_cv4_concat(g, trunk.det_ins,
                                        spatial_table(trunk.det_hw),
                                        pose->cv4, total, head_prefix);
  auto axes_ch  = g.add_init_int64(head_prefix + ".kpt.slice.ax", {1});
  auto step1    = g.add_init_int64(head_prefix + ".kpt.slice.st", {1});
  auto sk0      = g.add_init_int64(head_prefix + ".kpt.slice.s0", {0});
  auto sk1      = g.add_init_int64(head_prefix + ".kpt.slice.s1",
                                    {(int64_t)pose->nk});
  auto raw = g.node("Slice", {raw_full, sk0, sk1, axes_ch, step1}, {},
                    head_prefix + ".kpt.raw");

  auto tab = build_pose_anchor_tables(trunk.det_hw, model->stride);
  emit_kpt_decode(g, raw, pose->num_kpts, pose->kpt_dim, tab.total_A,
                   tab.anc_xy, tab.strides_per_a, head_prefix, "keypoints");

  int nc = model->nc;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("keypoints",     1, {-1, pose->nk, -1});
  write_model_proto(g, cfg, "yolo26_pose", path);
}

// ─── OBB exporters ────────────────────────────────────────────────────────

void export_yolo8_obb_onnx(models::Yolo8OBB& model,
                            const std::string& path,
                            const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v8_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 22;
  auto* obb = mlist[HEAD_IDX]->as<models::OBBImpl>();
  auto* d   = obb->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  // 1) Angle first (cv4 → sigmoid+shift → "angle" output).
  auto raw   = emit_task_cv4_concat(g, trunk.det_ins,
                                     spatial_table(trunk.det_hw),
                                     obb->cv4, obb->ne, head_prefix);
  int total_A = 0;
  for (auto p : trunk.det_hw) total_A += p.first * p.second;
  emit_obb_angle_decode(g, raw, obb->ne, total_A, head_prefix, "angle");
  // 2) cv2/cv3 + DFL + rotated decode using the angle.
  emit_detect_obb_dfl(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                       /*angle_in=*/"angle", /*v11_cv3=*/false,
                       head_prefix + ".detect", cfg.output_name);

  int nc = model->nc;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("angle",         1, {-1, -1});
  write_model_proto(g, cfg, "yolo8_obb", path);
}

void export_yolo11_obb_onnx(models::Yolo11OBB& model,
                             const std::string& path,
                             const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v11_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 23;
  auto* obb = mlist[HEAD_IDX]->as<models::OBBImpl>();
  auto* d   = obb->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  auto raw = emit_task_cv4_concat(g, trunk.det_ins,
                                   spatial_table(trunk.det_hw),
                                   obb->cv4, obb->ne, head_prefix);
  int total_A = 0;
  for (auto p : trunk.det_hw) total_A += p.first * p.second;
  emit_obb_angle_decode(g, raw, obb->ne, total_A, head_prefix, "angle");
  emit_detect_obb_dfl(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                       /*angle_in=*/"angle", /*v11_cv3=*/true,
                       head_prefix + ".detect", cfg.output_name);

  int nc = model->nc;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("angle",         1, {-1, -1});
  write_model_proto(g, cfg, "yolo11_obb", path);
}

void export_yolo26_obb_onnx(models::Yolo26OBB& model,
                             const std::string& path,
                             const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});
  auto& mlist = model->model;
  auto trunk = walk_v26_bb_neck(g, cfg.input_name, mlist, cfg.imgsz);

  const int HEAD_IDX = 23;
  auto* obb = mlist[HEAD_IDX]->as<models::OBB26Impl>();
  auto* d   = obb->detect.get();
  std::string head_prefix = "model." + std::to_string(HEAD_IDX);
  auto raw = emit_task_cv4_concat(g, trunk.det_ins,
                                   spatial_table(trunk.det_hw),
                                   obb->cv4, obb->ne, head_prefix);
  int total_A = 0;
  for (auto p : trunk.det_hw) total_A += p.first * p.second;
  emit_obb_angle_decode(g, raw, obb->ne, total_A, head_prefix, "angle");
  emit_detect_obb_v26(g, trunk.det_ins, d->ch, model->stride, d, cfg.imgsz,
                       /*angle_in=*/"angle",
                       head_prefix + ".detect", cfg.output_name);

  int nc = model->nc;
  g.set_output(cfg.output_name, 1, {-1, 4 + nc, -1});
  g.set_output("angle",         1, {-1, -1});
  write_model_proto(g, cfg, "yolo26_obb", path);
}

// ─── Classify exporters ───────────────────────────────────────────────────
//
// Architecture (yolo8-cls / yolo11-cls / yolo26-cls):
//   backbone (Conv / C2f or C3k2 / SPPF? not used / C2PSA for v11/v26)
//   final layer = ClassifyImpl: Conv 1x1 → AdaptiveAvgPool2d(1) → Flatten → Linear
// Output: [N, nc] pre-softmax logits.

namespace {

// Emit ClassifyImpl head: Conv (with BN folded) → GlobalAveragePool →
// Flatten(axis=1) → Gemm.
std::string emit_classify_head(GraphBuilder& g, const std::string& in,
                                const std::string& prefix,
                                models::ClassifyImpl* h,
                                const std::string& out_name) {
  auto y = emit_conv_module(g, in, prefix + ".conv", h->conv.get());
  // GlobalAveragePool — equivalent to AdaptiveAvgPool2d(1) on a [N,C,H,W]
  // feature map, returns [N, C, 1, 1].
  y = g.node("GlobalAveragePool", {y}, {}, prefix + ".gap");
  // Flatten axis=1 → [N, C].
  y = g.node("Flatten", {y}, {attr_int("axis", 1)}, prefix + ".flat");
  // Gemm: [N, C] @ W^T + b  (W is [nc, C], so set transB=1).
  auto* lin = h->linear.get();
  auto wname = g.add_init(prefix + ".linear.weight", lin->weight);
  auto bname = g.add_init(prefix + ".linear.bias",   lin->bias);
  return g.node("Gemm", {y, wname, bname},
                 {attr_float("alpha", 1.0f),
                  attr_float("beta",  1.0f),
                  attr_int  ("transB", 1)},
                 out_name);
}

}  // anonymous namespace

void export_yolo8_classify_onnx(models::Yolo8Classify& model,
                                const std::string& path,
                                const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});

  // yolo8-cls topology — Conv/C2f chain ending in Classify.
  struct Step { std::string kind; };
  static const std::vector<Step> yaml = {
      {"Conv"}, {"Conv"}, {"C2f"}, {"Conv"}, {"C2f"}, {"Conv"},
      {"C2f"},  {"Conv"}, {"C2f"}, {"Classify"},
  };
  auto& mlist = model->model;
  std::string prev = cfg.input_name;
  for (size_t i = 0; i < yaml.size(); ++i) {
    auto prefix = "model." + std::to_string(i);
    if (yaml[i].kind == "Conv") {
      auto* m = mlist[i]->as<models::ConvImpl>();
      prev = emit_conv_module(g, prev, prefix, m);
    } else if (yaml[i].kind == "C2f") {
      auto* m = mlist[i]->as<models::C2fImpl>();
      prev = emit_c2f(g, prev, prefix, m);
    } else if (yaml[i].kind == "Classify") {
      auto* m = mlist[i]->as<models::ClassifyImpl>();
      emit_classify_head(g, prev, prefix, m, cfg.output_name);
    }
  }
  g.set_output(cfg.output_name, 1, {-1, model->nc});
  write_model_proto(g, cfg, "yolo8_cls", path);
}

void export_yolo11_classify_onnx(models::Yolo11Classify& model,
                                  const std::string& path,
                                  const OnnxExportConfig& cfg) {
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});

  struct Step { std::string kind; };
  static const std::vector<Step> yaml = {
      {"Conv"},  {"Conv"},  {"C3k2"},
      {"Conv"},  {"C3k2"},
      {"Conv"},  {"C3k2"},
      {"Conv"},  {"C3k2"},
      {"C2PSA"},
      {"Classify"},
  };
  auto& mlist = model->model;
  std::string prev = cfg.input_name;
  std::pair<int,int> hw = {cfg.imgsz, cfg.imgsz};
  for (size_t i = 0; i < yaml.size(); ++i) {
    auto prefix = "model." + std::to_string(i);
    if (yaml[i].kind == "Conv") {
      auto* m = mlist[i]->as<models::ConvImpl>();
      prev = emit_conv_module(g, prev, prefix, m);
      int st = m->conv->options.stride()->at(0);
      hw = {hw.first / st, hw.second / st};
    } else if (yaml[i].kind == "C3k2") {
      auto* m = mlist[i]->as<models::C3k2Impl>();
      prev = emit_c3k2(g, prev, prefix, m);
    } else if (yaml[i].kind == "C2PSA") {
      auto* m = mlist[i]->as<models::C2PSAImpl>();
      prev = emit_c2psa(g, prev, hw.first, hw.second, prefix, m);
    } else if (yaml[i].kind == "Classify") {
      auto* m = mlist[i]->as<models::ClassifyImpl>();
      emit_classify_head(g, prev, prefix, m, cfg.output_name);
    }
  }
  g.set_output(cfg.output_name, 1, {-1, model->nc});
  write_model_proto(g, cfg, "yolo11_cls", path);
}

void export_yolo26_classify_onnx(models::Yolo26Classify& model,
                                  const std::string& path,
                                  const OnnxExportConfig& cfg) {
  // v26-cls topology mirrors v11-cls (no SPPF, C2PSA before Classify).
  model->eval();
  GraphBuilder g;
  g.set_input(cfg.input_name, 1, {-1, 3, cfg.imgsz, cfg.imgsz});

  struct Step { std::string kind; };
  static const std::vector<Step> yaml = {
      {"Conv"},  {"Conv"},  {"C3k2"},
      {"Conv"},  {"C3k2"},
      {"Conv"},  {"C3k2"},
      {"Conv"},  {"C3k2"},
      {"C2PSA"},
      {"Classify"},
  };
  auto& mlist = model->model;
  std::string prev = cfg.input_name;
  std::pair<int,int> hw = {cfg.imgsz, cfg.imgsz};
  for (size_t i = 0; i < yaml.size(); ++i) {
    auto prefix = "model." + std::to_string(i);
    if (yaml[i].kind == "Conv") {
      auto* m = mlist[i]->as<models::ConvImpl>();
      prev = emit_conv_module(g, prev, prefix, m);
      int st = m->conv->options.stride()->at(0);
      hw = {hw.first / st, hw.second / st};
    } else if (yaml[i].kind == "C3k2") {
      auto* m = mlist[i]->as<models::C3k2Impl>();
      prev = emit_c3k2(g, prev, prefix, m);
    } else if (yaml[i].kind == "C2PSA") {
      auto* m = mlist[i]->as<models::C2PSAImpl>();
      prev = emit_c2psa(g, prev, hw.first, hw.second, prefix, m);
    } else if (yaml[i].kind == "Classify") {
      auto* m = mlist[i]->as<models::ClassifyImpl>();
      emit_classify_head(g, prev, prefix, m, cfg.output_name);
    }
  }
  g.set_output(cfg.output_name, 1, {-1, model->nc});
  write_model_proto(g, cfg, "yolo26_cls", path);
}

}  // namespace yolocpp::serialization
