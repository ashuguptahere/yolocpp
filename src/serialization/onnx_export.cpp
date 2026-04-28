// Self-contained ONNX exporter for YOLOv8.
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

// AttributeProto field numbers (from onnx.proto):
//   1 name (string)
//   2 ref_attr_name
//   3 doc_string
//   4 (deprecated)
//   5 type (enum AttributeType)
//      0 UNDEFINED
//      1 FLOAT, 2 INT, 3 STRING, 4 TENSOR, 5 GRAPH
//      6 FLOATS, 7 INTS, 8 STRINGS, 9 TENSORS, 10 GRAPHS, 11 SPARSE_TENSOR
//      12 SPARSE_TENSORS, 13 TYPE_PROTO, 14 TYPE_PROTOS
//   6 f (float)
//   7 i (int64)
//   8 s (bytes)
//   10 t (TensorProto)
//   ...
//   for repeated: 7 ints, 8 floats, 9 strings, 11 tensors

std::string attr_int(const std::string& name, int64_t v) {
  Pb a;
  a.write_string_field(1, name);
  a.write_int32_field(5, /*INT=*/2);
  a.write_int64_field(3, v);  // 'i' in onnx.proto is field 3
  return a.bytes();
}
std::string attr_ints(const std::string& name, const std::vector<int64_t>& xs) {
  Pb a;
  a.write_string_field(1, name);
  a.write_int32_field(5, /*INTS=*/7);
  // Field 8 = 'ints' in onnx.proto, packed repeated.
  a.write_packed_int64s(8, xs);
  return a.bytes();
}
std::string attr_float(const std::string& name, float v) {
  Pb a;
  a.write_string_field(1, name);
  a.write_int32_field(5, /*FLOAT=*/1);
  // Field 2 = 'f' (single float), wire type 5 (fixed32).
  a.write_tag(2, 5);
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  a.bytes().push_back((char)(bits & 0xff));
  a.bytes().push_back((char)((bits >> 8) & 0xff));
  a.bytes().push_back((char)((bits >> 16) & 0xff));
  a.bytes().push_back((char)((bits >> 24) & 0xff));
  return a.bytes();
}
std::string attr_string(const std::string& name, const std::string& v) {
  Pb a;
  a.write_string_field(1, name);
  a.write_int32_field(5, /*STRING=*/3);
  // Field 4 = 's' (bytes).
  a.write_bytes_field(4, v);
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
  void set_output(const std::string& name, int dt,
                  const std::vector<int64_t>& shape) {
    output_name_  = name;
    output_dt_    = dt;
    output_shape_ = shape;
  }

  std::string build_graph_bytes(const std::string& graph_name) const {
    Pb g;
    for (const auto& n : nodes_)        g.write_bytes_field(1, n);
    g.write_string_field(2, graph_name);
    for (const auto& t : inits_)        g.write_bytes_field(5, t);
    g.write_bytes_field(11, value_info(input_name_,  input_dt_,  input_shape_));
    g.write_bytes_field(12, value_info(output_name_, output_dt_, output_shape_));
    return g.bytes();
  }

  std::string unique(const std::string& base) {
    return base + "_" + std::to_string(seq_++);
  }

 private:
  std::vector<std::string> nodes_;
  std::vector<std::string> inits_;
  std::string              input_name_, output_name_;
  int                      input_dt_ = 1, output_dt_ = 1;
  std::vector<int64_t>     input_shape_, output_shape_;
  int                      seq_ = 0;
};

// ──────────────────────────────────────────────────────────────────────────
// YOLOv8 → ONNX: the layer emitters.

// Fold BN into preceding Conv weights & bias (since our Conv has bias=False
// it becomes pure conv-with-bias post-fusion).
struct FusedConv {
  at::Tensor weight;  // [Cout, Cin/groups, kH, kW]
  at::Tensor bias;    // [Cout]
};
FusedConv fuse_conv_bn(const at::Tensor& cw,
                       const at::Tensor& bw, const at::Tensor& bb,
                       const at::Tensor& brm, const at::Tensor& brv,
                       double eps = 1e-5) {
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
                            bool act_silu) {
  auto fused = fuse_conv_bn(cw, bw, bb, brm, brv);
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
  // SiLU = x * sigmoid(x). Could also use HardSigmoid but YOLOv8 uses SiLU.
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
                            (int)cv1->conv->options.groups(), true);
  auto y2 = emit_conv_block(g, y1, prefix + ".cv2",
                            cv2->conv->weight, cv2->bn->weight,
                            cv2->bn->bias, cv2->bn->running_mean,
                            cv2->bn->running_var,
                            cv2->conv->options.stride()->at(0),
                            std::get<torch::ExpandingArray<2>>(cv2->conv->options.padding())->at(0),
                            (int)cv2->conv->options.groups(), true);
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
                         m->act_silu);
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

}  // anonymous namespace

void export_yolov8_onnx(models::YoloV8Detect& model,
                        const std::string&    path,
                        const OnnxExportConfig& cfg) {
  // Force eval (BN running stats stable).
  model->eval();

  GraphBuilder g;
  g.set_input(cfg.input_name, /*FLOAT=*/1,
              {-1, 3, cfg.imgsz, cfg.imgsz});

  // We replicate the v8 YAML's connectivity by re-reading the model's
  // ModuleList and the same yaml structure used in YoloV8DetectImpl.
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
      for (int f : s.from) {
        det_in.push_back(outs[f]);
        det_ch.push_back(d->ch[&f - s.from.data()]);
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
  model_pb.write_bytes_field(7, g.build_graph_bytes("yolov8"));
  model_pb.write_bytes_field(8, opset_id("", cfg.opset_version));

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(model_pb.bytes().data(),
          (std::streamsize)model_pb.bytes().size());
}

}  // namespace yolocpp::serialization
