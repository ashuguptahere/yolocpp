// Hand-written PyTorch-compatible .pt writer.
//
// We emit the minimum subset of pickle ops needed to round-trip through
// our own parser (and, as a bonus, through real PyTorch).
//
// Wire layout matches what torch.save produces:
//   • Zip archive (caffe2::serialize::PyTorchStreamWriter)
//   • record "data.pkl" — a single pickle producing a GenericDict
//     {"model": Object(class=DetectionModel, state=GenericDict{
//        "_parameters": OrderedDict[(name, Tensor), ...],
//        "_buffers":    OrderedDict[],
//        "_modules":    OrderedDict[],
//     })}
//   • records "data/0", "data/1", ... — raw tensor storage bytes
//
// The pt_loader unwraps this exactly: ckpt["model"] → state → _parameters.
// Listing every tensor under _parameters (rather than nested per-module)
// works because our load_from_state_dict matches by full dotted name and
// our _parameters keys ARE the full dotted names (e.g. "model.0.conv.weight").

#include "yolocpp/serialization/pt_save.hpp"

#include <caffe2/serialize/inline_container.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolocpp::serialization {

namespace {

// ── Pickle wire-format writer ─────────────────────────────────────────────
class PickleWriter {
 public:
  std::string& bytes() { return data_; }

  // Opcodes we use.
  void op_proto(uint8_t v)         { put_u8(0x80); put_u8(v); }
  void op_frame(uint64_t n)        { put_u8(0x95); put_u64(n); }
  void op_stop()                   { put_u8('.'); }
  void op_empty_dict()             { put_u8('}'); }
  void op_empty_list()             { put_u8(']'); }
  void op_empty_tuple()            { put_u8(')'); }
  void op_mark()                   { put_u8('('); }
  void op_setitems()               { put_u8('u'); }
  void op_appends()                { put_u8('e'); }
  void op_tuple1()                 { put_u8(0x85); }
  void op_tuple2()                 { put_u8(0x86); }
  void op_tuple3()                 { put_u8(0x87); }
  void op_tuple()                  { put_u8('t'); }
  void op_reduce()                 { put_u8('R'); }
  void op_build()                  { put_u8('b'); }
  void op_newtrue()                { put_u8(0x88); }
  void op_newfalse()               { put_u8(0x89); }
  void op_none()                   { put_u8('N'); }
  void op_binpersid()              { put_u8('Q'); }
  void op_memoize()                { put_u8(0x94); }

  void op_short_binunicode(const std::string& s) {
    if (s.size() < 256) {
      put_u8(0x8c); put_u8((uint8_t)s.size()); data_.append(s);
    } else {
      put_u8('X'); put_u32((uint32_t)s.size()); data_.append(s);
    }
  }
  void op_bininteger(int64_t v) {
    if (v >= 0 && v < 256) { put_u8('K'); put_u8((uint8_t)v); }
    else if (v >= -2147483648LL && v <= 2147483647LL) {
      put_u8('J'); put_u32((uint32_t)(int32_t)v);
    } else {
      // LONG1 with up to 8 bytes
      put_u8(0x8a); put_u8(8);
      for (int i = 0; i < 8; ++i) put_u8((uint8_t)((v >> (i * 8)) & 0xff));
    }
  }
  void op_global(const std::string& mod, const std::string& name) {
    // STACK_GLOBAL(0x93) + two SHORT_BINUNICODE strings on the stack.
    op_short_binunicode(mod);
    op_short_binunicode(name);
    put_u8(0x93);
  }

  void put_u8(uint8_t v)   { data_.push_back((char)v); }
  void put_u16(uint16_t v) {
    put_u8((uint8_t)(v & 0xff)); put_u8((uint8_t)((v >> 8) & 0xff));
  }
  void put_u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) put_u8((uint8_t)((v >> (i * 8)) & 0xff));
  }
  void put_u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) put_u8((uint8_t)((v >> (i * 8)) & 0xff));
  }

 private:
  std::string data_;
};

const char* dtype_storage_class(c10::ScalarType st) {
  switch (st) {
    case c10::ScalarType::Float:  return "FloatStorage";
    case c10::ScalarType::Half:   return "HalfStorage";
    case c10::ScalarType::Double: return "DoubleStorage";
    case c10::ScalarType::Long:   return "LongStorage";
    case c10::ScalarType::Int:    return "IntStorage";
    case c10::ScalarType::Byte:   return "ByteStorage";
    case c10::ScalarType::Char:   return "CharStorage";
    case c10::ScalarType::Bool:   return "BoolStorage";
    default: return nullptr;
  }
}

// Build the data.pkl bytes for a flat state_dict.
// Stack-machine rationale (each line ends with stack notation):
//
//   PROTO 2
//   FRAME ...
//   EMPTY_DICT                                       [ckpt_dict]
//   MEMOIZE
//   MARK                                             [ckpt_dict, MARK]
//     short_binunicode "model"                        [ ..., "model"]
//     <Object construction for the model:>
//       GLOBAL ultralytics.nn.tasks DetectionModel    [ ..., "model", cls]
//       MEMOIZE
//       EMPTY_TUPLE                                   [ ..., cls, ()]
//       NEWOBJ        # cls(*())   creates empty obj  [ ..., obj]
//       MEMOIZE
//       <state dict for the object:>
//         EMPTY_DICT                                  [ ..., obj, state]
//         MEMOIZE
//         MARK                                        [ ..., obj, state, M]
//           short_binunicode "_parameters"
//           <OrderedDict([(name, tensor), ...]):>
//             GLOBAL collections OrderedDict
//             MEMOIZE
//             EMPTY_LIST                              # the list of pairs
//             MEMOIZE
//             MARK
//               for each (name, t):
//                 short_binunicode name
//                 <tensor t emission>                 # tuple (name, tensor)
//                 TUPLE2
//             APPENDS                                 # list now populated
//             TUPLE1                                  # (list,)
//             REDUCE                                  # OrderedDict(list)
//           short_binunicode "_buffers"
//             <empty OrderedDict>
//           short_binunicode "_modules"
//             <empty OrderedDict>
//         SETITEMS                                    # state dict populated
//       BUILD          # obj.__setstate__(state)      [ ..., model_obj]
//   SETITEMS                                          [ckpt_dict]
//   STOP
//
// Tensor emission (one per (name, tensor) pair):
//   GLOBAL torch._utils _rebuild_tensor_v2
//   MEMOIZE
//   MARK
//     <storage:>
//       BINPERSID with persistent_id =
//         ("storage", FloatStorage_class, "<id>", "cpu", numel)
//     BININT 0       # storage_offset
//     <size tuple>
//     <stride tuple>
//     NEWFALSE       # requires_grad
//     EMPTY_DICT     # backward_hooks (or None)
//   TUPLE
//   REDUCE
std::string build_data_pkl(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  PickleWriter w;
  w.op_proto(2);
  w.op_frame(0);  // bytes-after; the parser tolerates 0

  auto emit_storage_persid = [&](size_t id, c10::ScalarType st, int64_t numel) {
    // Persistent ID is a tuple-on-the-stack; push fields then BINPERSID.
    w.op_mark();
    w.op_short_binunicode("storage");
    // Storage class as a torch.<TypeStorage> GLOBAL.
    auto cls_name = dtype_storage_class(st);
    if (!cls_name) throw std::runtime_error("unsupported dtype for save");
    w.op_global("torch", cls_name);
    w.op_short_binunicode(std::to_string(id));
    w.op_short_binunicode("cpu");
    w.op_bininteger(numel);
    w.op_tuple();
    w.op_binpersid();
  };

  auto emit_int_tuple = [&](const std::vector<int64_t>& xs) {
    w.op_mark();
    for (auto v : xs) w.op_bininteger(v);
    w.op_tuple();
  };

  auto emit_tensor = [&](size_t storage_id, const at::Tensor& t) {
    auto sizes   = t.sizes().vec();
    auto strides = t.strides().vec();
    int64_t numel = t.numel();
    // GLOBAL torch._utils _rebuild_tensor_v2
    w.op_global("torch._utils", "_rebuild_tensor_v2");
    w.op_mark();
    emit_storage_persid(storage_id, t.scalar_type(), numel);
    w.op_bininteger(0);                 // storage_offset
    emit_int_tuple(sizes);
    emit_int_tuple(strides);
    w.op_newfalse();                    // requires_grad
    w.op_empty_dict();                  // backward_hooks
    w.op_tuple();
    w.op_reduce();
  };

  // Outer ckpt dict.
  w.op_empty_dict();
  w.op_mark();
  w.op_short_binunicode("model");

  // Object stub: a class reference + NEWOBJ + BUILD. The class name
  // emitted here ("ultralytics.nn.tasks.DetectionModel") is a pickle
  // wire-format token, not a branding statement — downstream tooling
  // that consumes upstream-format `.pt` files looks for this exact
  // string. Renaming would break interop with every existing reader,
  // including our own (pt_loader treats the GLOBAL as opaque and
  // extracts only the tensor data, but other readers don't).
  w.op_global("ultralytics.nn.tasks", "DetectionModel");
  w.op_empty_tuple();
  w.put_u8(0x81);  // NEWOBJ — calls cls(*args=()) — opcode for our parser

  // State dict for the object.
  w.op_empty_dict();
  w.op_mark();

  // _parameters: OrderedDict([(name, tensor), ...])
  w.op_short_binunicode("_parameters");
  w.op_global("collections", "OrderedDict");
  w.op_empty_list();
  w.op_mark();
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& [name, t] = entries[i];
    w.op_short_binunicode(name);
    emit_tensor(i, t);
    w.op_tuple2();
  }
  w.op_appends();
  // Wrap list in tuple, call OrderedDict on it.
  w.op_tuple1();
  w.op_reduce();

  // _buffers: empty OrderedDict
  w.op_short_binunicode("_buffers");
  w.op_global("collections", "OrderedDict");
  w.op_empty_list();
  w.op_tuple1();
  w.op_reduce();

  // _modules: empty OrderedDict
  w.op_short_binunicode("_modules");
  w.op_global("collections", "OrderedDict");
  w.op_empty_list();
  w.op_tuple1();
  w.op_reduce();

  // Close inner state dict.
  w.op_setitems();

  // Object BUILD with the state dict — our parser stores state on the obj.
  w.op_build();

  // Close outer ckpt_dict.
  w.op_setitems();
  w.op_stop();
  return w.bytes();
}

}  // anonymous namespace

void save_state_dict(
    const std::string& path,
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  caffe2::serialize::PyTorchStreamWriter writer(path);
  // 1) data.pkl
  auto pkl = build_data_pkl(entries);
  writer.writeRecord("data.pkl", pkl.data(), pkl.size());

  // 2) data/<id> for each tensor — raw little-endian contiguous bytes.
  for (size_t i = 0; i < entries.size(); ++i) {
    auto t   = entries[i].second.detach().to(torch::kCPU).contiguous();
    auto sz  = t.numel() * t.element_size();
    auto rec = "data/" + std::to_string(i);
    writer.writeRecord(rec, t.data_ptr(), sz);
  }
  writer.writeEndOfFile();
}

}  // namespace yolocpp::serialization
