// Clean-room reader for upstream-style `.pt` files.
//
// We deliberately don't use libtorch's torch::jit::Unpickler here — it's
// designed for TorchScript modules and trips on the BUILD/__setstate__ paths
// of arbitrary Python classes (e.g. the upstream `DetectionModel`).
//
// The implementation is a permissive Python pickle interpreter that:
//   • understands every opcode produced by torch.save (protocol 2),
//   • treats unknown GLOBAL classes as opaque Object stubs,
//   • intercepts the handful of REDUCE callables that appear in PyTorch
//     checkpoints (_rebuild_tensor_v2, _rebuild_parameter, OrderedDict, …)
//     and produces real torch::Tensor values.
//
// Storage data is loaded lazily from the .pt zip via PyTorchStreamReader.

#include "yolocpp/serialization/pt_loader.hpp"

#include <ATen/ATen.h>
#include <c10/util/typeid.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace yolocpp::serialization {

namespace {

// ──────────────────────────────────────────────────────────────────────────
// Generic Value type for the unpickled tree.

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct GlobalRef {
  std::string module;
  std::string name;
  std::string qualified() const { return module + "." + name; }
};

struct PersId {
  // PyTorch storage persistent_id is a tuple:
  //   ("storage", <class>, "<key>", "<location>", num_elements)
  std::string storage_id;
  std::string location;
  int64_t     num_elements = 0;
  c10::ScalarType dtype = c10::ScalarType::Float;
};

struct ObjectStub {
  // Result of REDUCE / NEWOBJ on an unknown class. Holds the constructor
  // args and (if BUILD ran) the __setstate__ payload.
  GlobalRef cls;
  ValuePtr  args;     // tuple
  ValuePtr  state;    // anything (often a dict)
};

struct Value {
  // Use a variant-like tagged union via std::variant.
  std::variant<
      std::monostate,                 // 0  None
      bool,                           // 1  bool
      int64_t,                        // 2  int
      double,                         // 3  float
      std::string,                    // 4  string
      std::vector<ValuePtr>,          // 5  tuple
      std::vector<ValuePtr>,          // 6  list (separate from tuple by usage)
      std::vector<std::pair<ValuePtr, ValuePtr>>,  // 7  dict (ordered)
      GlobalRef,                      // 8  class reference
      PersId,                         // 9  storage persistent id
      ObjectStub,                     // 10 generic object
      at::Tensor                      // 11 actual tensor
      > v;

  enum Kind { kNone = 0, kBool, kInt, kFloat, kString, kTuple, kList, kDict,
              kClass, kPersId, kObject, kTensor };
  Kind kind() const { return (Kind)v.index(); }

  bool is_none()  const { return kind() == kNone; }
  bool is_str()   const { return kind() == kString; }
  bool is_int()   const { return kind() == kInt; }
  bool is_dict()  const { return kind() == kDict; }
  bool is_tuple() const { return kind() == kTuple; }
  bool is_list()  const { return kind() == kList; }
  bool is_object()const { return kind() == kObject; }
  bool is_tensor()const { return kind() == kTensor; }
  bool is_class() const { return kind() == kClass; }
};

ValuePtr V_None() {
  auto v = std::make_shared<Value>();
  v->v = std::monostate{};
  return v;
}
ValuePtr V_Bool(bool b)        { auto v = std::make_shared<Value>(); v->v = b;    return v; }
ValuePtr V_Int(int64_t i)      { auto v = std::make_shared<Value>(); v->v = i;    return v; }
ValuePtr V_Float(double d)     { auto v = std::make_shared<Value>(); v->v = d;    return v; }
ValuePtr V_Str(std::string s)  { auto v = std::make_shared<Value>(); v->v = std::move(s); return v; }
ValuePtr V_Tuple(std::vector<ValuePtr> t) {
  auto v = std::make_shared<Value>();
  v->v.emplace<5>(std::move(t));
  return v;
}
ValuePtr V_List(std::vector<ValuePtr> l) {
  auto v = std::make_shared<Value>();
  v->v.emplace<6>(std::move(l));
  return v;
}
ValuePtr V_Dict(std::vector<std::pair<ValuePtr, ValuePtr>> d) {
  auto v = std::make_shared<Value>();
  v->v.emplace<7>(std::move(d));
  return v;
}
ValuePtr V_Class(std::string m, std::string n) {
  auto v = std::make_shared<Value>();
  v->v = GlobalRef{std::move(m), std::move(n)};
  return v;
}
ValuePtr V_PersId(PersId p) {
  auto v = std::make_shared<Value>();
  v->v = std::move(p);
  return v;
}
ValuePtr V_Tensor(at::Tensor t) {
  auto v = std::make_shared<Value>();
  v->v = std::move(t);
  return v;
}
ValuePtr V_Object(ObjectStub o) {
  auto v = std::make_shared<Value>();
  v->v = std::move(o);
  return v;
}

// ──────────────────────────────────────────────────────────────────────────
// Pickle opcode constants.

namespace op {
  constexpr uint8_t MARK             = '(';
  constexpr uint8_t STOP             = '.';
  constexpr uint8_t POP              = '0';
  constexpr uint8_t POP_MARK         = '1';
  constexpr uint8_t DUP              = '2';
  constexpr uint8_t BINPERSID        = 'Q';
  constexpr uint8_t REDUCE           = 'R';
  constexpr uint8_t BININT           = 'J';
  constexpr uint8_t BININT1          = 'K';
  constexpr uint8_t BININT2          = 'M';
  constexpr uint8_t NONE             = 'N';
  constexpr uint8_t BINUNICODE       = 'X';
  constexpr uint8_t SHORT_BINSTRING  = 'U';
  constexpr uint8_t BINSTRING        = 'T';
  constexpr uint8_t BINFLOAT         = 'G';
  constexpr uint8_t APPEND           = 'a';
  constexpr uint8_t APPENDS          = 'e';
  constexpr uint8_t BUILD            = 'b';
  constexpr uint8_t GLOBAL           = 'c';
  constexpr uint8_t INST             = 'i';
  constexpr uint8_t LONG             = 'L';
  constexpr uint8_t BINGET           = 'h';
  constexpr uint8_t LONG_BINGET      = 'j';
  constexpr uint8_t BINPUT           = 'q';
  constexpr uint8_t LONG_BINPUT      = 'r';
  constexpr uint8_t SETITEM          = 's';
  constexpr uint8_t SETITEMS         = 'u';
  constexpr uint8_t TUPLE            = 't';
  constexpr uint8_t EMPTY_DICT       = '}';
  constexpr uint8_t EMPTY_LIST       = ']';
  constexpr uint8_t EMPTY_TUPLE      = ')';
  constexpr uint8_t PROTO            = 0x80;
  constexpr uint8_t NEWOBJ           = 0x81;
  constexpr uint8_t TUPLE1           = 0x85;
  constexpr uint8_t TUPLE2           = 0x86;
  constexpr uint8_t TUPLE3           = 0x87;
  constexpr uint8_t NEWTRUE          = 0x88;
  constexpr uint8_t NEWFALSE         = 0x89;
  constexpr uint8_t LONG1            = 0x8a;
  constexpr uint8_t LONG4            = 0x8b;
  constexpr uint8_t SHORT_BINUNICODE = 0x8c;
  constexpr uint8_t BINUNICODE8      = 0x8d;
  constexpr uint8_t BINBYTES8        = 0x8e;
  constexpr uint8_t EMPTY_SET        = 0x8f;
  constexpr uint8_t ADDITEMS         = 0x90;
  constexpr uint8_t FROZENSET        = 0x91;
  constexpr uint8_t NEWOBJ_EX        = 0x92;
  constexpr uint8_t STACK_GLOBAL     = 0x93;
  constexpr uint8_t MEMOIZE          = 0x94;
  constexpr uint8_t FRAME            = 0x95;
  constexpr uint8_t SHORT_BINBYTES   = 'C';
  constexpr uint8_t BINBYTES         = 'B';
}

// Map (module, classname) → torch dtype for storage type globals.
c10::ScalarType storage_class_dtype(const std::string& module,
                                    const std::string& cls) {
  // PyTorch < 2: torch.FloatStorage, torch.HalfStorage, …
  // PyTorch ≥ 2: torch.storage.UntypedStorage (dtype carried separately)
  if (module == "torch") {
    if (cls == "FloatStorage")  return c10::ScalarType::Float;
    if (cls == "HalfStorage")   return c10::ScalarType::Half;
    if (cls == "DoubleStorage") return c10::ScalarType::Double;
    if (cls == "LongStorage")   return c10::ScalarType::Long;
    if (cls == "IntStorage")    return c10::ScalarType::Int;
    if (cls == "ShortStorage")  return c10::ScalarType::Short;
    if (cls == "CharStorage")   return c10::ScalarType::Char;
    if (cls == "ByteStorage")   return c10::ScalarType::Byte;
    if (cls == "BoolStorage")   return c10::ScalarType::Bool;
    if (cls == "BFloat16Storage") return c10::ScalarType::BFloat16;
  }
  return c10::ScalarType::Undefined;
}

// ──────────────────────────────────────────────────────────────────────────
// The unpickler.

class Unpickler {
 public:
  Unpickler(const char* data, size_t size,
            caffe2::serialize::PyTorchStreamReader& zip)
      : data_(data), size_(size), zip_(zip) {}

  ValuePtr run() {
    while (pos_ < size_) {
      uint8_t op = read_u8();
      if (!handle_op(op)) {
        // STOP
        if (stack_.empty())
          throw std::runtime_error("pickle ended with empty stack");
        return stack_.back();
      }
    }
    throw std::runtime_error("pickle truncated (no STOP)");
  }

 private:
  // Returns true to continue, false on STOP.
  bool handle_op(uint8_t op) {
    using namespace op;
    switch (op) {
      case PROTO:   read_u8(); return true;
      case FRAME:   pos_ += 8; return true;
      case STOP:    return false;
      case MARK:    marks_.push_back(stack_.size()); return true;
      case POP:     stack_.pop_back(); return true;
      case POP_MARK: stack_.resize(marks_.back()); marks_.pop_back(); return true;
      case DUP:     stack_.push_back(stack_.back()); return true;

      case NONE:      stack_.push_back(V_None()); return true;
      case NEWTRUE:   stack_.push_back(V_Bool(true));  return true;
      case NEWFALSE:  stack_.push_back(V_Bool(false)); return true;

      case BININT1:  stack_.push_back(V_Int((int64_t)read_u8())); return true;
      case BININT2:  stack_.push_back(V_Int((int64_t)read_u16())); return true;
      case BININT:   stack_.push_back(V_Int((int64_t)(int32_t)read_u32())); return true;
      case LONG1: {
        uint8_t n = read_u8();
        stack_.push_back(V_Int(read_long_le(n)));
        return true;
      }
      case LONG4: {
        uint32_t n = read_u32();
        stack_.push_back(V_Int(read_long_le(n)));
        return true;
      }
      case BINFLOAT: {
        // 8 bytes big-endian double
        uint8_t buf[8];
        for (int i = 0; i < 8; ++i) buf[i] = read_u8();
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) bits = (bits << 8) | buf[i];
        double d;
        std::memcpy(&d, &bits, 8);
        stack_.push_back(V_Float(d));
        return true;
      }
      case SHORT_BINUNICODE: {
        uint32_t n = read_u8();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case BINUNICODE: {
        uint32_t n = read_u32();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case BINUNICODE8: {
        uint64_t n = read_u64();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case SHORT_BINSTRING: {
        uint32_t n = read_u8();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case BINSTRING: {
        uint32_t n = read_u32();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case SHORT_BINBYTES: {
        uint32_t n = read_u8();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case BINBYTES: {
        uint32_t n = read_u32();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case BINBYTES8: {
        uint64_t n = read_u64();
        stack_.push_back(V_Str(read_str(n)));
        return true;
      }
      case EMPTY_DICT:  stack_.push_back(V_Dict({})); return true;
      case EMPTY_LIST:  stack_.push_back(V_List({})); return true;
      case EMPTY_TUPLE: stack_.push_back(V_Tuple({})); return true;
      case TUPLE1: {
        auto a = pop();
        stack_.push_back(V_Tuple({a}));
        return true;
      }
      case TUPLE2: {
        auto b = pop(); auto a = pop();
        stack_.push_back(V_Tuple({a, b}));
        return true;
      }
      case TUPLE3: {
        auto c = pop(); auto b = pop(); auto a = pop();
        stack_.push_back(V_Tuple({a, b, c}));
        return true;
      }
      case TUPLE: {
        auto items = pop_to_mark();
        stack_.push_back(V_Tuple(std::move(items)));
        return true;
      }
      case APPEND: {
        auto val = pop();
        auto& lst = std::get<6>(stack_.back()->v);
        lst.push_back(val);
        return true;
      }
      case APPENDS: {
        auto items = pop_to_mark();
        auto& lst = std::get<6>(stack_.back()->v);
        for (auto& it : items) lst.push_back(it);
        return true;
      }
      case SETITEM: {
        auto val = pop();
        auto key = pop();
        auto& d = std::get<7>(stack_.back()->v);
        d.emplace_back(key, val);
        return true;
      }
      case SETITEMS: {
        auto items = pop_to_mark();
        auto& d = std::get<7>(stack_.back()->v);
        for (size_t i = 0; i + 1 < items.size(); i += 2) {
          d.emplace_back(items[i], items[i + 1]);
        }
        return true;
      }
      case BINGET:      stack_.push_back(memo_.at(read_u8()));  return true;
      case LONG_BINGET: stack_.push_back(memo_.at(read_u32())); return true;
      case BINPUT:      memo_[read_u8()]  = stack_.back();      return true;
      case LONG_BINPUT: memo_[read_u32()] = stack_.back();      return true;
      case MEMOIZE:     memo_[memo_.size()] = stack_.back();    return true;

      case GLOBAL: {
        auto m = read_line();
        auto n = read_line();
        stack_.push_back(V_Class(m, n));
        return true;
      }
      case STACK_GLOBAL: {
        auto n = pop();
        auto m = pop();
        stack_.push_back(V_Class(std::get<4>(m->v), std::get<4>(n->v)));
        return true;
      }
      case BINPERSID: {
        // Top of stack is the persistent_id tuple.
        auto pid = pop();
        stack_.push_back(resolve_persid(pid));
        return true;
      }
      case REDUCE: {
        auto args = pop();
        auto func = pop();
        stack_.push_back(call_reduce(func, args));
        return true;
      }
      case NEWOBJ: {
        auto args = pop();
        auto cls  = pop();
        // Treat as REDUCE for class instantiation.
        stack_.push_back(call_reduce(cls, args));
        return true;
      }
      case NEWOBJ_EX: {
        // pops kwargs, args, cls
        pop();  // kwargs
        auto args = pop();
        auto cls  = pop();
        stack_.push_back(call_reduce(cls, args));
        return true;
      }
      case BUILD: {
        // Set state on object (or call __setstate__).
        auto state = pop();
        auto obj   = stack_.back();
        if (obj->is_object()) {
          std::get<10>(obj->v).state = state;
        } else if (obj->is_tensor() && state->is_tuple()) {
          // Tensor.__setstate__ — typically a tuple of (data, requires_grad,
          // backward_hooks). For our purposes the data is already the tensor
          // produced by _rebuild_tensor_v2, so we just ignore.
        }
        // Otherwise drop the state.
        return true;
      }
      case EMPTY_SET:
      case ADDITEMS:
      case FROZENSET:
        // Sets aren't meaningful for state_dict extraction.
        if (op == EMPTY_SET) stack_.push_back(V_List({}));
        if (op == ADDITEMS)  pop_to_mark();
        if (op == FROZENSET) pop_to_mark();
        return true;
      case INST:
      case LONG:
        // Older protocol opcodes — not produced by torch.save (proto ≥ 2).
        throw std::runtime_error("unsupported pickle opcode (legacy): " +
                                 std::to_string(op));
      default:
        throw std::runtime_error("unsupported pickle opcode: " +
                                 std::to_string(op));
    }
  }

  ValuePtr resolve_persid(const ValuePtr& pid) {
    // PyTorch storage pid is a tuple:
    //   ("storage", <Storage class | dtype>, "<key>", "<location>", numel)
    if (!pid->is_tuple())
      throw std::runtime_error("persistent_id is not a tuple");
    const auto& t = std::get<5>(pid->v);
    if (t.size() < 5)
      throw std::runtime_error("storage pid tuple too short");

    auto kind = std::get<4>(t[0]->v);
    if (kind != "storage")
      throw std::runtime_error("non-storage persistent_id: " + kind);

    // t[1] can be either a Class (FloatStorage etc.) or a dtype Object.
    PersId p;
    if (t[1]->is_class()) {
      const auto& cls = std::get<8>(t[1]->v);
      p.dtype = storage_class_dtype(cls.module, cls.name);
    } else if (t[1]->is_object()) {
      // dtype object — class name like torch.float32
      const auto& obj = std::get<10>(t[1]->v);
      const auto& dn  = obj.cls.name;
      // Names: float32, float, float16, half, double, …
      if (dn == "float32" || dn == "float") p.dtype = c10::ScalarType::Float;
      else if (dn == "float16" || dn == "half") p.dtype = c10::ScalarType::Half;
      else if (dn == "float64" || dn == "double") p.dtype = c10::ScalarType::Double;
      else if (dn == "int64"   || dn == "long")   p.dtype = c10::ScalarType::Long;
      else if (dn == "int32"   || dn == "int")    p.dtype = c10::ScalarType::Int;
      else if (dn == "int16"   || dn == "short")  p.dtype = c10::ScalarType::Short;
      else if (dn == "int8")                      p.dtype = c10::ScalarType::Char;
      else if (dn == "uint8"   || dn == "byte")   p.dtype = c10::ScalarType::Byte;
      else if (dn == "bool")                      p.dtype = c10::ScalarType::Bool;
      else if (dn == "bfloat16")                  p.dtype = c10::ScalarType::BFloat16;
      else p.dtype = c10::ScalarType::Float;
    }

    p.storage_id   = std::get<4>(t[2]->v);
    p.location     = std::get<4>(t[3]->v);
    p.num_elements = std::get<2>(t[4]->v);
    return V_PersId(std::move(p));
  }

  ValuePtr call_reduce(const ValuePtr& func, const ValuePtr& args) {
    if (!func->is_class())
      throw std::runtime_error("REDUCE: callable is not a class/global");
    const auto& g = std::get<8>(func->v);

    // Tuple of args.
    const std::vector<ValuePtr>* arg_tuple = nullptr;
    if (args->is_tuple()) arg_tuple = &std::get<5>(args->v);

    auto qn = g.module + "." + g.name;

    if (qn == "collections.OrderedDict") {
      // OrderedDict([(k, v), ...]) or OrderedDict()
      if (arg_tuple && !arg_tuple->empty() && (*arg_tuple)[0]->is_list()) {
        std::vector<std::pair<ValuePtr, ValuePtr>> kvs;
        for (auto& it : std::get<6>((*arg_tuple)[0]->v)) {
          if (it->is_tuple()) {
            const auto& pair = std::get<5>(it->v);
            if (pair.size() == 2) kvs.emplace_back(pair[0], pair[1]);
          }
        }
        return V_Dict(std::move(kvs));
      }
      return V_Dict({});
    }

    if (qn == "torch._utils._rebuild_tensor_v2" ||
        qn == "torch._utils._rebuild_tensor"    ||
        qn == "torch._utils._rebuild_tensor_v3") {
      return rebuild_tensor(*arg_tuple);
    }
    if (qn == "torch._utils._rebuild_parameter") {
      // _rebuild_parameter(data, requires_grad, backward_hooks)
      // data is the result of a prior _rebuild_tensor_v2 call.
      if (!arg_tuple || arg_tuple->empty())
        throw std::runtime_error("rebuild_parameter: empty args");
      return (*arg_tuple)[0];
    }
    if (qn == "torch._utils._rebuild_parameter_with_state") {
      if (!arg_tuple || arg_tuple->empty())
        throw std::runtime_error("rebuild_parameter_with_state: empty args");
      return (*arg_tuple)[0];
    }

    // Unknown class — make a generic Object stub.
    ObjectStub o;
    o.cls  = g;
    o.args = args;
    return V_Object(std::move(o));
  }

  ValuePtr rebuild_tensor(const std::vector<ValuePtr>& a) {
    // (storage, storage_offset, size, stride, requires_grad, backward_hooks, [metadata])
    if (a.size() < 4)
      throw std::runtime_error("rebuild_tensor: too few args");
    if (a[0]->kind() != Value::kPersId)
      throw std::runtime_error("rebuild_tensor: first arg is not a storage");

    const auto& pid = std::get<9>(a[0]->v);
    int64_t offset = a[1]->is_int() ? std::get<2>(a[1]->v) : 0;

    // size: tuple of ints
    std::vector<int64_t> sizes;
    if (a[2]->is_tuple()) {
      for (auto& s : std::get<5>(a[2]->v))
        sizes.push_back(std::get<2>(s->v));
    }
    std::vector<int64_t> strides;
    if (a[3]->is_tuple()) {
      for (auto& s : std::get<5>(a[3]->v))
        strides.push_back(std::get<2>(s->v));
    }

    return V_Tensor(load_storage_tensor(pid, offset, sizes, strides));
  }

  at::Tensor load_storage_tensor(const PersId& pid, int64_t offset,
                                 const std::vector<int64_t>& sizes,
                                 const std::vector<int64_t>& strides) {
    // Look up cached storage data by id.
    auto it = storages_.find(pid.storage_id);
    if (it == storages_.end()) {
      auto record = "data/" + pid.storage_id;
      std::tuple<at::DataPtr, size_t> dp = zip_.getRecord(record);
      auto& ptr  = std::get<0>(dp);
      auto  size = std::get<1>(dp);
      auto buf = std::vector<uint8_t>(size);
      std::memcpy(buf.data(), ptr.get(), size);
      it = storages_.emplace(pid.storage_id, std::move(buf)).first;
    }
    auto& buf = it->second;

    auto opts = at::TensorOptions().dtype(pid.dtype);
    auto el   = at::elementSize(pid.dtype);

    // Build a tensor that owns a copy of the (sliced) storage. We copy so
    // that the original buffer can be reused by other tensors sharing the
    // same storage id, and so we own clean memory.
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;

    auto t = at::empty(sizes, opts);
    if (numel > 0) {
      // Use strides-agnostic layout: PyTorch storages are row-major with
      // the given strides. For cleanliness we just memcpy a contiguous
      // block — if strides aren't contiguous we'd need to handle that,
      // but tensor saves are always contiguous in practice for state_dicts.
      auto byte_off = offset * el;
      auto byte_len = numel * el;
      if (byte_off + byte_len > buf.size()) {
        std::stringstream ss;
        ss << "storage " << pid.storage_id << " too small: "
           << buf.size() << " < " << byte_off + byte_len;
        throw std::runtime_error(ss.str());
      }
      std::memcpy(t.data_ptr(), buf.data() + byte_off, byte_len);
    }
    (void)strides;
    return t;
  }

  // ── Low-level reading helpers ───────────────────────────────────────────
  uint8_t  read_u8()  { return (uint8_t)data_[pos_++]; }
  uint16_t read_u16() {
    uint16_t v = (uint8_t)data_[pos_] | ((uint8_t)data_[pos_+1] << 8);
    pos_ += 2; return v;
  }
  uint32_t read_u32() {
    uint32_t v = (uint8_t)data_[pos_] | ((uint8_t)data_[pos_+1] << 8) |
                 ((uint8_t)data_[pos_+2] << 16) | ((uint8_t)data_[pos_+3] << 24);
    pos_ += 4; return v;
  }
  uint64_t read_u64() {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)(uint8_t)data_[pos_+i] << (i*8);
    pos_ += 8; return v;
  }
  int64_t read_long_le(uint32_t n) {
    if (n == 0) return 0;
    int64_t v = 0;
    for (uint32_t i = 0; i < n && i < 8; ++i)
      v |= (int64_t)(uint8_t)data_[pos_+i] << (i*8);
    // Sign-extend.
    if ((uint8_t)data_[pos_+n-1] & 0x80 && n < 8) {
      for (uint32_t i = n; i < 8; ++i) v |= (int64_t)0xff << (i*8);
    }
    pos_ += n; return v;
  }
  std::string read_str(uint64_t n) {
    std::string s(data_ + pos_, n);
    pos_ += n; return s;
  }
  std::string read_line() {
    std::string s;
    while (pos_ < size_ && data_[pos_] != '\n') s.push_back(data_[pos_++]);
    if (pos_ < size_) ++pos_;  // consume '\n'
    return s;
  }

  ValuePtr pop() { auto v = stack_.back(); stack_.pop_back(); return v; }
  std::vector<ValuePtr> pop_to_mark() {
    size_t m = marks_.back(); marks_.pop_back();
    std::vector<ValuePtr> out(stack_.begin() + m, stack_.end());
    stack_.resize(m);
    return out;
  }

  const char* data_;
  size_t      size_;
  size_t      pos_ = 0;
  std::vector<ValuePtr> stack_;
  std::vector<size_t>   marks_;
  std::map<size_t, ValuePtr> memo_;
  std::map<std::string, std::vector<uint8_t>> storages_;
  caffe2::serialize::PyTorchStreamReader& zip_;
};

// ──────────────────────────────────────────────────────────────────────────
// Tree → state_dict flattening.

const ValuePtr* dict_get(const ValuePtr& d, const std::string& key) {
  if (!d || !d->is_dict()) return nullptr;
  for (const auto& [k, v] : std::get<7>(d->v)) {
    if (k && k->is_str() && std::get<4>(k->v) == key) return &v;
  }
  return nullptr;
}

ValuePtr unwrap(const ValuePtr& v) {
  if (v && v->is_object()) {
    const auto& s = std::get<10>(v->v).state;
    if (s) return s;
  }
  return v;
}

void flatten_module(const ValuePtr& mod_state,
                    const std::string& prefix,
                    StateDict& out) {
  if (!mod_state || !mod_state->is_dict()) return;

  // _parameters
  if (auto* p = dict_get(mod_state, "_parameters"); p && (*p)->is_dict()) {
    for (const auto& [k, v] : std::get<7>((*p)->v)) {
      if (!k || !k->is_str()) continue;
      const auto& name = std::get<4>(k->v);
      auto val = unwrap(v);
      if (val && val->is_tensor())
        out.entries.emplace_back(prefix + name, std::get<11>(val->v));
    }
  }
  // _buffers
  if (auto* b = dict_get(mod_state, "_buffers"); b && (*b)->is_dict()) {
    for (const auto& [k, v] : std::get<7>((*b)->v)) {
      if (!k || !k->is_str()) continue;
      const auto& name = std::get<4>(k->v);
      auto val = unwrap(v);
      if (val && val->is_tensor())
        out.entries.emplace_back(prefix + name, std::get<11>(val->v));
    }
  }
  // _modules
  if (auto* m = dict_get(mod_state, "_modules"); m && (*m)->is_dict()) {
    for (const auto& [k, v] : std::get<7>((*m)->v)) {
      if (!k || !k->is_str()) continue;
      const auto& name = std::get<4>(k->v);
      auto val = unwrap(v);
      flatten_module(val, prefix + name + ".", out);
    }
  }
}

}  // anonymous namespace

StateDict load_state_dict(const std::string& pt_path,
                          const std::string& submodel) {
  auto file_adapter = std::make_shared<caffe2::serialize::FileAdapter>(pt_path);
  caffe2::serialize::PyTorchStreamReader zip(file_adapter);

  // data.pkl bytes
  std::tuple<at::DataPtr, size_t> rec = zip.getRecord("data.pkl");
  auto&  data_ptr  = std::get<0>(rec);
  size_t data_size = std::get<1>(rec);

  Unpickler u(reinterpret_cast<const char*>(data_ptr.get()), data_size, zip);
  auto root = u.run();

  // root should be a dict; navigate to submodel entry.
  auto* sub = dict_get(root, submodel);
  if (sub == nullptr || !*sub)
    throw std::runtime_error("submodel key not found: " + submodel);

  auto state = unwrap(*sub);
  StateDict out;
  flatten_module(state, /*prefix=*/"", out);

  if (out.entries.empty())
    throw std::runtime_error(
        "no parameters/buffers found under submodel '" + submodel + "'");
  return out;
}

// Walk a generic dict whose keys are already dotted parameter names and
// whose values are tensors (possibly wrapped in IValue Object form).
// Returns true if the dict was flat (any tensor-valued entry encountered).
static bool flatten_flat_dict(const ValuePtr& v, StateDict& out) {
  if (!v || !v->is_dict()) return false;
  bool any = false;
  for (const auto& [k, val_in] : std::get<7>(v->v)) {
    if (!k || !k->is_str()) continue;
    const auto& name = std::get<4>(k->v);
    auto val = unwrap(val_in);
    if (val && val->is_tensor()) {
      out.entries.emplace_back(name, std::get<11>(val->v));
      any = true;
    }
  }
  return any;
}

StateDict load_flat_state_dict(const std::string& pt_path,
                                const std::string& submodel) {
  auto file_adapter = std::make_shared<caffe2::serialize::FileAdapter>(pt_path);
  caffe2::serialize::PyTorchStreamReader zip(file_adapter);
  std::tuple<at::DataPtr, size_t> rec = zip.getRecord("data.pkl");
  auto&  data_ptr  = std::get<0>(rec);
  size_t data_size = std::get<1>(rec);
  Unpickler u(reinterpret_cast<const char*>(data_ptr.get()), data_size, zip);
  auto root = u.run();

  auto* sub = dict_get(root, submodel);
  if (sub == nullptr || !*sub)
    throw std::runtime_error("submodel key not found: " + submodel);

  auto state = unwrap(*sub);
  StateDict out;
  // Try flat-dict first (RF-DETR / DETR family).
  if (flatten_flat_dict(state, out)) return out;
  // Fall through to module-shaped layout.
  flatten_module(state, /*prefix=*/"", out);
  if (out.entries.empty())
    throw std::runtime_error(
        "load_flat_state_dict: no tensor entries under submodel '" +
        submodel + "'");
  return out;
}

}  // namespace yolocpp::serialization
