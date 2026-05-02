# yolocpp examples

Self-contained programs that demonstrate the public C++ API
(`#include <yolocpp/api.hpp>`). All link against `yolocpp_core`
the same way the `yolocpp` CLI does.

| file                          | what it shows                                          |
|-------------------------------|--------------------------------------------------------|
| `predict_image.cpp`           | single-image predict; print per-det bbox / conf / cls  |
| `predict_directory.cpp`       | fan out over a directory of images                     |
| `predict_video.cpp`           | frame loop on a video file / RTSP URL / webcam index   |
| `train_finetune.cpp`          | fine-tune from pretrained `.pt`, auto-export to ONNX   |
| `export_to_onnx.cpp`          | `.pt` → `.onnx` or `.trt` (fp32 / fp16)                |
| `benchmark_model.cpp`         | latency / throughput across PT + TRT backends          |
| `end_to_end.cpp`              | chained `train → val → predict` in one fluent call     |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target predict_image
./build/examples/predict_image data/yolo11s.pt data/bus.jpg
```

To skip the example targets:

```bash
cmake -S . -B build -DYOLOCPP_BUILD_EXAMPLES=OFF
```

## Public API in two lines

```cpp
#include <yolocpp/api.hpp>

yolocpp::YOLO model("yolo11s.pt");
auto dets = model.to("auto").predict({.source = "bus.jpg"});
```

Every method takes a designated-initialiser Args struct so you only
specify the fields you care about. See `include/yolocpp/api.hpp` for
the full surface.
