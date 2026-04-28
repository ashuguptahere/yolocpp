#include "yolocpp/models/yolo11.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo11Impl::Yolo11Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo11Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO11 forward not implemented yet — see include/yolocpp/models/yolo11.hpp");
}

}  // namespace yolocpp::models
