#include "yolocpp/models/yolo10.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo10Impl::Yolo10Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo10Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO10 forward not implemented yet — see include/yolocpp/models/yolo10.hpp");
}

}  // namespace yolocpp::models
