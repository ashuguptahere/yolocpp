#include "yolocpp/models/yolo9.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo9Impl::Yolo9Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo9Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO9 forward not implemented yet — see include/yolocpp/models/yolo9.hpp");
}

}  // namespace yolocpp::models
