#include "yolocpp/models/yolo7.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo7Impl::Yolo7Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo7Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO7 forward not implemented yet — see include/yolocpp/models/yolo7.hpp");
}

}  // namespace yolocpp::models
