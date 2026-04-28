#include "yolocpp/models/yolo6.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo6Impl::Yolo6Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo6Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO6 forward not implemented yet — see include/yolocpp/models/yolo6.hpp");
}

}  // namespace yolocpp::models
