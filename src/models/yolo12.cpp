#include "yolocpp/models/yolo12.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo12Impl::Yolo12Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo12Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO12 forward not implemented yet — see include/yolocpp/models/yolo12.hpp");
}

}  // namespace yolocpp::models
