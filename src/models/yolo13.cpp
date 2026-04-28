#include "yolocpp/models/yolo13.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo13Impl::Yolo13Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo13Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO13 forward not implemented yet — see include/yolocpp/models/yolo13.hpp");
}

}  // namespace yolocpp::models
