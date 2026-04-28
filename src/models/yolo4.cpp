#include "yolocpp/models/yolo4.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo4Impl::Yolo4Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo4Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO4 forward not implemented yet — see include/yolocpp/models/yolo4.hpp");
}

}  // namespace yolocpp::models
