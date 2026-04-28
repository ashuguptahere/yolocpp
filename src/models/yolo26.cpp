#include "yolocpp/models/yolo26.hpp"

#include <stdexcept>

namespace yolocpp::models {

Yolo26Impl::Yolo26Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo26Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "YOLO26 forward not implemented yet — see include/yolocpp/models/yolo26.hpp");
}

}  // namespace yolocpp::models
