#pragma once
//
// Ultralytics-compatible dataset YAML parser.
//
//   path:  ../datasets/coco8       # dataset root (relative to YAML or abs)
//   train: ./images/train          # relative to `path`
//   val:   ./images/val            # relative to `path`
//   names:
//     0: person
//     1: bicycle
//     ...
//
// Resolved fields are filesystem-absolute. `names` is index-ordered.

#include <string>
#include <vector>

namespace yolocpp::cli {

struct DataYaml {
  std::string root;                     // absolute path to dataset root
  std::string train_dir;                // absolute path or empty
  std::string val_dir;                  // absolute path or empty
  std::string test_dir;                 // absolute path or empty
  std::vector<std::string> names;       // class names, index-ordered
  std::string download_url;             // optional `download:` URL (Ultralytics)
  std::string yaml_path;                // absolute path to the YAML itself
};

// Parse a YAML file written in the Ultralytics dataset format. Throws
// std::runtime_error on missing file / malformed YAML / inconsistent keys.
DataYaml parse_data_yaml(const std::string& yaml_path);

}  // namespace yolocpp::cli
