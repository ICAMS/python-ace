#include <fstream>
#include <iostream>
#include <vector>

#include "yaml-cpp/eventhandler.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

class NullEventHandler : public YAML::EventHandler {
 public:
  void OnDocumentStart(const YAML_PACE::Mark&) override {}
  void OnDocumentEnd() override {}

  void OnNull(const YAML_PACE::Mark&, YAML_PACE::anchor_t) override {}
  void OnAlias(const YAML_PACE::Mark&, YAML_PACE::anchor_t) override {}
  void OnScalar(const YAML_PACE::Mark&, const std::string&, YAML_PACE::anchor_t,
                const std::string&) override {}

  void OnSequenceStart(const YAML_PACE::Mark&, const std::string&, YAML_PACE::anchor_t,
                       YAML_PACE::EmitterStyle::value) override {}
  void OnSequenceEnd() override {}

  void OnMapStart(const YAML_PACE::Mark&, const std::string&, YAML_PACE::anchor_t,
                  YAML_PACE::EmitterStyle::value) override {}
  void OnMapEnd() override {}
};

void parse(std::istream& input) {
  try {
    YAML_PACE::Node doc = YAML_PACE::Load(input);
    std::cout << doc << "\n";
  } catch (const YAML_PACE::Exception& e) {
    std::cerr << e.what() << "\n";
  }
}

int main(int argc, char** argv) {
  if (argc > 1) {
    std::ifstream fin;
    fin.open(argv[1]);
    parse(fin);
  } else {
    parse(std::cin);
  }

  return 0;
}
