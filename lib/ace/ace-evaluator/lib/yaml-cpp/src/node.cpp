#include "yaml-cpp/node/node.h"
#include "nodebuilder.h"
#include "nodeevents.h"

namespace YAML_PACE {
    Node Clone(const Node &node) {
        NodeEvents events(node);
        NodeBuilder builder;
        events.Emit(builder);
        return builder.Root();
    }
}  // namespace YAML
