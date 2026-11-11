#include "extra_class_registry.h"

namespace zylann::voxel {

// No-op fallback, compiled only when the build has no host project supplying its
// own definition. See the voxel_extra_classes option in common.py.
void register_extra_classes() {}

} // namespace zylann::voxel
