#ifndef VOXEL_EXTRA_CLASS_REGISTRY_H
#define VOXEL_EXTRA_CLASS_REGISTRY_H

namespace zylann::voxel {

// Hook for classes contributed by code outside this repository.
//
// Host projects that add their own sources to this module's build define this
// function and register their classes in it. This repository only declares it,
// so it does not need to know their names.
//
// extra_class_registry.cpp holds a no-op definition for builds without a host
// project. Set the voxel_extra_classes build option to leave that file out when
// a host project supplies its own definition.
void register_extra_classes();

} // namespace zylann::voxel

#endif // VOXEL_EXTRA_CLASS_REGISTRY_H
