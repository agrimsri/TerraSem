/**
 * @file voxel_grid.hpp
 * @brief Sparse voxel grid (C++ port of mapping/voxel_grid.py).
 *
 * Uses std::unordered_map<uint64_t, Voxel> for O(1) lookup.
 * See BUILD.md Phase 8.4.
 * TODO (Phase 8): implement.
 */
#pragma once

#include <cstdint>
#include <unordered_map>
#include <array>

namespace terrasem {

constexpr int NUM_CLASSES = 11;

struct Voxel {
    float log_odds = 0.0f;
    std::array<float, NUM_CLASSES> class_counts{};
    int32_t hit_count = 0;
    int32_t miss_count = 0;
    float z_min = 1e9f;
    float z_max = -1e9f;
};

class VoxelGrid {
public:
    explicit VoxelGrid(float voxel_size = 0.2f) : voxel_size_(voxel_size) {}

    // TODO (Phase 8): implement world_to_index, insert, lookup, update methods.

    size_t size() const { return grid_.size(); }
    float voxel_size() const { return voxel_size_; }

private:
    float voxel_size_;
    std::unordered_map<uint64_t, Voxel> grid_;
};

}  // namespace terrasem
