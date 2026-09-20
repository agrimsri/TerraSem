#ifndef TERRASEM_ROS_VOXEL_GRID_HPP_
#define TERRASEM_ROS_VOXEL_GRID_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace terrasem {

constexpr int NUM_CLASSES = 11;
constexpr float LOG_ODDS_MIN = -2.0f;
constexpr float LOG_ODDS_MAX = 3.5f;
constexpr float TAU_RANGE = 30.0f;

struct Voxel {
    float log_odds = 0.0f;
    std::array<float, NUM_CLASSES> alpha;

    Voxel() {
        alpha.fill(1.0f);  // Uniform Dirichlet prior alpha_0 = 1
    }

    void update_log_odds(float delta) {
        log_odds += delta;
        if (log_odds < LOG_ODDS_MIN) log_odds = LOG_ODDS_MIN;
        if (log_odds > LOG_ODDS_MAX) log_odds = LOG_ODDS_MAX;
    }

    void update_semantics(int class_id, float conf, float range) {
        if (class_id < 0 || class_id >= NUM_CLASSES) return;
        float weight = conf * std::exp(-range / TAU_RANGE);
        alpha[class_id] += weight;
    }

    float p_occ() const {
        return 1.0f / (1.0f + std::exp(-log_odds));
    }

    std::array<float, NUM_CLASSES> semantic_probs() const {
        float sum = 0.0f;
        for (float a : alpha) sum += a;
        std::array<float, NUM_CLASSES> probs;
        for (size_t i = 0; i < NUM_CLASSES; ++i) {
            probs[i] = alpha[i] / sum;
        }
        return probs;
    }

    int argmax_class() const {
        int best_cls = 0;
        float best_a = alpha[0];
        for (int i = 1; i < NUM_CLASSES; ++i) {
            if (alpha[i] > best_a) {
                best_a = alpha[i];
                best_cls = i;
            }
        }
        return best_cls;
    }

    float entropy() const {
        auto probs = semantic_probs();
        float h = 0.0f;
        for (float p : probs) {
            if (p > 1e-7f) {
                h -= p * std::log(p);
            }
        }
        constexpr float max_h = 2.397895f; // ln(11)
        return std::min(std::max(h / max_h, 0.0f), 1.0f);
    }
};

struct VoxelCoord {
    int64_t x, y, z;

    bool operator==(const VoxelCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelCoordHash {
    std::size_t operator()(const VoxelCoord& c) const {
        std::size_t h1 = std::hash<int64_t>{}(c.x);
        std::size_t h2 = std::hash<int64_t>{}(c.y);
        std::size_t h3 = std::hash<int64_t>{}(c.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class SparseVoxelGrid {
public:
    explicit SparseVoxelGrid(float voxel_size = 0.2f) : voxel_size_(voxel_size) {}

    VoxelCoord world_to_voxel(float x, float y, float z) const {
        return VoxelCoord{
            static_cast<int64_t>(std::floor(x / voxel_size_)),
            static_cast<int64_t>(std::floor(y / voxel_size_)),
            static_cast<int64_t>(std::floor(z / voxel_size_))
        };
    }

    std::array<float, 3> voxel_to_world(const VoxelCoord& c) const {
        return {
            (c.x + 0.5f) * voxel_size_,
            (c.y + 0.5f) * voxel_size_,
            (c.z + 0.5f) * voxel_size_
        };
    }

    Voxel& get_or_create(const VoxelCoord& c) {
        return grid_[c];
    }

    const std::unordered_map<VoxelCoord, Voxel, VoxelCoordHash>& voxels() const {
        return grid_;
    }

    std::size_t size() const {
        return grid_.size();
    }

    void clear() {
        grid_.clear();
    }

    float voxel_size() const { return voxel_size_; }

private:
    float voxel_size_;
    std::unordered_map<VoxelCoord, Voxel, VoxelCoordHash> grid_;
};

}  // namespace terrasem

#endif  // TERRASEM_ROS_VOXEL_GRID_HPP_
