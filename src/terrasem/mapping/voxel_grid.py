"""3D Voxel Grid with Bayesian occupancy and Dirichlet semantic belief.

Per BUILD.md Phase 5:
- Sparse voxel grid keyed by integer (ix, iy, iz) coordinates.
- Each voxel stores:
    - occupancy log-odds float32
    - Dirichlet semantic alpha parameters (float32 vector, length 11)
    - hit / miss observation counts
    - min / max Z elevation observed
- Coordinate conversions: world/local metre <-> integer voxel index.
- Computes calibrated occupancy probability, Dirichlet class posteriors, and normalized entropy uncertainty.
"""

from __future__ import annotations

import numpy as np

from terrasem.datasets.ontology import NUM_CLASSES


class Voxel:
    """Per-voxel data structure."""

    __slots__ = ("log_odds", "alpha", "hit_count", "miss_count", "z_min", "z_max")

    def __init__(self, prior_alpha: float = 0.1) -> None:
        self.log_odds: float = 0.0  # 0.0 log-odds <=> p=0.5 prior
        # Dirichlet parameters (alpha_0 = prior_alpha per class)
        self.alpha: np.ndarray = np.full(NUM_CLASSES, prior_alpha, dtype=np.float32)
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.z_min: float = float("inf")
        self.z_max: float = float("-inf")

    @property
    def occupancy_prob(self) -> float:
        """Occupancy probability p in [0.0, 1.0] from log-odds."""
        # p = 1 / (1 + exp(-l)) = exp(l) / (1 + exp(l))
        # Numerically stable sigmoid:
        if self.log_odds >= 0:
            return float(1.0 / (1.0 + np.exp(-self.log_odds)))
        else:
            z = np.exp(self.log_odds)
            return float(z / (1.0 + z))

    @property
    def class_probabilities(self) -> np.ndarray:
        """Dirichlet posterior mean class probabilities P(c) = alpha_c / sum(alpha)."""
        total = float(np.sum(self.alpha))
        if total <= 0.0:
            return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)
        return self.alpha / total

    @property
    def dominant_class(self) -> int:
        """Argmax semantic class ID (ignoring void=0 if other evidence present)."""
        probs = self.class_probabilities.copy()
        # If any class > 0 has received hits, exclude void (0)
        if np.any(probs[1:] > (1.0 / NUM_CLASSES)):
            probs[0] = -1.0
        return int(np.argmax(probs))

    @property
    def normalized_entropy(self) -> float:
        """Normalized Shannon entropy uncertainty H in [0.0, 1.0].

        H = -sum_c P(c) * log(P(c)) / log(num_classes)
        """
        probs = self.class_probabilities
        # Filter p > 0
        p_valid = probs[probs > 1e-7]
        entropy = -np.sum(p_valid * np.log(p_valid))
        max_entropy = np.log(NUM_CLASSES)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))


class VoxelGrid:
    """Sparse 3D voxel grid backed by a Python dict keyed by (ix, iy, iz).

    Args:
        voxel_size: Edge length of each cubic voxel in metres (default 0.2m).
        prior_alpha: Dirichlet prior parameter (default 0.1).
        bounds: Optional (min_x, max_x, min_y, max_y, min_z, max_z) bounding box.
    """

    def __init__(
        self,
        voxel_size: float = 0.2,
        prior_alpha: float = 0.1,
        bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.prior_alpha = float(prior_alpha)
        self.bounds = bounds
        self._grid: dict[tuple[int, int, int], Voxel] = {}

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """Convert Nx3 world coordinates (metres) to integer voxel indices.

        Args:
            xyz: (N, 3) or (3,) array in metres.

        Returns:
            (N, 3) or (3,) int64 array of voxel indices.
        """
        return np.floor(np.asarray(xyz) / self.voxel_size).astype(np.int64)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Convert Nx3 integer indices to voxel centroid coordinates in metres.

        Args:
            idx: (N, 3) or (3,) int array.

        Returns:
            (N, 3) or (3,) float64 centroid coordinates.
        """
        return (np.asarray(idx, dtype=np.float64) + 0.5) * self.voxel_size

    def is_in_bounds(self, xyz: np.ndarray) -> np.ndarray:
        """Check if Nx3 coordinates lie within configured bounding box."""
        if self.bounds is None:
            return np.ones(len(xyz), dtype=bool)
        min_x, max_x, min_y, max_y, min_z, max_z = self.bounds
        return (
            (xyz[:, 0] >= min_x)
            & (xyz[:, 0] <= max_x)
            & (xyz[:, 1] >= min_y)
            & (xyz[:, 1] <= max_y)
            & (xyz[:, 2] >= min_z)
            & (xyz[:, 2] <= max_z)
        )

    def get_or_create(self, key: tuple[int, int, int]) -> Voxel:
        """Retrieve voxel at index or create a new initialized voxel."""
        voxel = self._grid.get(key)
        if voxel is None:
            voxel = Voxel(prior_alpha=self.prior_alpha)
            self._grid[key] = voxel
        return voxel

    def get(self, key: tuple[int, int, int]) -> Voxel | None:
        """Retrieve voxel at index or None."""
        return self._grid.get(key)

    def __contains__(self, key: tuple[int, int, int]) -> bool:
        return key in self._grid

    def __len__(self) -> int:
        return len(self._grid)

    def keys(self):
        return self._grid.keys()

    def items(self):
        return self._grid.items()

    def values(self):
        return self._grid.values()

    def clear(self) -> None:
        self._grid.clear()
