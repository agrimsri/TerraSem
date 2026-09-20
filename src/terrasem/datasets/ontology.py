"""TerraSem-11 unified label ontology and dataset→TerraSem mapping dicts.

This file defines the canonical label space used throughout TerraSem.
All dataset loaders must map their native class IDs into this space.

See BUILD.md §5 Phase 2 for the full specification.

IMPORTANT
---------
``IGNORE_INDEX = 0``  — void/unlabelled pixels are **not** used in the loss.
This is set once here and imported everywhere (loss, metrics, eval scripts).
Do not change it without updating *all* call sites.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TerraSem-11 label space
# ---------------------------------------------------------------------------

IGNORE_INDEX: int = 0
"""Class ID used as ignore index in CrossEntropyLoss and metric computation."""

NUM_CLASSES: int = 11
"""Total number of TerraSem classes including void (ID 0)."""

# (id, name, hex_colour)
CLASSES: list[tuple[int, str, str]] = [
    (0, "void_unlabeled", "#000000"),
    (1, "smooth_traversable", "#808080"),  # asphalt, concrete, hard dirt
    (2, "rough_traversable", "#7fc97f"),   # gravel, grass, sand
    (3, "high_cost_terrain", "#8b4513"),   # mud, tall grass, bush
    (4, "non_traversable_veg", "#228b22"),  # tree trunk, dense vegetation
    (5, "water", "#1e90ff"),               # puddle, stream
    (6, "obstacle_static", "#d2691e"),     # rock, rubble, pole, fence, building
    (7, "obstacle_dynamic", "#ff4500"),    # vehicle, person, animal
    (8, "sky", "#87ceeb"),                 # sky
    (9, "barrier", "#a9a9a9"),             # fence, barrier, wall
    (10, "unknown_other", "#ffffff"),      # catch-all
]

CLASS_NAMES: list[str] = [c[1] for c in CLASSES]
CLASS_COLOURS_HEX: list[str] = [c[2] for c in CLASSES]


# ---------------------------------------------------------------------------
# RELLIS-3D → TerraSem-11
# ---------------------------------------------------------------------------
# Verified against data/rellis3d/Rellis_3D_ontology/ontology.yaml (Phase 2).
# Exactly 20 classes defined in official RELLIS-3D ontology.

RELLIS3D_TO_TERRASEM: dict[int, int] = {
    0: 0,    # void        → void_unlabeled (ignore_index)
    1: 1,    # dirt        → smooth_traversable (dirt path)
    3: 2,    # grass       → rough_traversable
    4: 4,    # tree        → non_traversable_veg
    5: 6,    # pole        → obstacle_static
    6: 5,    # water       → water
    7: 8,    # sky         → sky
    8: 7,    # vehicle     → obstacle_dynamic
    9: 6,    # object      → obstacle_static
    10: 1,   # asphalt     → smooth_traversable
    12: 6,   # building    → obstacle_static
    15: 6,   # log         → obstacle_static
    17: 7,   # person      → obstacle_dynamic
    18: 9,   # fence       → barrier
    19: 3,   # bush        → high_cost_terrain
    23: 1,   # concrete    → smooth_traversable
    27: 9,   # barrier     → barrier
    31: 5,   # puddle      → water
    33: 3,   # mud         → high_cost_terrain
    34: 6,   # rubble      → obstacle_static
}

# UNMAPPED RELLIS-3D class IDs (none — all 20 official classes are mapped)
RELLIS3D_UNMAPPED: dict[int, str] = {}


# ---------------------------------------------------------------------------
# RUGD → TerraSem-11
# ---------------------------------------------------------------------------
# Source: http://rugd.vision/
# TODO (Phase 2): Verify against RUGD annotation files after download.

RUGD_TO_TERRASEM: dict[int, int] = {
    0: 0,    # void
    1: 2,    # dirt         → rough_traversable
    2: 3,    # sand         → high_cost_terrain (fine sand may be high-cost)
    3: 2,    # grass        → rough_traversable
    4: 2,    # gravel       → rough_traversable
    5: 3,    # mud          → high_cost_terrain
    6: 5,    # water        → water
    7: 4,    # rock         → obstacle_static
    8: 6,    # tree         → non_traversable_veg
    9: 3,    # log          → obstacle_static
    10: 4,   # bush         → non_traversable_veg
    11: 6,   # bridge       → obstacle_static
    12: 8,   # sky          → sky
    13: 6,   # building     → obstacle_static
    14: 7,   # vehicle      → obstacle_dynamic
    15: 7,   # person       → obstacle_dynamic
    16: 7,   # animal       → obstacle_dynamic
    17: 6,   # fence        → obstacle_static
    18: 9,   # barrier      → barrier
    19: 10,  # sign         → unknown_other
    20: 1,   # concrete     → smooth_traversable
    21: 1,   # asphalt      → smooth_traversable
    22: 6,   # bicycle      → obstacle_dynamic
    23: 3,   # flowers      → high_cost_terrain
}

RUGD_UNMAPPED: dict[int, str] = {}


# ---------------------------------------------------------------------------
# GOOSE → TerraSem-11
# ---------------------------------------------------------------------------
# Source: https://goose-dataset.de/
# TODO (Phase 2): Pull the official ontology JSON from the dataset and verify.

GOOSE_TO_TERRASEM: dict[int, int] = {
    # Placeholder — will be filled after inspecting GOOSE annotation format.
    # See docs/dataset_inventory.md (auto-generated in Phase 2).
    0: 0,   # void / unlabelled
}

GOOSE_UNMAPPED: dict[int, str] = {
    -1: "ALL — GOOSE mapping not yet verified; update after Phase 2 inspection",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rellis3d_to_terrasem(label_id: int) -> int:
    """Map a RELLIS-3D label ID to a TerraSem-11 ID.

    Unknown IDs map to ``unknown_other`` (10).
    """
    return RELLIS3D_TO_TERRASEM.get(label_id, 10)


def rugd_to_terrasem(label_id: int) -> int:
    """Map a RUGD label ID to a TerraSem-11 ID."""
    return RUGD_TO_TERRASEM.get(label_id, 10)


def goose_to_terrasem(label_id: int) -> int:
    """Map a GOOSE label ID to a TerraSem-11 ID.

    TODO (Phase 2): Implement after inspecting the GOOSE ontology.
    """
    return GOOSE_TO_TERRASEM.get(label_id, 10)
