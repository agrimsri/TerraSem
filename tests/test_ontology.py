"""Unit tests for TerraSem-11 ontology mappings.

Asserts every source class maps into [0, 10] and no class is silently dropped.
See BUILD.md §6 testing requirements.
"""

from __future__ import annotations

from terrasem.datasets.ontology import (
    NUM_CLASSES,
    RELLIS3D_TO_TERRASEM,
    RUGD_TO_TERRASEM,
    goose_to_terrasem,
    rellis3d_to_terrasem,
    rugd_to_terrasem,
)


class TestOntologyMapping:
    def test_rellis3d_all_values_in_range(self):
        for src_id, tgt_id in RELLIS3D_TO_TERRASEM.items():
            assert 0 <= tgt_id < NUM_CLASSES, (
                f"RELLIS-3D class {src_id} maps to {tgt_id}, outside [0, {NUM_CLASSES-1}]"
            )

    def test_rugd_all_values_in_range(self):
        for src_id, tgt_id in RUGD_TO_TERRASEM.items():
            assert 0 <= tgt_id < NUM_CLASSES, (
                f"RUGD class {src_id} maps to {tgt_id}, outside [0, {NUM_CLASSES-1}]"
            )

    def test_goose_all_values_in_range(self):
        from terrasem.datasets.ontology import GOOSE_TO_TERRASEM
        for src_id, tgt_id in GOOSE_TO_TERRASEM.items():
            assert 0 <= tgt_id < NUM_CLASSES, (
                f"GOOSE class {src_id} maps to {tgt_id}, outside [0, {NUM_CLASSES-1}]"
            )


    def test_rellis3d_unknown_maps_to_unknown_other(self):
        """Any RELLIS-3D ID not in the table should map to unknown_other (10)."""
        assert rellis3d_to_terrasem(9999) == 10

    def test_rugd_unknown_maps_to_unknown_other(self):
        assert rugd_to_terrasem(9999) == 10

    def test_goose_unknown_maps_to_unknown_other(self):
        assert goose_to_terrasem(9999) == 10

    def test_void_maps_to_zero(self):
        """Void/unlabelled in every dataset maps to TerraSem ID 0."""
        assert rellis3d_to_terrasem(0) == 0
        assert rugd_to_terrasem(0) == 0
        assert goose_to_terrasem(0) == 0
