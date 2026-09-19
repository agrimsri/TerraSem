"""Unit tests for dataset split integrity.

Asserts no sequence appears in more than one split.
See BUILD.md §6 testing requirements.
TODO (Phase 2): add real split file checks once splits are generated.
"""

from __future__ import annotations

import pytest


def test_no_sequence_overlap_placeholder():
    """Placeholder — activate once data/splits/ exists (Phase 2)."""
    pytest.skip("Phase 2 splits not yet generated")
