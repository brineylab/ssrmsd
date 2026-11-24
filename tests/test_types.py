from __future__ import annotations

from pathlib import Path

import numpy as np

from ssrmsd.types import (
    ChainData,
    EnsembleResult,
    EnsembleSpec,
    PairwiseDistance,
    ScoringOptions,
    SsrmsdError,
    StructureData,
    StructureInput,
)


def test_default_scoring_options_match_spec() -> None:
    opts = ScoringOptions()
    assert opts.min_chain_id == 0.90
    assert opts.min_coverage == 0.80
    assert opts.linker_min_len == 10
    assert opts.align_chains is None
    assert opts.score_chains is None
    assert opts.ref_source is None
    assert opts.glob_pattern is None
    assert opts.include_top_level is False
    assert opts.split_models is False


def test_error_result_factory() -> None:
    res = EnsembleResult.from_error("mAb1", n_structures=1, error="fewer than 2 structures")
    assert res.ensemble_id == "mAb1"
    assert res.n_structures == 1
    assert res.status == "error"
    assert res.error == "fewer than 2 structures"
    assert res.n_ca_align is None
    assert res.n_ca_score is None
    assert res.ssRMSD is None
    assert res.SD is None
    assert res.mean_sq_rmsd is None
    assert res.pairwise == ()


def test_core_dataclasses_and_error() -> None:
    coords = np.zeros((3, 3), dtype=np.float64)
    chain = ChainData(chain_id="A", sequence="ACD", ca_coords=coords)
    struct = StructureData(name="m0", source="af3", chains=(chain,))
    inp = StructureInput(path=Path("m0.cif"), source="af3")
    spec = EnsembleSpec(ensemble_id="mAb1", inputs=(inp,))
    pair = PairwiseDistance(
        ensemble_id="mAb1",
        struct_i="m0",
        struct_j="m1",
        source_i="af3",
        source_j="esmfold",
        rmsd=1.25,
    )
    assert struct.was_split is False
    assert spec.ensemble_id == "mAb1"
    assert pair.rmsd == 1.25
    assert issubclass(SsrmsdError, ValueError)
