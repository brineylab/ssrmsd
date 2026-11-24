from __future__ import annotations

import numpy as np
import pytest
from conftest import AG_SEQ, HC_SEQ, LC_SEQ, make_helix_coords

from ssrmsd.align import align_ensemble
from ssrmsd.types import ChainData, ScoringOptions, SsrmsdError, StructureData


def _make_struct(
    name: str,
    chains: list[tuple[str, str, np.ndarray]],
    *,
    source: str | None = None,
) -> StructureData:
    return StructureData(
        name=name,
        source=source,
        chains=tuple(ChainData(cid, seq, xyz) for cid, seq, xyz in chains),
    )


def test_matches_reordered_and_renamed_chains_across_models() -> None:
    ag_xyz = make_helix_coords(40, offset=(0.0, 0.0, 0.0))
    hc_xyz = make_helix_coords(35, offset=(15.0, 5.0, 0.0))
    lc_xyz = make_helix_coords(35, offset=(15.0, -5.0, 0.0))

    s0 = _make_struct(
        "af3_0",
        [("A", AG_SEQ[:40], ag_xyz), ("B", HC_SEQ[:35], hc_xyz), ("C", LC_SEQ[:35], lc_xyz)],
        source="af3",
    )
    s1 = _make_struct(
        "esm_0",
        [("L", LC_SEQ[:35], lc_xyz), ("H", HC_SEQ[:35], hc_xyz), ("X", AG_SEQ[:40], ag_xyz)],
        source="esmfold",
    )
    aligned = align_ensemble([s0, s1], ScoringOptions())
    assert aligned.chain_mapping[0] == {"A": "A", "B": "B", "C": "C"}
    assert aligned.chain_mapping[1] == {"A": "X", "B": "H", "C": "L"}
    assert aligned.align_coords[0].shape == (110, 3)
    assert aligned.align_coords[1].shape == (110, 3)
    np.testing.assert_allclose(aligned.align_coords[0], aligned.align_coords[1])


def test_poly_g_linker_splitting_regardless_of_sort_order() -> None:
    ag_seq = AG_SEQ[:30]
    hc_seq = HC_SEQ[:25]
    lc_seq = LC_SEQ[:25]
    linker = "G" * 20

    ag_xyz = make_helix_coords(len(ag_seq), offset=(0.0, 0.0, 0.0))
    hc_xyz = make_helix_coords(len(hc_seq), offset=(15.0, 5.0, 0.0))
    lc_xyz = make_helix_coords(len(lc_seq), offset=(15.0, -5.0, 0.0))
    link1 = make_helix_coords(len(linker), offset=(5.0, 0.0, 0.0))
    link2 = make_helix_coords(len(linker), offset=(10.0, 0.0, 0.0))

    linked_seq = ag_seq + linker + hc_seq + linker + lc_seq
    linked_xyz = np.vstack([ag_xyz, link1, hc_xyz, link2, lc_xyz])

    # Single-chain ESMFold structure sorted BEFORE multi-chain AF3 structure
    s_esm = _make_struct("1_esm", [("A", linked_seq, linked_xyz)], source="esmfold")
    s_af3 = _make_struct(
        "2_af3",
        [("A", ag_seq, ag_xyz), ("B", hc_seq, hc_xyz), ("C", lc_seq, lc_xyz)],
        source="af3",
    )

    aligned = align_ensemble([s_esm, s_af3], ScoringOptions())
    assert aligned.align_coords[0].shape == (80, 3)
    assert aligned.align_coords[1].shape == (80, 3)
    # Default reference should prefer naturally 3-chain AF3 structure -> real chain IDs A, B, C
    assert set(aligned.chain_mapping[1].keys()) == {"A", "B", "C"}
    np.testing.assert_allclose(aligned.align_coords[0], aligned.align_coords[1])

    # Disabling linker splitting must fail
    with pytest.raises(SsrmsdError):
        align_ensemble([s_esm, s_af3], ScoringOptions(linker_min_len=0))


def test_strict_ref_source_error_when_missing() -> None:
    ag_xyz = make_helix_coords(30)
    s0 = _make_struct("e0", [("A", AG_SEQ[:30], ag_xyz)], source="esmfold")
    s1 = _make_struct("e1", [("A", AG_SEQ[:30], ag_xyz)], source="esmfold")
    with pytest.raises(SsrmsdError, match="ref-source"):
        align_ensemble([s0, s1], ScoringOptions(ref_source="af3"))


def test_align_and_score_chain_subsets_and_spectator_chains() -> None:
    ag_xyz = make_helix_coords(30, offset=(0.0, 0.0, 0.0))
    hc_xyz = make_helix_coords(25, offset=(15.0, 5.0, 0.0))
    lc_xyz = make_helix_coords(25, offset=(15.0, -5.0, 0.0))
    extra_seq = "WYFHKRDEQNPSTVWYFFHKRDEQ"
    extra_xyz = make_helix_coords(len(extra_seq), offset=(-20.0, 0.0, 0.0))

    s0 = _make_struct(
        "af3_0",
        [("A", AG_SEQ[:30], ag_xyz), ("B", HC_SEQ[:25], hc_xyz), ("C", LC_SEQ[:25], lc_xyz)],
        source="af3",
    )
    s1_extra = _make_struct(
        "other_0",
        [
            ("A", AG_SEQ[:30], ag_xyz),
            ("B", HC_SEQ[:25], hc_xyz),
            ("C", LC_SEQ[:25], lc_xyz),
            ("D", extra_seq, extra_xyz),
        ],
        source="other",
    )

    # Default (no chain-selection flags): extra spectator chain rejected
    with pytest.raises(SsrmsdError):
        align_ensemble([s0, s1_extra], ScoringOptions())

    # With both align_chains & score_chains: spectator chain ignored
    aligned_both = align_ensemble(
        [s0, s1_extra],
        ScoringOptions(align_chains=("A",), score_chains=("B", "C")),
    )
    assert aligned_both.align_coords[0].shape == (30, 3)
    assert aligned_both.score_coords[0].shape == (50, 3)

    # With only align_chains: score_coords covers all reference chains
    aligned_a_only = align_ensemble([s0, s1_extra], ScoringOptions(align_chains=("A",)))
    assert aligned_a_only.align_coords[0].shape == (30, 3)
    assert aligned_a_only.score_coords[0].shape == (80, 3)

    # With only score_chains: align_coords covers all reference chains
    aligned_s_only = align_ensemble([s0, s1_extra], ScoringOptions(score_chains=("B", "C")))
    assert aligned_s_only.align_coords[0].shape == (80, 3)
    assert aligned_s_only.score_coords[0].shape == (50, 3)

    # Unknown chain ID rejected
    with pytest.raises(SsrmsdError, match="Z"):
        align_ensemble([s0, s1_extra], ScoringOptions(align_chains=("Z",)))


def test_terminal_truncation_overlap_identity_and_symmetric_coverage() -> None:
    full_seq = AG_SEQ[:40]
    full_xyz = make_helix_coords(40)
    # 85% coverage (34 of 40 residues), 100% identity over overlap
    trunc_85_seq = full_seq[3:37]
    trunc_85_xyz = full_xyz[3:37]

    s_full = _make_struct("s0", [("A", full_seq, full_xyz)])
    s_trunc85 = _make_struct("s1", [("A", trunc_85_seq, trunc_85_xyz)])
    aligned = align_ensemble([s_full, s_trunc85], ScoringOptions())
    assert aligned.align_coords[0].shape == (34, 3)
    assert aligned.align_coords[1].shape == (34, 3)
    np.testing.assert_allclose(aligned.align_coords[0], aligned.align_coords[1])

    # 50% coverage must fail even when truncated structure sorts FIRST as reference
    trunc_50_seq = full_seq[10:30]
    trunc_50_xyz = full_xyz[10:30]
    s_trunc50 = _make_struct("0_trunc", [("A", trunc_50_seq, trunc_50_xyz)])
    with pytest.raises(SsrmsdError, match="coverage"):
        align_ensemble([s_trunc50, s_full], ScoringOptions())


def test_homomer_and_ambiguous_chain_guard() -> None:
    ag_seq = AG_SEQ[:30]
    hc_seq = HC_SEQ[:25]
    ag_xyz = make_helix_coords(30)
    hc_xyz = make_helix_coords(25, offset=(15.0, 0.0, 0.0))

    # Reference has two identical chains A and D -> homomer guard fires
    s_homo_0 = _make_struct(
        "m0", [("A", ag_seq, ag_xyz), ("D", ag_seq, ag_xyz + 10.0), ("B", hc_seq, hc_xyz)]
    )
    s_homo_1 = _make_struct(
        "m1", [("A", ag_seq, ag_xyz), ("D", ag_seq, ag_xyz + 10.0), ("B", hc_seq, hc_xyz)]
    )
    with pytest.raises(SsrmsdError, match="ambiguous|homomer"):
        align_ensemble([s_homo_0, s_homo_1], ScoringOptions())

    # Non-reference has two copies matching ref chain A above threshold -> ambiguous guard fires
    s_ref = _make_struct("ref", [("A", ag_seq, ag_xyz), ("B", hc_seq, hc_xyz)])
    s_dup = _make_struct(
        "dup",
        [("A", ag_seq, ag_xyz), ("D", ag_seq, ag_xyz + 5.0), ("B", hc_seq, hc_xyz)],
    )
    with pytest.raises(SsrmsdError, match="ambiguous|homomer"):
        align_ensemble([s_ref, s_dup], ScoringOptions(align_chains=("A",), score_chains=("B",)))
