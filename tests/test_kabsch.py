from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import apply_rigid_transform, make_complex_chains, make_helix_coords, rotation_matrix

from ssrmsd.align import AlignedEnsemble
from ssrmsd.kabsch import aggregate_metrics, compute_pairwise_matrix, kabsch_rmsd


def _legacy_af3_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    """Exact copy of `_kabsch_rmsd` from the legacy AF3 Slurm worker script."""
    P32 = P.astype(np.float32)
    Q32 = Q.astype(np.float32)
    Pc = P32 - P32.mean(0)
    Qc = Q32 - Q32.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Vt[-1] *= -1
    R = Vt.T @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff * diff).sum() / len(P32)))


def test_kabsch_identical_and_rigid_transformed_is_zero() -> None:
    p = make_helix_coords(40)
    assert kabsch_rmsd(p, p) == 0.0
    assert not math.isnan(kabsch_rmsd(p, p))

    rot = rotation_matrix((1.0, 2.0, -0.5), 1.23)
    q = apply_rigid_transform(p, rot=rot, shift=(12.5, -7.2, 4.1))
    assert kabsch_rmsd(p, q) < 1e-12

    # Mirror reflection (det = -1) must NOT be collapsed to zero RMSD
    q_reflected = p.copy()
    q_reflected[:, 2] *= -1.0
    assert kabsch_rmsd(p, q_reflected) > 0.5


def test_kabsch_separate_align_and_score_sets() -> None:
    ag = make_helix_coords(30, offset=(0.0, 0.0, 0.0))
    ab = make_helix_coords(25, offset=(15.0, 0.0, 0.0))
    ab_shifted = ab + np.array([3.0, 4.0, 0.0])  # exact 5.0 A translation

    # Full-complex global Kabsch spreads the error across Ag+Ab (< 5.0 A)
    full_p = np.vstack([ag, ab])
    full_q = np.vstack([ag, ab_shifted])
    assert kabsch_rmsd(full_p, full_q) < 5.0

    # Align on antigen (exact 0 rotation/translation), score on antibody -> exact 5.0 A
    pose_rmsd = kabsch_rmsd(ag, ag, ab, ab_shifted)
    assert pose_rmsd == pytest.approx(5.0, abs=1e-12)

    # Also invariant when whole complex q is rigidly rotated+translated
    rot = rotation_matrix((0.0, 1.0, 1.0), 0.9)
    ag_q = apply_rigid_transform(ag, rot=rot, shift=(-10.0, 3.0, 8.0))
    ab_q = apply_rigid_transform(ab_shifted, rot=rot, shift=(-10.0, 3.0, 8.0))
    assert kabsch_rmsd(ag, ag_q, ab, ab_q) == pytest.approx(5.0, abs=1e-12)


def test_legacy_parity_with_af3_worker_formula() -> None:
    shifts = [
        (0.0, 0.0, 0.0),
        (1.2, -0.5, 0.3),
        (-0.8, 2.1, 0.0),
        (3.5, 1.0, -1.5),
        (0.4, 0.4, 2.2),
    ]
    coords_list = [
        np.vstack(
            [xyz for _, _, xyz in make_complex_chains(ab_shift=s, noise_seed=i, noise_scale=0.15)]
        )
        for i, s in enumerate(shifts)
    ]

    # Legacy AF3 worker calculation
    n = len(coords_list)
    legacy_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = _legacy_af3_kabsch(coords_list[i], coords_list[j])
            legacy_mat[i, j] = v
            legacy_mat[j, i] = v
    legacy_ssrmsd = round(float(np.sum(legacy_mat**2)), 4)
    legacy_sd = round(float(np.std(legacy_mat)), 4)

    aligned = AlignedEnsemble(
        names=tuple(f"sample-{i}" for i in range(n)),
        sources=("af3",) * n,
        align_coords=tuple(coords_list),
        score_coords=tuple(coords_list),
        chain_mapping=({"A": "A", "B": "B", "C": "C"},) * n,
    )
    mat = compute_pairwise_matrix(aligned)
    res = aggregate_metrics("mab1", aligned, mat)

    assert res.status == "ok"
    assert res.ssRMSD is not None and res.SD is not None and res.mean_sq_rmsd is not None
    assert abs(round(res.ssRMSD, 4) - legacy_ssrmsd) <= 1e-4
    assert abs(round(res.SD, 4) - legacy_sd) <= 1e-4
    assert res.mean_sq_rmsd == pytest.approx(res.ssRMSD / (n * (n - 1)), abs=1e-10)
    assert res.rms_rmsd == pytest.approx(math.sqrt(res.mean_sq_rmsd), abs=1e-10)


def test_normalized_metrics_scale_invariance() -> None:
    # Constant pairwise distance d = 2.0 across n=5 vs n=10
    def _const_result(n: int) -> tuple[float, float, float]:
        mat = np.full((n, n), 2.0, dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        dummy = make_helix_coords(10)
        aligned = AlignedEnsemble(
            names=tuple(f"m{i}" for i in range(n)),
            sources=(None,) * n,
            align_coords=(dummy,) * n,
            score_coords=(dummy,) * n,
            chain_mapping=({"A": "A"},) * n,
        )
        r = aggregate_metrics(f"n{n}", aligned, mat)
        assert r.ssRMSD is not None and r.mean_sq_rmsd is not None and r.rms_rmsd is not None
        return r.ssRMSD, r.mean_sq_rmsd, r.rms_rmsd

    ss5, msq5, rms5 = _const_result(5)
    ss10, msq10, rms10 = _const_result(10)
    assert ss10 / ss5 == pytest.approx(90.0 / 20.0)
    assert msq5 == pytest.approx(4.0)
    assert msq10 == pytest.approx(4.0)
    assert rms5 == pytest.approx(2.0)
    assert rms10 == pytest.approx(2.0)


def test_multi_source_within_and_cross_metrics() -> None:
    ag = make_helix_coords(20, offset=(0.0, 0.0, 0.0))
    ab = make_helix_coords(15, offset=(15.0, 0.0, 0.0))

    # 3 tight AF3 poses + 2 tight ESMFold poses shifted 6 A away
    af3_coords = [ab + np.array([0.1 * i, 0.0, 0.0]) for i in range(3)]
    esm_coords = [ab + np.array([6.0 + 0.1 * i, 0.0, 0.0]) for i in range(2)]
    all_score = tuple(af3_coords + esm_coords)
    aligned = AlignedEnsemble(
        names=("a0", "a1", "a2", "e0", "e1"),
        sources=("af3", "af3", "af3", "esmfold", "esmfold"),
        align_coords=(ag,) * 5,
        score_coords=all_score,
        chain_mapping=({"A": "A", "B": "B"},) * 5,
    )
    mat = compute_pairwise_matrix(aligned)
    res = aggregate_metrics("mix", aligned, mat, include_pairwise=True)
    assert res.sources == "af3:3,esmfold:2"
    assert res.within_source_mean_rmsd is not None
    assert res.cross_source_mean_rmsd is not None
    assert res.within_source_mean_rmsd < 0.2
    assert res.cross_source_mean_rmsd > 5.5
    assert len(res.pairwise) == 10

    # 1 af3 + 1 esmfold: 0 within-source pairs -> within_source_mean_rmsd is None
    aligned_2 = AlignedEnsemble(
        names=("a0", "e0"),
        sources=("af3", "esmfold"),
        align_coords=(ag, ag),
        score_coords=(af3_coords[0], esm_coords[0]),
        chain_mapping=({"A": "A", "B": "B"},) * 2,
    )
    res2 = aggregate_metrics("pair", aligned_2, compute_pairwise_matrix(aligned_2))
    assert res2.sources == "af3:1,esmfold:1"
    assert res2.within_source_mean_rmsd is None
    assert res2.cross_source_mean_rmsd == pytest.approx(6.0, abs=1e-10)

    # Unlabeled structures -> empty sources and None multi-source fields
    aligned_unlabeled = AlignedEnsemble(
        names=("m0", "m1"),
        sources=(None, None),
        align_coords=(ag, ag),
        score_coords=(af3_coords[0], esm_coords[0]),
        chain_mapping=({"A": "A", "B": "B"},) * 2,
    )
    res_unlabeled = aggregate_metrics(
        "u", aligned_unlabeled, compute_pairwise_matrix(aligned_unlabeled)
    )
    assert res_unlabeled.sources == ""
    assert res_unlabeled.within_source_mean_rmsd is None
    assert res_unlabeled.cross_source_mean_rmsd is None


def test_pairwise_records_emitted_when_requested() -> None:
    ag = make_helix_coords(15)
    ab = make_helix_coords(12, offset=(10.0, 0.0, 0.0))
    aligned = AlignedEnsemble(
        names=("s0", "s1", "s2"),
        sources=("af3", "af3", "esm"),
        align_coords=(ag, ag, ag),
        score_coords=(ab, ab + np.array([1.0, 0.0, 0.0]), ab + np.array([3.0, 0.0, 0.0])),
        chain_mapping=({"A": "A", "B": "B"},) * 3,
    )
    mat = compute_pairwise_matrix(aligned)
    res_no = aggregate_metrics("e1", aligned, mat, include_pairwise=False)
    assert res_no.pairwise == ()

    res_yes = aggregate_metrics("e1", aligned, mat, include_pairwise=True)
    assert len(res_yes.pairwise) == 3
    assert [(p.struct_i, p.struct_j, round(p.rmsd, 4)) for p in res_yes.pairwise] == [
        ("s0", "s1", 1.0),
        ("s0", "s2", 3.0),
        ("s1", "s2", 2.0),
    ]
