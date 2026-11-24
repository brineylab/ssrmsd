from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import (
    AG_SEQ,
    HC_SEQ,
    LC_SEQ,
    make_af3_job_dir,
    make_complex_chains,
    make_helix_coords,
    make_tar_gz,
    write_pdb,
)

import ssrmsd
from ssrmsd.score import score_batch, score_ensemble
from ssrmsd.types import EnsembleSpec, SsrmsdError, StructureInput


def test_score_ensemble_mixed_af3_tar_gz_and_esmfold_pdb(tmp_path: Path) -> None:
    # 5-sample AF3 .tar.gz with top-level duplicate (must yield 5 structures)
    af3_dir = make_af3_job_dir(tmp_path / "af3", "mab1", n_samples=5, include_top_level_dup=True)
    af3_tar = make_tar_gz(af3_dir, tmp_path / "mab1.tar.gz")

    # 2 ESMFold single-chain poly-G linker PDBs with 3-residue N-term Ag truncation (90% coverage)
    ag_seq = AG_SEQ[:30]
    hc_seq = HC_SEQ[:25]
    lc_seq = LC_SEQ[:25]
    linker = "G" * 20

    esm_paths: list[Path] = []
    for k in range(2):
        base = make_complex_chains(
            ag_seq=ag_seq,
            hc_seq=hc_seq,
            lc_seq=lc_seq,
            ab_shift=(5.0 + 0.2 * k, 0.0, 0.0),
            noise_seed=200 + k,
            noise_scale=0.1,
        )
        ag_trunc_seq = base[0][1][3:]
        ag_trunc_xyz = base[0][2][3:]
        link1 = make_helix_coords(len(linker), offset=(5.0, 0.0, 0.0))
        link2 = make_helix_coords(len(linker), offset=(10.0, 0.0, 0.0))
        linked_seq = ag_trunc_seq + linker + base[1][1] + linker + base[2][1]
        linked_xyz = np.vstack([ag_trunc_xyz, link1, base[1][2], link2, base[2][2]])
        esm_paths.append(write_pdb(tmp_path / f"esm_{k}.pdb", [("A", linked_seq, linked_xyz)]))

    res = score_ensemble(
        [
            StructureInput(af3_tar, source="af3"),
            StructureInput(esm_paths[0], source="esmfold"),
            StructureInput(esm_paths[1], source="esmfold"),
        ],
        ensemble_id="mab1",
        include_pairwise=True,
    )
    assert res.status == "ok"
    assert res.ensemble_id == "mab1"
    assert res.n_structures == 7
    assert res.n_ca_align == 77  # 27 Ag + 25 HC + 25 LC
    assert res.n_ca_score == 77
    assert res.sources == "af3:5,esmfold:2"
    assert res.within_source_mean_rmsd is not None
    assert res.cross_source_mean_rmsd is not None
    assert len(res.pairwise) == 21


def test_score_ensemble_fewer_than_two_structures(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    r0 = score_ensemble([empty_dir], ensemble_id="e0")
    assert r0.status == "error"
    assert "fewer than 2 structures" in r0.error
    assert r0.n_structures == 0

    single = write_pdb(tmp_path / "only.pdb", make_complex_chains())
    r1 = score_ensemble([single], ensemble_id="e1")
    assert r1.status == "error"
    assert "fewer than 2 structures" in r1.error
    assert r1.n_structures == 1


def test_score_batch_multiprocess_determinism_and_error_isolation(tmp_path: Path) -> None:
    specs: list[EnsembleSpec] = []
    for i in range(4):
        d = make_af3_job_dir(tmp_path / "jobs", f"mab_{i}", n_samples=3)
        specs.append(EnsembleSpec(f"mab_{i}", (StructureInput(d, source="af3"),)))

    # 1 corrupt .tar.gz
    bad_tar = tmp_path / "corrupt.tar.gz"
    bad_tar.write_bytes(b"not a valid tar gz archive")
    specs.insert(2, EnsembleSpec("mab_corrupt", (StructureInput(bad_tar, source="af3"),)))

    # 1 single-structure ensemble
    single = write_pdb(tmp_path / "one.pdb", make_complex_chains())
    specs.append(EnsembleSpec("mab_single", (StructureInput(single),)))

    results = score_batch(specs, workers=2)
    assert [r.ensemble_id for r in results] == [s.ensemble_id for s in specs]
    ok_ids = [r.ensemble_id for r in results if r.status == "ok"]
    err_rows = {r.ensemble_id: r for r in results if r.status == "error"}
    assert ok_ids == ["mab_0", "mab_1", "mab_2", "mab_3"]
    assert set(err_rows.keys()) == {"mab_corrupt", "mab_single"}
    assert "corrupt.tar.gz" in err_rows["mab_corrupt"].error
    assert "fewer than 2 structures" in err_rows["mab_single"].error


def test_score_batch_fail_fast_raises(tmp_path: Path) -> None:
    d = make_af3_job_dir(tmp_path, "good", n_samples=2)
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"corrupt")
    specs = [
        EnsembleSpec("good", (StructureInput(d),)),
        EnsembleSpec("bad", (StructureInput(bad),)),
    ]
    with pytest.raises(SsrmsdError):
        score_batch(specs, workers=2, fail_fast=True)


def test_public_api_imports() -> None:
    assert ssrmsd.score_ensemble is score_ensemble
    assert ssrmsd.score_batch is score_batch
    assert hasattr(ssrmsd, "EnsembleSpec")
    assert hasattr(ssrmsd, "EnsembleResult")
    assert hasattr(ssrmsd, "ScoringOptions")
    assert hasattr(ssrmsd, "StructureInput")


def test_score_batch_fail_fast_cancels_pending_futures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ProcessPoolExecutor as RealPool

    shutdown_calls: list[dict[str, object]] = []

    class TrackingPool(RealPool):
        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr("ssrmsd.score.ProcessPoolExecutor", TrackingPool)

    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"corrupt")
    d = make_af3_job_dir(tmp_path, "good", n_samples=2)
    specs = [
        EnsembleSpec("bad", (StructureInput(bad),)),
        *[EnsembleSpec(f"good_{i}", (StructureInput(d),)) for i in range(6)],
    ]
    with pytest.raises(SsrmsdError):
        score_batch(specs, workers=2, fail_fast=True)

    assert any(call["cancel_futures"] is True for call in shutdown_calls)
