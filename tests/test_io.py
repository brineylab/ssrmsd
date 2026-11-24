from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import (
    make_af3_job_dir,
    make_complex_chains,
    make_tar_gz,
    make_zip,
    write_cif,
    write_pdb,
)

from ssrmsd.io import (
    discover_batch_dirs,
    load_ensemble_structures,
    load_manifest,
    parse_labeled_path,
    read_structures_from_bytes,
)
from ssrmsd.types import ScoringOptions, SsrmsdError, StructureInput


def test_parse_cif_pdb_and_gzipped_variants(tmp_path: Path) -> None:
    chains = make_complex_chains(noise_seed=1, noise_scale=0.05)
    paths = [
        write_cif(tmp_path / "a.cif", chains, add_water=True, add_altloc_b=True),
        write_pdb(tmp_path / "a.pdb", chains, add_water=True, add_altloc_b=True),
        write_cif(tmp_path / "a.cif.gz", chains, gzipped=True, add_water=True, add_altloc_b=True),
        write_pdb(tmp_path / "a.pdb.gz", chains, gzipped=True, add_water=True, add_altloc_b=True),
    ]
    parsed = [
        load_ensemble_structures([StructureInput(p)], ScoringOptions())[0] for p in paths
    ]
    ref = parsed[0]
    assert [c.chain_id for c in ref.chains] == ["A", "B", "C"]
    assert ref.chains[0].ca_coords.dtype == np.float64
    for s in parsed:
        assert [c.chain_id for c in s.chains] == ["A", "B", "C"]
        for c_ref, c_cur, (_, seq, xyz) in zip(ref.chains, s.chains, chains, strict=True):
            assert c_cur.sequence == seq
            # Primary conformer (altloc A) must be chosen, not altloc B (+5.0 shifted)
            np.testing.assert_allclose(c_cur.ca_coords, xyz, atol=1e-3)
            np.testing.assert_allclose(c_cur.ca_coords, c_ref.ca_coords, atol=1e-3)


def test_modified_residues_and_missing_ca_atom(tmp_path: Path) -> None:
    chains = make_complex_chains(ag_seq="MKTAYIAKQR", hc_seq="EVQLVESGGG", lc_seq="DIQMTQSPSS")
    cif = write_cif(
        tmp_path / "mod.cif",
        chains,
        resname_overrides={("A", 1): "MSE", ("A", 2): "ZZZ"},
        omit_ca={("A", 4)},
    )
    loaded = read_structures_from_bytes(cif.read_bytes(), cif.name)
    assert len(loaded) == 1
    chain_a = loaded[0].chains[0]
    # Residue 1 MSE -> M, residue 2 ZZZ -> X, residue 4 (A) omitted because CA missing
    assert chain_a.sequence == "MXTYIAKQR"
    assert chain_a.ca_coords.shape == (9, 3)


def test_multi_model_default_vs_split_models(tmp_path: Path) -> None:
    c0 = make_complex_chains(ab_shift=(0.0, 0.0, 0.0))
    c1 = make_complex_chains(ab_shift=(4.0, 0.0, 0.0))
    pdb = write_pdb(tmp_path / "multi.pdb", c0, extra_models=[c1])

    default = load_ensemble_structures([StructureInput(pdb)], ScoringOptions(split_models=False))
    assert len(default) == 1
    assert default[0].name == "multi.pdb"

    split = load_ensemble_structures([StructureInput(pdb)], ScoringOptions(split_models=True))
    assert len(split) == 2
    assert [s.name for s in split] == ["multi.pdb#model0", "multi.pdb#model1"]
    # Second model's chain B should be shifted by +4.0 in X relative to first model
    delta = split[1].chains[1].ca_coords - split[0].chains[1].ca_coords
    np.testing.assert_allclose(delta[:, 0], 4.0, atol=1e-3)


def test_af3_dir_and_tar_gz_skip_top_level_duplicate(tmp_path: Path) -> None:
    job_dir = make_af3_job_dir(tmp_path, "mab1", n_samples=5, include_top_level_dup=True)
    tar_path = make_tar_gz(job_dir, tmp_path / "mab1.tar.gz")

    dir_default = load_ensemble_structures([StructureInput(job_dir)], ScoringOptions())
    assert len(dir_default) == 5
    assert all("seed-1_sample-" in s.name for s in dir_default)

    tar_default = load_ensemble_structures([StructureInput(tar_path)], ScoringOptions())
    assert len(tar_default) == 5
    assert all("seed-1_sample-" in s.name for s in tar_default)

    dir_incl = load_ensemble_structures(
        [StructureInput(job_dir)], ScoringOptions(include_top_level=True)
    )
    assert len(dir_incl) == 6

    tar_incl = load_ensemble_structures(
        [StructureInput(tar_path)], ScoringOptions(include_top_level=True)
    )
    assert len(tar_incl) == 6

    glob_only = load_ensemble_structures(
        [StructureInput(job_dir)], ScoringOptions(glob_pattern="*sample-0/*.cif")
    )
    assert len(glob_only) == 1


def test_af3_server_zip_in_memory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    files: list[tuple[str, Path]] = []
    for idx in range(5):
        p = write_cif(staging / f"mab1_model_{idx}.cif", make_complex_chains(noise_seed=idx))
        files.append((f"mab1_model_{idx}.cif", p))
    zip_path = make_zip(files, tmp_path / "mab1.zip")

    loaded = load_ensemble_structures([StructureInput(zip_path, source="af3")], ScoringOptions())
    assert len(loaded) == 5
    assert [s.name for s in loaded] == [f"mab1_model_{i}.cif" for i in range(5)]
    assert all(s.source == "af3" for s in loaded)


def test_parse_labeled_path_handles_equals_in_dirname(tmp_path: Path) -> None:
    assert parse_labeled_path("af3=/tmp/some_job") == StructureInput(Path("/tmp/some_job"), "af3")

    eq_dir = tmp_path / "run_lr=1e-4" / "job1"
    eq_dir.mkdir(parents=True)
    assert parse_labeled_path(str(eq_dir)) == StructureInput(eq_dir, None)

    rel_eq = "dir_with_slash/a=b.cif"
    assert parse_labeled_path(rel_eq) == StructureInput(Path(rel_eq), None)


def test_discover_batch_dirs_union_and_intersection(tmp_path: Path) -> None:
    af3_root = tmp_path / "af3_runs"
    esm_root = tmp_path / "esm_runs"

    # mab1: AF3 dir + ESM flat <mab>_model_<k>.pdb files
    make_af3_job_dir(af3_root, "mab1", n_samples=2)
    for k in range(2):
        write_pdb(esm_root / f"mab1_model_{k}.pdb", make_complex_chains(noise_seed=k))

    # mab2: AF3 .tar.gz only
    staging = make_af3_job_dir(tmp_path / "stage", "mab2", n_samples=2)
    make_tar_gz(staging, af3_root / "mab2.tar.gz")

    # mab3: ESM .zip only
    z_src = write_pdb(tmp_path / "stage" / "mab3_0.pdb", make_complex_chains())
    make_zip([("mab3_0.pdb", z_src)], esm_root / "mab3.zip")

    roots = [StructureInput(af3_root, "af3"), StructureInput(esm_root, "esm")]

    union_specs = discover_batch_dirs(roots, join="union")
    assert [s.ensemble_id for s in union_specs] == ["mab1", "mab2", "mab3"]
    mab1_inputs = union_specs[0].inputs
    assert len(mab1_inputs) == 3  # 1 af3 dir + 2 esm flat pdbs
    assert {inp.source for inp in mab1_inputs} == {"af3", "esm"}

    inter_specs = discover_batch_dirs(roots, join="intersection")
    assert [s.ensemble_id for s in inter_specs] == ["mab1"]


def test_load_manifest_csv_and_tsv(tmp_path: Path) -> None:
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "ensemble_id,path,source\n"
        "mab1,/data/af3/mab1.tar.gz,af3\n"
        "mab1,/data/esm/mab1_0.pdb,esmfold\n"
        "mab2,/data/af3/mab2,\n",
        encoding="utf-8",
    )
    specs = load_manifest(csv_path)
    assert [s.ensemble_id for s in specs] == ["mab1", "mab2"]
    assert specs[0].inputs == (
        StructureInput(Path("/data/af3/mab1.tar.gz"), "af3"),
        StructureInput(Path("/data/esm/mab1_0.pdb"), "esmfold"),
    )
    assert specs[1].inputs == (StructureInput(Path("/data/af3/mab2"), None),)

    tsv_path = tmp_path / "manifest.tsv"
    tsv_path.write_text("ensemble_id\tpath\nmab9\t/tmp/m9.cif\n", encoding="utf-8")
    tsv_specs = load_manifest(tsv_path)
    assert len(tsv_specs) == 1
    assert tsv_specs[0].ensemble_id == "mab9"

    bad = tmp_path / "bad.csv"
    bad.write_text("id,file\nmab1,/tmp/x.cif\n", encoding="utf-8")
    with pytest.raises(SsrmsdError, match="ensemble_id"):
        load_manifest(bad)
