from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from conftest import make_af3_job_dir, make_complex_chains, write_pdb
from typer.testing import CliRunner

from ssrmsd.cli import app
from ssrmsd.format import PAIRWISE_COLUMNS, SUMMARY_COLUMNS

runner = CliRunner()

EXPECTED_SUMMARY_COLUMNS = (
    "ensemble_id",
    "n_structures",
    "n_ca_align",
    "n_ca_score",
    "mean_sq_rmsd",
    "rms_rmsd",
    "mean_rmsd",
    "median_rmsd",
    "rmsd_sd",
    "ssRMSD",
    "SD",
    "within_source_mean_rmsd",
    "cross_source_mean_rmsd",
    "sources",
    "status",
    "error",
)


def test_cli_score_csv_tsv_jsonl_and_pairwise_out(tmp_path: Path) -> None:
    af3_dir = make_af3_job_dir(tmp_path / "af3", "mab1", n_samples=3)
    esm_dir = tmp_path / "esm"
    for k in range(2):
        write_pdb(
            esm_dir / f"esm_{k}.pdb",
            make_complex_chains(ab_shift=(4.0 + 0.1 * k, 0.0, 0.0), noise_seed=50 + k),
        )

    csv_out = tmp_path / "out.csv"
    pairs_out = tmp_path / "pairs.csv"
    res = runner.invoke(
        app,
        [
            "score",
            f"af3={af3_dir}",
            f"esmfold={esm_dir}",
            "--id",
            "mab1",
            "--align-chains",
            "A",
            "--score-chains",
            "B,C",
            "-o",
            str(csv_out),
            "--pairwise-out",
            str(pairs_out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert SUMMARY_COLUMNS == EXPECTED_SUMMARY_COLUMNS

    with csv_out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert tuple(rows[0].keys()) == EXPECTED_SUMMARY_COLUMNS
    assert rows[0]["ensemble_id"] == "mab1"
    assert rows[0]["n_structures"] == "5"
    assert rows[0]["sources"] == "af3:3,esmfold:2"
    assert rows[0]["status"] == "ok"
    assert len(rows[0]["mean_sq_rmsd"].split(".")[-1]) <= 4

    with pairs_out.open("r", encoding="utf-8", newline="") as f:
        prows = list(csv.DictReader(f))
    assert tuple(prows[0].keys()) == PAIRWISE_COLUMNS
    assert len(prows) == 10

    # TSV format
    tsv_out = tmp_path / "out.tsv"
    res_tsv = runner.invoke(
        app,
        ["score", str(af3_dir), "--format", "tsv", "-o", str(tsv_out)],
    )
    assert res_tsv.exit_code == 0, res_tsv.output
    assert "\t" in tsv_out.read_text(encoding="utf-8").splitlines()[0]

    # JSONL format
    jsonl_out = tmp_path / "out.jsonl"
    res_jsonl = runner.invoke(
        app,
        ["score", str(af3_dir), "--format", "jsonl", "-o", str(jsonl_out)],
    )
    assert res_jsonl.exit_code == 0, res_jsonl.output
    rec = json.loads(jsonl_out.read_text(encoding="utf-8").strip())
    assert rec["ensemble_id"] == "mab1"
    assert rec["status"] == "ok"
    assert rec["within_source_mean_rmsd"] is None


def test_cli_score_exits_1_on_error_row(tmp_path: Path) -> None:
    single = write_pdb(tmp_path / "only.pdb", make_complex_chains())
    out = tmp_path / "err.csv"
    res = runner.invoke(app, ["score", str(single), "-o", str(out)])
    assert res.exit_code == 1
    with out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "fewer than 2 structures" in rows[0]["error"]


def test_cli_batch_dirs_union_intersection_and_exit_codes(tmp_path: Path) -> None:
    af3_root = tmp_path / "af3"
    esm_root = tmp_path / "esm"
    make_af3_job_dir(af3_root, "mab1", n_samples=2)
    make_af3_job_dir(af3_root, "mab2", n_samples=1)  # will fail with <2 structures
    for k in range(2):
        write_pdb(esm_root / f"mab1_model_{k}.pdb", make_complex_chains(noise_seed=k))

    # Intersection: only mab1 -> exit 0
    inter_out = tmp_path / "inter.csv"
    res_inter = runner.invoke(
        app,
        [
            "batch",
            "--dir",
            f"af3={af3_root}",
            "--dir",
            f"esm={esm_root}",
            "--join",
            "intersection",
            "--workers",
            "1",
            "-o",
            str(inter_out),
        ],
    )
    assert res_inter.exit_code == 0, res_inter.output
    with inter_out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["ensemble_id"] for r in rows] == ["mab1"]
    assert rows[0]["status"] == "ok"

    # Union without --fail-fast: mab2 errors, still exits 0
    union_out = tmp_path / "union.csv"
    res_union = runner.invoke(
        app,
        [
            "batch",
            "--dir",
            f"af3={af3_root}",
            "--dir",
            f"esm={esm_root}",
            "--join",
            "union",
            "--workers",
            "1",
            "-o",
            str(union_out),
        ],
    )
    assert res_union.exit_code == 0, res_union.output
    with union_out.open("r", encoding="utf-8", newline="") as f:
        urows = list(csv.DictReader(f))
    assert {r["ensemble_id"]: r["status"] for r in urows} == {"mab1": "ok", "mab2": "error"}

    # Union with --fail-fast: exits non-zero
    res_ff = runner.invoke(
        app,
        [
            "batch",
            "--dir",
            f"af3={af3_root}",
            "--join",
            "union",
            "--workers",
            "1",
            "--fail-fast",
        ],
    )
    assert res_ff.exit_code != 0


def test_cli_batch_manifest_and_mutual_exclusion(tmp_path: Path) -> None:
    d = make_af3_job_dir(tmp_path / "af3", "mab1", n_samples=3)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"ensemble_id,path,source\nmab1,{d},af3\n", encoding="utf-8")

    out = tmp_path / "m_out.csv"
    res_ok = runner.invoke(
        app,
        ["batch", "--manifest", str(manifest), "--workers", "1", "-o", str(out)],
    )
    assert res_ok.exit_code == 0, res_ok.output
    with out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"

    # Mutual exclusion: --dir and --manifest together
    res_both = runner.invoke(
        app,
        ["batch", "--dir", str(tmp_path / "af3"), "--manifest", str(manifest)],
    )
    assert res_both.exit_code != 0

    # Zero discovered ensembles -> non-zero exit
    empty = tmp_path / "empty_root"
    empty.mkdir()
    res_empty = runner.invoke(app, ["batch", "--dir", str(empty)])
    assert res_empty.exit_code != 0


def test_cli_console_entrypoint() -> None:
    exe = Path(sys.executable).parent / "ssrmsd"
    proc = subprocess.run([str(exe), "--help"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "score" in proc.stdout
    assert "batch" in proc.stdout


def test_cli_version_flag() -> None:
    from ssrmsd import __version__

    res_long = runner.invoke(app, ["--version"])
    assert res_long.exit_code == 0, res_long.output
    assert res_long.stdout.strip() == f"ssrmsd {__version__}"

    res_short = runner.invoke(app, ["-V"])
    assert res_short.exit_code == 0, res_short.output
    assert res_short.stdout.strip() == f"ssrmsd {__version__}"
