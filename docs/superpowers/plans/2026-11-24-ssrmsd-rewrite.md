# `ssrmsd` Package Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `brineylab/ssrmsd` (on branch `rewrite`) with a clean, CPU-only Python package and CLI (`ssrmsd`) that computes normalized and legacy ssRMSD for single or batched multi-model antibody–antigen structure ensembles and drops straight into a slim `brineylab/toolshed` container.

**Architecture:** Six focused modules under `src/ssrmsd/`: dataclasses (`types.py`), `gemmi`-backed structure/archive/batch/manifest I/O (`io.py`), sequence-based cross-model chain matching and ensemble-wide Cα alignment (`align.py`), float64 Kabsch SVD and metric aggregation (`kabsch.py`), parallel ensemble/batch orchestration (`score.py`), and tabular/JSONL formatting plus `argparse` CLI (`format.py`, `cli.py`). Keeping chain matching separate from Kabsch leaves homomer symmetry permutation easy to plug in later.

**Tech Stack:** Python 3.11+, `hatchling`, `gemmi`, `numpy`, stdlib `argparse`/`tarfile`/`zipfile`/`gzip`/`csv`/`json`/`concurrent.futures`; dev tools: `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-11-24-ssrmsd-rewrite-design.md`

## Global Constraints

- Repository: `brineylab/ssrmsd` (`rewrite` branch); replace legacy `ssRMSD.py`, preserve `LICENSE` (MIT).
- Python 3.11+, `from __future__ import annotations`, full type annotations, `pathlib.Path` (no `os.path`).
- Runtime dependencies strictly limited to `gemmi` and `numpy` (stdlib `argparse`, no BioPython, no pandas, no GPU).
- Zero runtime network access; archives (`.tar.gz`, `.tgz`, `.tar`, `.zip`) streamed in memory with no temp-file extraction.
- Deterministic outputs regardless of filesystem glob order or `ProcessPoolExecutor` completion order.
- Legacy `ssRMSD` (`np.sum(M ** 2)`) and `SD` (`np.std(M, ddof=0)`) match the direct-`.cif` AF3 worker `compute_ssrmsd` within `1e-4` on 5-sample AF3 ensembles.
- Tabular outputs round numeric metrics to 4 decimal places; exact 16 summary columns: `ensemble_id, n_structures, n_ca_align, n_ca_score, mean_sq_rmsd, rms_rmsd, mean_rmsd, median_rmsd, rmsd_sd, ssRMSD, SD, within_source_mean_rmsd, cross_source_mean_rmsd, sources, status, error`.

## Review Focus

1. **Literal paths containing `=` vs. `label=path` syntax:** A directory like `run_lr=1e-4/job1` must be treated as an unlabeled path (not label `run_lr`), whereas `af3=/data/job1` splits into `("af3", Path("/data/job1"))`. (Pinned in Task 2.)
2. **Modified polymer residues (`MSE`, `SEP`, `UNK`) and residues missing `CA`:** Modified residues map to 1-letter codes via `gemmi` and residues lacking a `CA` atom are skipped without shifting sequence/coordinate indices. (Pinned in Task 2.)
3. **Identical structures ($d_{ij} = 0$) in Kabsch SVD:** Floating-point noise when `P == Q` must clamp squared residual sums at `0.0` so RMSD is exact `0.0`, never `NaN`. (Pinned in Task 4.)
4. **Multi-source ensemble with 0 within-source pairs (e.g., 1 `af3` + 1 `esmfold`):** `cross_source_mean_rmsd` is computed while `within_source_mean_rmsd` is `None` (empty in CSV/TSV), never `NaN` or `ZeroDivisionError`. (Pinned in Task 4.)
5. **Corrupt/truncated archive or malformed structure inside a batch:** Must emit a `status=error` row with a diagnostic message rather than crashing `ProcessPoolExecutor` or aborting other ensembles. (Pinned in Task 5.)

---

## File Structure

```text
ssrmsd/
├── LICENSE                         # Preserved (MIT)
├── README.md                       # Rewritten (CLI, Python API, toolshed Containerfile & SKILL.md recipe)
├── pyproject.toml                  # hatchling build, deps: gemmi, numpy; CLI entrypoint `ssrmsd`
├── src/ssrmsd/
│   ├── __init__.py                 # Public API exports
│   ├── types.py                    # Core dataclasses and errors
│   ├── io.py                       # Structure parsing, in-memory archive reading, AF3 detection, batch/manifest loading
│   ├── align.py                    # Linker splitting, ref selection, chain matching, homomer guard, Cα intersection
│   ├── kabsch.py                   # Float64 Kabsch SVD, pairwise matrix, normalized/legacy/multi-source metrics
│   ├── score.py                    # score_ensemble(), score_batch() with ProcessPoolExecutor
│   ├── format.py                   # CSV/TSV/JSONL writers for summary and pairwise outputs
│   └── cli.py                      # argparse CLI (`ssrmsd score`, `ssrmsd batch`) and exit-code contract
└── tests/
    ├── conftest.py                 # Synthetic mmCIF/PDB/archive/AF3-dir fixture builders
    ├── test_types.py               # Smoke & dataclass tests
    ├── test_io.py                  # Parsing, archives, AF3 dup filtering, discovery, manifests
    ├── test_align.py               # Linker split, ref selection, chain matching, coverage/identity, homomer guard
    ├── test_kabsch.py              # Kabsch math, subset align/score, legacy parity, multi-source metrics
    ├── test_score.py               # End-to-end single/batch scoring, determinism, mixed models, error resilience
    └── test_cli.py                 # Subcommands, output formats, --pairwise-out, exit codes
```

---

### Task 1: Package Scaffolding, Core Types, and Legacy Cleanup

**Files:**
- Delete: `ssRMSD.py`
- Create: `pyproject.toml`
- Create: `src/ssrmsd/types.py`
- Create: `src/ssrmsd/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_types.py`

**Interfaces:**
- Consumes: none.
- Produces in `src/ssrmsd/types.py`:
  - `class SsrmsdError(ValueError)` — domain error caught per-ensemble into `status="error"`.
  - `@dataclass(frozen=True) class ChainData`: `chain_id: str`, `sequence: str`, `ca_coords: np.ndarray` (shape `(L, 3)`, `float64`).
  - `@dataclass(frozen=True) class StructureData`: `name: str`, `source: str | None`, `chains: tuple[ChainData, ...]`, `was_split: bool = False`.
  - `@dataclass(frozen=True) class StructureInput`: `path: Path`, `source: str | None = None`.
  - `@dataclass(frozen=True) class ScoringOptions`: `align_chains: tuple[str, ...] | None = None`, `score_chains: tuple[str, ...] | None = None`, `ref_source: str | None = None`, `glob_pattern: str | None = None`, `include_top_level: bool = False`, `split_models: bool = False`, `min_chain_id: float = 0.90`, `min_coverage: float = 0.80`, `linker_min_len: int = 10`.
  - `@dataclass(frozen=True) class PairwiseDistance`: `ensemble_id: str`, `struct_i: str`, `struct_j: str`, `source_i: str | None`, `source_j: str | None`, `rmsd: float`.
  - `@dataclass(frozen=True) class EnsembleSpec`: `ensemble_id: str`, `inputs: tuple[StructureInput, ...]`.
  - `@dataclass(frozen=True) class EnsembleResult`: exact fields for all 16 summary columns (`ensemble_id: str`, `n_structures: int`, `n_ca_align: int | None`, `n_ca_score: int | None`, `mean_sq_rmsd: float | None`, `rms_rmsd: float | None`, `mean_rmsd: float | None`, `median_rmsd: float | None`, `rmsd_sd: float | None`, `ssRMSD: float | None`, `SD: float | None`, `within_source_mean_rmsd: float | None`, `cross_source_mean_rmsd: float | None`, `sources: str`, `status: Literal["ok", "error"]`, `error: str`) plus `pairwise: tuple[PairwiseDistance, ...] = ()`.
- Produces in `tests/conftest.py`:
  - Helpers to generate valid synthetic `.cif` and `.pdb` strings/files from `(chain_id, sequence, coords)` tuples, apply rigid-body rotations/translations (`apply_transform`), build fake AF3 job dirs (`seed-1_sample-0` … `seed-1_sample-4` plus duplicate top-level `<job>_model.cif`), AlphaFold Server `.zip` archives, and `.tar.gz` archives.

- [ ] **Step 1: Remove `ssRMSD.py`, create `pyproject.toml`, and set up `.venv`**

Create `pyproject.toml` with `hatchling` build backend, `name = "ssrmsd"`, `version = "0.1.0"`, `requires-python = ">=3.11"`, `dependencies = ["gemmi>=0.6.5", "numpy>=1.26"]`, `[project.optional-dependencies] dev = ["pytest>=8.0", "ruff>=0.6"]`, `[project.scripts] ssrmsd = "ssrmsd.cli:main"`, `[tool.hatch.build.targets.wheel] packages = ["src/ssrmsd"]`, and `[tool.ruff] line-length = 100`. Remove `ssRMSD.py` via `git rm ssRMSD.py`. Create `.venv` and install editable with dev deps:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

(If `gemmi` lacks a wheel for system Python 3.14, create `.venv` with Python 3.12 via `micromamba`.)

- [ ] **Step 2: Write failing test in `tests/test_types.py`**

```python
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

def test_default_scoring_options_match_spec():
    opts = ScoringOptions()
    assert opts.min_chain_id == 0.90
    assert opts.min_coverage == 0.80
    assert opts.linker_min_len == 10
    assert opts.align_chains is None
    assert opts.score_chains is None
    assert opts.ref_source is None
    assert opts.include_top_level is False
    assert opts.split_models is False

def test_error_result_factory():
    res = EnsembleResult.from_error("mAb1", n_structures=1, error="fewer than 2 structures")
    assert res.status == "error"
    assert res.error == "fewer than 2 structures"
    assert res.ssRMSD is None
    assert res.mean_sq_rmsd is None
    assert res.pairwise == ()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssrmsd.types'`.

- [ ] **Step 4: Implement `src/ssrmsd/types.py`, minimal `src/ssrmsd/__init__.py`, and `tests/conftest.py`**

Implement all dataclasses listed in **Interfaces** in `src/ssrmsd/types.py`, including `EnsembleResult.from_error(ensemble_id: str, *, n_structures: int = 0, sources: str = "", error: str) -> EnsembleResult`. Add `tests/conftest.py` with clean builders (`write_cif`, `write_pdb`, `make_helix_coords`, `apply_rigid_transform`, `make_af3_job_dir`, `make_tar_gz`, `make_zip`) using `gemmi.Structure` so synthetic fixtures are guaranteed valid mmCIF/PDB files.

- [ ] **Step 5: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_types.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ssrmsd/__init__.py src/ssrmsd/types.py tests/conftest.py tests/test_types.py
git commit -m "chore: scaffold ssrmsd package, core types, and test fixtures"
```

---

### Task 2: Structure I/O, Archive Streaming, and Ensemble Discovery

**Files:**
- Create: `src/ssrmsd/io.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Consumes: `ChainData`, `EnsembleSpec`, `ScoringOptions`, `SsrmsdError`, `StructureData`, `StructureInput` from `ssrmsd.types`.
- Produces in `src/ssrmsd/io.py`:
  - `SUPPORTED_STRUCTURE_SUFFIXES: tuple[str, ...]` (`.cif`, `.mmcif`, `.pdb`, `.ent`, `.cif.gz`, `.mmcif.gz`, `.pdb.gz`, `.ent.gz`)
  - `SUPPORTED_ARCHIVE_SUFFIXES: tuple[str, ...]` (`.tar.gz`, `.tgz`, `.tar`, `.zip`)
  - `parse_labeled_path(raw: str) -> StructureInput`
  - `read_structures_from_bytes(data: bytes, filename: str, *, source: str | None = None, split_models: bool = False) -> list[StructureData]`
  - `load_ensemble_structures(inputs: Sequence[StructureInput], options: ScoringOptions) -> list[StructureData]`
  - `discover_batch_dirs(roots: Sequence[StructureInput], *, join: Literal["union", "intersection"] = "union") -> list[EnsembleSpec]`
  - `load_manifest(path: Path, *, delimiter: str | None = None) -> list[EnsembleSpec]`

- [ ] **Step 1: Write failing tests in `tests/test_io.py`**

Write tests covering:
- `test_parse_cif_pdb_and_gzipped_variants`: identical Cα coords (`float64`) and 1-letter sequences extracted from `.cif`, `.pdb`, `.cif.gz`, `.pdb.gz`; waters/ligands ignored; primary conformer (`altloc` `""`/`"A"`) selected when alternate conformers exist.
- `test_modified_residues_and_missing_ca_atom` *(Review Focus #2)*: `MSE` maps to `M`, unknown polymer residue maps to `X`, and a residue missing a `CA` atom is skipped so `len(chain.sequence) == len(chain.ca_coords)`.
- `test_multi_model_default_vs_split_models`: 2-model PDB returns 1 `StructureData` by default (model 0) and 2 `StructureData` entries (`name#model0`, `name#model1`) with `split_models=True`.
- `test_af3_dir_and_tar_gz_skip_top_level_duplicate`: AF3 directory and `.tar.gz` containing 5 `seed-1_sample-{0..4}/*_model.cif` + 1 top-level `mab1_model.cif` yield exactly 5 structures by default, and 6 when `include_top_level=True`; explicit `glob_pattern` overrides auto-detection.
- `test_af3_server_zip_in_memory`: `.zip` containing `mab1_model_0.cif` … `mab1_model_4.cif` yields 5 structures in deterministic sorted order.
- `test_parse_labeled_path_handles_equals_in_dirname` *(Review Focus #1)*: `"af3=/tmp/x"` -> `StructureInput(Path("/tmp/x"), "af3")`; existing path `"run_lr=1e-4/job1"` or `"dir/a=b.cif"` -> `StructureInput(Path(...), None)`.
- `test_discover_batch_dirs_union_and_intersection`: two roots (`af3=...`, `esm=...`) with child dirs, `.tar.gz`, `.zip`, and flat `<mab>_model_<k>.pdb` files strip extensions/suffixes cleanly, merge matching `ensemble_id`s, and filter properly under `join="union"` vs `join="intersection"`.
- `test_load_manifest_csv_and_tsv`: groups multiple rows sharing `ensemble_id` into one `EnsembleSpec` preserving optional `source`; raises `SsrmsdError` if required columns (`ensemble_id`, `path`) are missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssrmsd.io'`.

- [ ] **Step 3: Implement `src/ssrmsd/io.py`**

Key implementation rules:
- Decompress `.gz` via `gzip.decompress` in memory; parse mmCIF via `gemmi.cif.read_string` + `gemmi.make_structure_from_block(doc.sole_block())` and PDB via `gemmi.read_pdb_string`.
- Extract polymer Cα via `chain.get_polymer()` (or iterate polymer residues using `gemmi.find_tabulated_residue(res.name)`); take `res.find_atom("CA", "*")` on `first_conformer()` so `altloc` `" "`/`"A"` is chosen deterministically; skip residues where `CA` is `None`; map residue name via `gemmi.find_tabulated_residue(res.name).one_letter_code.upper()` (fallback `"X"` for blank/`"?"`).
- Stream `.tar.gz`/`.tgz`/`.tar` via `tarfile.open(..., "r:*")` and `.zip` via `zipfile.ZipFile`; sort member paths deterministically before reading.
- AF3 detection: inside a directory or archive, if any member matches `seed-*_sample-*/*` (using `re.search(r"(^|/)seed-[^/]+_sample-\d+/", relpath)`), filter to those members unless `include_top_level=True` or `glob_pattern` is provided.
- `parse_labeled_path(raw)`: if `"="` in `raw` and `not Path(raw).exists()` and `"/" not in raw.split("=", 1)[0]`, split into `(label, Path(rest))`; otherwise `(None, Path(raw))`.
- Wrap corrupt archives / unparseable structure exceptions in `SsrmsdError` with the offending filename.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_io.py -v`
Expected: PASS (all tests in `tests/test_io.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ssrmsd/io.py tests/test_io.py
git commit -m "feat: add structure parser, archive streaming, and batch/manifest discovery"
```

---

### Task 3: Cross-Model Chain Matching and Residue Alignment

**Files:**
- Create: `src/ssrmsd/align.py`
- Create: `tests/test_align.py`

**Interfaces:**
- Consumes: `ChainData`, `ScoringOptions`, `SsrmsdError`, `StructureData` from `ssrmsd.types`.
- Produces in `src/ssrmsd/align.py`:
  - `@dataclass(frozen=True) class AlignedEnsemble`:
    - `names: tuple[str, ...]`
    - `sources: tuple[str | None, ...]`
    - `align_coords: tuple[np.ndarray, ...]` — length $n$, each `(N_align, 3)` `float64`
    - `score_coords: tuple[np.ndarray, ...]` — length $n$, each `(N_score, 3)` `float64`
    - `chain_mapping: tuple[dict[str, str], ...]` — per-structure `{ref_chain_id: struct_chain_id}` (clean boundary for future homomer permutation)
  - `align_ensemble(structures: Sequence[StructureData], options: ScoringOptions) -> AlignedEnsemble`

- [ ] **Step 1: Write failing tests in `tests/test_align.py`**

Write tests covering:
- `test_matches_reordered_and_renamed_chains_across_models`: structure 0 has chains `A (Ag), B (HC), C (LC)`; structure 1 has chains `L (LC), H (HC), X (Ag)`; `align_ensemble` maps `A->X, B->H, C->L` and produces matching coordinate shapes.
- `test_poly_g_linker_splitting_regardless_of_sort_order`: single-chain ESMFold-style structure (`Ag + "G"*20 + HC + "G"*20 + LC`) passed **before** a 3-chain AF3 structure splits cleanly before reference selection, selects the naturally 3-chain AF3 structure as default reference (preserving real chain IDs `A, B, C`), and aligns all 3 chains. Also verify `linker_min_len=0` disables splitting and fails with `SsrmsdError`.
- `test_strict_ref_source_error_when_missing`: passing `ref_source="af3"` when only `esmfold` structures are present raises `SsrmsdError` mentioning `ref-source`.
- `test_align_and_score_chain_subsets_and_spectator_chains`:
  - Default (neither flag): extra spectator chain `D` in structure 1 raises `SsrmsdError`.
  - With `align_chains=("A",), score_chains=("B", "C")`: extra spectator chain `D` in structure 1 is ignored, `align_coords` has only chain `A`'s Cαs, `score_coords` has `B`+`C`'s Cαs.
  - With only `align_chains=("A",)`: `score_coords` covers all reference chains.
  - With only `score_chains=("B", "C")`: `align_coords` covers all reference chains.
  - Unknown chain ID in `--align-chains`/`--score-chains` raises `SsrmsdError`.
- `test_terminal_truncation_overlap_identity_and_symmetric_coverage`:
  - Structure 1 truncated by 15% at N/C termini (85% coverage, 100% identity over overlap) passes default `min_chain_id=0.90` and `min_coverage=0.80`; `align_coords` uses the ensemble-wide intersection (85% of residues) for *all* structures.
  - Structure truncated by 50% fails `min_coverage=0.80` **even when sorted first as reference** (verifying coverage denominator is `max_chain_len`).
- `test_homomer_and_ambiguous_chain_guard`: reference containing two identical-sequence chains (`A` and `D`) raises `SsrmsdError` mentioning `"homomer"` / `"ambiguous"`; non-reference containing two copies tying above `min_chain_id` for a required reference chain also raises `SsrmsdError`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_align.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssrmsd.align'`.

- [ ] **Step 3: Implement `src/ssrmsd/align.py`**

Exact pipeline order (from Spec §2):
1. Compute `max_chains = max(len(s.chains) for s in structures)`. If `options.linker_min_len > 0`, split chains matching `re.compile(f"G{{{options.linker_min_len},}}")` on any structure with `len(s.chains) < max_chains`, marking `was_split=True` and assigning synthetic IDs (`A, B, C, ...` or `f"{cid}_{k}"`).
2. Reference selection: if `options.ref_source` is set, pick first structure with `s.source == options.ref_source` (raise `SsrmsdError` if none); else pick first structure with `not s.was_split and len(s.chains) == max_chains`, falling back to `structures[0]` (normalizing chain IDs to `A, B, C, ...` if all were split).
3. Resolve `align_ids`, `score_ids`, `required_ids = align_ids | score_ids`; validate every requested ID exists in reference chains. When neither `align_chains` nor `score_chains` is set, verify every structure has `len(s.chains) == len(ref.chains)`.
4. Pairwise sequence alignment helper using `gemmi.align_string_sequences(list(seq1), list(seq2), [])`: compute `overlap_identity = exact_matches / aligned_non_gap_pairs` and aligned index map `{ref_idx: target_idx}` from `alignment.cigar_str()` (walking `M`/`I`/`D` operations).
5. Homomer/ambiguity check: verify no pair of distinct chains in `required_ids` has `overlap_identity >= options.min_chain_id`; for each structure, verify no single required ref chain has `>= 2` candidate chains at `>= options.min_chain_id` (and solve optimal 1-to-1 assignment maximizing total overlap identity via pure-Python Hungarian / exact backtracking since chain counts are small, rejecting if any required chain scores `< options.min_chain_id`).
6. For each required chain, intersect valid aligned reference positions across all $n$ structures; check `len(common_positions) / max_chain_len >= options.min_coverage`; assemble concatenated `(N_align, 3)` and `(N_score, 3)` `float64` arrays per structure in canonical reference chain order.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_align.py -v`
Expected: PASS (all tests in `tests/test_align.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ssrmsd/align.py tests/test_align.py
git commit -m "feat: add cross-model chain matching, linker splitting, and CA alignment"
```

---

### Task 4: Kabsch Superposition and Metric Aggregation

**Files:**
- Create: `src/ssrmsd/kabsch.py`
- Create: `tests/test_kabsch.py`

**Interfaces:**
- Consumes: `AlignedEnsemble` from `ssrmsd.align`; `EnsembleResult`, `PairwiseDistance` from `ssrmsd.types`.
- Produces in `src/ssrmsd/kabsch.py`:
  - `kabsch_rmsd(p_align: np.ndarray, q_align: np.ndarray, p_score: np.ndarray | None = None, q_score: np.ndarray | None = None) -> float`
  - `compute_pairwise_matrix(aligned: AlignedEnsemble) -> np.ndarray` (symmetric `(n, n)` `float64`, zero diagonal)
  - `aggregate_metrics(ensemble_id: str, aligned: AlignedEnsemble, matrix: np.ndarray, *, include_pairwise: bool = False) -> EnsembleResult`

- [ ] **Step 1: Write failing tests in `tests/test_kabsch.py`**

Write tests covering:
- `test_kabsch_identical_and_rigid_transformed_is_zero` *(Review Focus #3)*: `kabsch_rmsd(P, P) == 0.0` (exact, no `NaN`); rigid rotation + translation gives `< 1e-12`; improper reflection (`det < 0`) is corrected rather than producing mirror-image zero RMSD.
- `test_kabsch_separate_align_and_score_sets`: keep chain A fixed and translate chain B by `(3.0, 4.0, 0.0)` (5.0 Å shift): full-complex RMSD `< 5.0`, whereas aligning on chain A and scoring on chain B gives exact `5.0` Å!
- `test_legacy_parity_with_af3_worker_formula`: generate 5 realistic 3-chain structures (varying poses), run the exact legacy AF3 worker `_kabsch_rmsd` / `np.sum(mat**2)` / `np.std(mat)` formula directly in the test, and assert `abs(res.ssRMSD - legacy_ssrmsd) <= 1e-4`, `abs(res.SD - legacy_sd) <= 1e-4`, and `res.mean_sq_rmsd == pytest.approx(res.ssRMSD / (5 * 4), abs=1e-4)`.
- `test_normalized_metrics_scale_invariance`: duplicating a tight ensemble's pairwise distance distribution from $n=5$ to $n=10$ scales legacy `ssRMSD` by `90/20 = 4.5x` while `mean_sq_rmsd` and `rms_rmsd` stay constant.
- `test_multi_source_within_and_cross_metrics`:
  - 3 `af3` (tight cluster) + 2 `esmfold` (shifted cluster): `within_source_mean_rmsd < cross_source_mean_rmsd`, `sources == "af3:3,esmfold:2"`.
  - 1 `af3` + 1 `esmfold` *(Review Focus #4)*: `cross_source_mean_rmsd` is populated and `within_source_mean_rmsd is None` (no `NaN`, no `ZeroDivisionError`).
  - Unlabeled structures: `within_source_mean_rmsd is None`, `cross_source_mean_rmsd is None`, `sources == ""`.
- `test_pairwise_records_emitted_when_requested`: `include_pairwise=True` returns `n*(n-1)//2` `PairwiseDistance` entries matching the upper triangle of `matrix`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kabsch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssrmsd.kabsch'`.

- [ ] **Step 3: Implement `src/ssrmsd/kabsch.py`**

Implement `kabsch_rmsd` in `float64` per Spec §3 (with `max(0.0, float((diff * diff).sum()))` clamp before `sqrt`). In `aggregate_metrics`, compute upper-triangle vector `u = matrix[np.triu_indices(n, k=1)]`, `mean_sq = float(np.mean(u ** 2))`, `rms = float(np.sqrt(mean_sq))`, `mean = float(np.mean(u))`, `median = float(np.median(u))`, `rmsd_sd = float(np.std(u, ddof=0))`, `legacy_ss = float(np.sum(matrix ** 2))`, `legacy_sd = float(np.std(matrix, ddof=0))`. Compute `within_source_mean_rmsd` / `cross_source_mean_rmsd` only when `len({s for s in aligned.sources if s is not None}) >= 2` (and only if the respective pair list is non-empty); format `sources` deterministically sorted by source name (`f"{k}:{v}"`). Store full-precision floats on `EnsembleResult` (4-decimal rounding happens at tabular formatting in `format.py`, matching Python API expectations).

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_kabsch.py -v`
Expected: PASS (all tests in `tests/test_kabsch.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ssrmsd/kabsch.py tests/test_kabsch.py
git commit -m "feat: add float64 Kabsch RMSD and normalized/legacy/multi-source metrics"
```

---

### Task 5: High-Level Scoring API & Parallel Batch Orchestration

**Files:**
- Create: `src/ssrmsd/score.py`
- Modify: `src/ssrmsd/__init__.py`
- Create: `tests/test_score.py`

**Interfaces:**
- Consumes: `load_ensemble_structures`, `parse_labeled_path` from `ssrmsd.io`; `align_ensemble` from `ssrmsd.align`; `compute_pairwise_matrix`, `aggregate_metrics` from `ssrmsd.kabsch`; `EnsembleResult`, `EnsembleSpec`, `ScoringOptions`, `SsrmsdError`, `StructureInput` from `ssrmsd.types`.
- Produces in `src/ssrmsd/score.py` (and re-exported from `src/ssrmsd/__init__.py`):
  - `score_ensemble(inputs: Sequence[StructureInput | Path | str], *, ensemble_id: str | None = None, options: ScoringOptions | None = None, include_pairwise: bool = False) -> EnsembleResult`
  - `score_batch(ensembles: Sequence[EnsembleSpec], *, options: ScoringOptions | None = None, workers: int = 1, include_pairwise: bool = False, fail_fast: bool = False) -> list[EnsembleResult]`

- [ ] **Step 1: Write failing tests in `tests/test_score.py`**

Write tests covering:
- `test_score_ensemble_mixed_af3_tar_gz_and_esmfold_pdb`: combines a 5-sample AF3 `.tar.gz` (chains `A, B, C` with top-level dup) and 2 ESMFold `.pdb` files (single-chain poly-G linker, 5-residue terminal truncation) into one 7-structure ensemble; returns `status == "ok"`, `n_structures == 7`, `sources == "af3:5,esmfold:2"`, non-empty `within_source_mean_rmsd` and `cross_source_mean_rmsd`.
- `test_score_ensemble_fewer_than_two_structures`: 0 or 1 structure returns `status == "error"` with `"fewer than 2 structures"` in `error`.
- `test_score_batch_multiprocess_determinism_and_error_isolation` *(Review Focus #5)*: batch of 6 ensembles including 1 corrupt `.tar.gz` and 1 single-structure ensemble run with `workers=2` returns all 6 results in exact input order, with 4 `status == "ok"` and 2 `status == "error"` rows (corrupt archive error mentions filename/archive).
- `test_score_batch_fail_fast_raises`: same batch with `fail_fast=True` raises `SsrmsdError`.
- `test_public_api_imports`: `from ssrmsd import score_ensemble, score_batch, EnsembleSpec, EnsembleResult, ScoringOptions, StructureInput` succeeds.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssrmsd.score'`.

- [ ] **Step 3: Implement `src/ssrmsd/score.py` and update `src/ssrmsd/__init__.py`**

In `score_ensemble`: normalize `inputs` (`str` via `parse_labeled_path`, `Path` via `StructureInput`), derive default `ensemble_id` from single input stem or `"ensemble"`, load structures, verify `len(structures) >= 2`, call `align_ensemble` -> `compute_pairwise_matrix` -> `aggregate_metrics`, catching `SsrmsdError` (and unexpected parse/I/O errors) into `EnsembleResult.from_error(...)`. In `score_batch`: run inline when `workers <= 1` or `len(ensembles) <= 1`, otherwise use `ProcessPoolExecutor(max_workers=workers)` with a top-level picklable worker function, preserving exact input order; if `fail_fast=True` and any result has `status == "error"`, raise `SsrmsdError(result.error)`. Re-export public symbols from `src/ssrmsd/__init__.py`.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: PASS (all tests in `tests/test_score.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ssrmsd/score.py src/ssrmsd/__init__.py tests/test_score.py
git commit -m "feat: add score_ensemble and parallel score_batch API"
```

---

### Task 6: Output Formatting, CLI, README, and Full Verification

**Files:**
- Create: `src/ssrmsd/format.py`
- Create: `src/ssrmsd/cli.py`
- Modify: `README.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `discover_batch_dirs`, `load_manifest`, `parse_labeled_path` from `ssrmsd.io`; `score_ensemble`, `score_batch` from `ssrmsd.score`; `EnsembleResult`, `PairwiseDistance`, `ScoringOptions`, `SsrmsdError` from `ssrmsd.types`.
- Produces in `src/ssrmsd/format.py`:
  - `SUMMARY_COLUMNS: tuple[str, ...]` (the exact 16 columns from Global Constraints)
  - `PAIRWISE_COLUMNS: tuple[str, ...]` (`"ensemble_id", "struct_i", "struct_j", "source_i", "source_j", "rmsd"`)
  - `write_results(results: Sequence[EnsembleResult], stream: TextIO, *, fmt: Literal["csv", "tsv", "jsonl"] = "csv") -> None`
  - `write_pairwise(results: Sequence[EnsembleResult], stream: TextIO, *, fmt: Literal["csv", "tsv", "jsonl"] = "csv") -> None`
- Produces in `src/ssrmsd/cli.py`:
  - `build_parser() -> argparse.ArgumentParser`
  - `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing tests in `tests/test_cli.py`**

Write tests covering:
- `test_cli_score_csv_tsv_jsonl_and_pairwise_out`: runs `main(["score", ...])` on a multi-model ensemble; checks exact 16 CSV headers, 4-decimal formatting, valid JSONL parsing, `--pairwise-out` file rows/columns, and exit code `0`.
- `test_cli_score_exits_1_on_error_row`: single structure input writes a `status=error` row to stdout/output file and returns exit code `1`.
- `test_cli_batch_dirs_union_intersection_and_exit_codes`: runs `main(["batch", "--dir", "af3=...", "--dir", "esm=...", "--join", "intersection", ...])`, verifies intersection filtering; verifies batch with one failing ensemble writes `status=error` and returns exit code `0` without `--fail-fast`, and returns non-zero with `--fail-fast`.
- `test_cli_batch_manifest_and_mutual_exclusion`: runs `main(["batch", "--manifest", ...])`; verifies passing both `--dir` and `--manifest`, or zero discovered ensembles, exits non-zero.
- `test_cli_console_entrypoint`: invokes `.venv/bin/ssrmsd --help` via `subprocess.run` and asserts exit code `0` with `score` and `batch` subcommands listed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssrmsd.cli'`.

- [ ] **Step 3: Implement `src/ssrmsd/format.py`, `src/ssrmsd/cli.py`, and rewrite `README.md`**

- `format.py`: round float metrics with `round(val, 4)` (empty string `""` for `None` in CSV/TSV; `null` in JSONL); emit exact `SUMMARY_COLUMNS` and `PAIRWISE_COLUMNS`.
- `cli.py`: `argparse` subcommands `score` and `batch` with every option from Spec §4 (`--id`, `--dir`, `--join`, `--manifest`, `--delimiter`, `--align-chains`, `--score-chains`, `--ref-source`, `--glob`, `--include-top-level`, `--split-models`, `--min-chain-id`, `--min-coverage`, `--linker-min-len`, `--format`, `-o/--output`, `--pairwise-out`, `--workers`, `--fail-fast`). Implement exact exit-code contract from Spec §4.4.
- `README.md`: replace legacy docs with concise package overview, installation, `ssrmsd score` / `ssrmsd batch` CLI examples, output column table, Python API snippet, and the downstream `brineylab/toolshed` `Containerfile` + `SKILL.md` recipe.

- [ ] **Step 4: Run full test suite and linter**

Run:
```bash
.venv/bin/python -m pytest -v
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```
Expected: all tests PASS, `ruff check` and `ruff format --check` clean with 0 issues.

- [ ] **Step 5: Commit**

```bash
git add src/ssrmsd/format.py src/ssrmsd/cli.py tests/test_cli.py README.md
git commit -m "feat: add CLI, output formatters, and updated README"
```
