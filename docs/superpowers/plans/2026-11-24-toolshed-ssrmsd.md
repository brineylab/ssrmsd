# `toolshed` Tool for `ssRMSD` (and `ssRMSD v0.1.1`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release `ssRMSD v0.1.1` (adding flat `*_sample_<k>` batch grouping and `ssrmsd --version`) to PyPI and package it as `ghcr.io/brineylab/toolshed/ssrmsd:1.0.0` in `brineylab/toolshed` with a threshold-crossing offline test and batch-oriented `SKILL.md`.

**Architecture:** Cut a minimal `v0.1.1` patch via PR on `brineylab/ssrmsd` so all five structure predictors in `brineylab/toolshed` (`alphafold3-blackwell`, `boltz-2`, `esmfold2`/`esmfold2-fast`, `openfold3`, `opendde`) drop directly into `ssrmsd batch --dir`. Once the user merges the PR, tag/publish the `v0.1.1` GitHub Release to PyPI, then build a slim digest-pinned `python:3.12-slim-bookworm` container under `brineylab/toolshed/tools/ssrmsd/` installing pinned `ssRMSD==0.1.1` from PyPI with Pyxis `python3 -I` isolation, an offline multi-ensemble `test/run.sh`, and a prose `SKILL.md`.

**Tech Stack:** Python 3.12, Typer, Gemmi, NumPy, Pytest/Ruff/ty, Podman (`linux/amd64`), `brineylab/toolshed` (`scripts/build`, `scripts/push`, `scripts/test`, `scripts/check.py`), GitHub CLI (`gh`).

**Spec:** `docs/superpowers/specs/2026-11-24-toolshed-ssrmsd-design.md`

## Global Constraints

- `brineylab/ssrmsd` `v0.1.1` goes through a PR (never direct push to `main`); CI checks `ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/`, and `pytest` must all pass.
- Once the user merges the `v0.1.1` PR, tagging `v0.1.1` and publishing the GitHub Release (triggering `.github/workflows/python-publish.yaml` to PyPI) is pre-authorized with no extra checkpoint.
- `brineylab/toolshed` tool directory is `tools/ssrmsd/`, `SKILL.md` frontmatter `name: ssrmsd`, `metadata.version: 1.0.0` (immutable semver), body sections in order: `Image:`, `## Commands`, `## Inputs and outputs`, `## Compute`, `## Running it`, `## Gotchas`, every command example a full command line.
- `Containerfile` uses digest-pinned `docker.io/library/python:3.12-slim-bookworm@sha256:...`, pins `ssRMSD==0.1.1` and every transitive Python dependency at `==`, has no `ENTRYPOINT`, no `bin/` wrapper, no GPU env vars, no credential `ARG`/`ENV`, sets `ENV PYTHONNOUSERSITE=1` and patches `/usr/local/bin/ssrmsd` shebang to `#!/usr/local/bin/python3 -I`, copies MIT `LICENSE` to `/opt/ssrmsd/LICENSE`, and ends with `COPY test /toolshed/test`.
- `tools/ssrmsd/test/run.sh` passes under `--network none`, writes only under `/tmp` (cleaning up on exit), crosses `score_batch`'s `ProcessPoolExecutor` threshold (`len(ensembles) > 1` and `--workers > 1`), and checks analytical RMSDs within `1e-4`.
- Ask the user before running `scripts/push ssrmsd` (which triggers `seed.yml` and pushes to GHCR) or opening the PR on `brineylab/toolshed`.

## Review Focus

- Flat `*_sample_<k>` filenames with `.cif.gz`/`.pdb.gz` double extensions, multi-digit indices (`_sample_10`), and underscores in target names (`mab_v2_sample_0.cif`) must group under `mab_v2` without truncating target IDs. (Pinned in Task 1.)
- Non-structure sidecar files (`<target>_sample_0.json`, `<target>_sample_0.npz`) produced by `esmfold2` beside `.cif` outputs must be ignored by `discover_batch_dirs` without creating bogus ensembles or errors. (Pinned in Task 1 and Task 3.)
- Pyxis environment and `PATH` leakage (`PYTHONPATH` holding a shadowing `ssrmsd.py` module and stripped default `PATH=/usr/local/bin:/usr/bin:/bin`) must not break `/usr/local/bin/ssrmsd` or shadow the container's packages. (Pinned in Task 3.)
- Invoking `bash /toolshed/test/run.sh` when `/toolshed/test` and the caller's working directory are read-only (as under Pyxis) must succeed writing strictly under `/tmp`. (Pinned in Task 3.)
- `ssrmsd batch --manifest` under `--workers 2` with a mix of valid multi-source ensembles and a homomer error ensemble must emit both `ok` and `error` JSONL rows in input order and exit `0`, while `ssrmsd score` on that homomer exits `1`. (Pinned in Task 3.)

---

### Task 1: `ssRMSD v0.1.1` Patch (`*_sample_<k>` Grouping, `--version`, Docs) and PR

**Files:**
- Modify: `src/ssrmsd/io.py:44-46`
- Modify: `src/ssrmsd/__init__.py:1-26`
- Modify: `src/ssrmsd/cli.py:33-48`
- Modify: `README.md:45-115`
- Test: `tests/test_io.py`, `tests/test_cli.py`
- Commit alongside: `docs/superpowers/specs/2026-11-24-toolshed-ssrmsd-design.md`, `docs/superpowers/plans/2026-11-24-toolshed-ssrmsd.md`

**Interfaces:**
- Consumes: existing `discover_batch_dirs`, `strip_known_suffix`, `app` (`typer.Typer`), `tests/conftest.py` helpers.
- Produces:
  - `ssrmsd.__version__: str`
  - `ssrmsd --version` / `ssrmsd -V` CLI option printing `ssrmsd <version>` and exiting `0`
  - `_MODEL_GROUP_RE` matching both `<id>_model_<k>` and `<id>_sample_<k>` in `discover_batch_dirs`

- [ ] **Step 1: Create branch `release/0.1.1` from `main`**

```bash
git checkout -b release/0.1.1
```

- [ ] **Step 2: Write failing tests for flat `*_sample_<k>` grouping and `--version` / `-V`**

In `tests/test_io.py`, add `test_discover_batch_dirs_groups_esmfold2_sample_files_and_ignores_sidecars(tmp_path: Path) -> None`:
- Creates `af3_root` with `make_af3_job_dir(af3_root, "mab_v2", n_samples=2)`.
- Creates `esm_root` with `mab_v2_sample_0.cif`, `mab_v2_sample_1.cif.gz`, `mab_v2_sample_10.pdb.gz`, plus sidecars `mab_v2_sample_0.json` and `mab_v2_sample_0.npz`.
- Asserts `discover_batch_dirs([StructureInput(af3_root, "af3"), StructureInput(esm_root, "esmfold2")], join="intersection")` returns `[spec]` with `spec.ensemble_id == "mab_v2"`, `len(spec.inputs) == 4` (1 AF3 dir + 3 ESM structure files, sidecars excluded), and `{inp.source for inp in spec.inputs} == {"af3", "esmfold2"}`.

In `tests/test_cli.py`, add `test_cli_version_flag() -> None`:
- Imports `__version__` from `ssrmsd`.
- Invokes `runner.invoke(app, ["--version"])` and `runner.invoke(app, ["-V"])`; asserts both exit `0` and `res.stdout.strip() == f"ssrmsd {__version__}"`.

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/pytest tests/test_io.py::test_discover_batch_dirs_groups_esmfold2_sample_files_and_ignores_sidecars tests/test_cli.py::test_cli_version_flag -v`
Expected: FAIL (`_sample_` files not grouped under `mab_v2`; `__version__` / `--version` not defined).

- [ ] **Step 4: Implement `_MODEL_GROUP_RE`, `__version__`, and `--version` / `-V`**

- In `src/ssrmsd/io.py:45`: update `_MODEL_GROUP_RE = re.compile(r"^(.+)_(?:model|sample)_\d+$")`.
- In `src/ssrmsd/__init__.py`: expose `__version__: str` via `importlib.metadata.version("ssRMSD")` catching `PackageNotFoundError` -> `"0.0.0+unknown"`, and add `"__version__"` to `__all__`.
- In `src/ssrmsd/cli.py`: add `@app.callback()` with `version: Annotated[bool, typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit.")] = False` where `_version_callback(value: bool) -> None` echoes `f"ssrmsd {__version__}"` and raises `typer.Exit(code=0)` when `value` is True.
- In `README.md`: update the batch comment (`<mab>_{model,sample}_<k>`) and the toolshed image reference (`ghcr.io/brineylab/toolshed/ssrmsd:1.0.0` / `ssRMSD==0.1.1`).

- [ ] **Step 5: Run full local CI checks**

Run: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/ && .venv/bin/ty check src/ && .venv/bin/pytest -v`
Expected: all checks pass with 0 errors and all tests green.

- [ ] **Step 6: Commit, push branch, and open PR on `brineylab/ssrmsd`**

```bash
git add docs/superpowers/specs/2026-11-24-toolshed-ssrmsd-design.md docs/superpowers/plans/2026-11-24-toolshed-ssrmsd.md src/ssrmsd/io.py src/ssrmsd/__init__.py src/ssrmsd/cli.py tests/test_io.py tests/test_cli.py README.md
git commit -m "feat: group flat *_sample_<k> batch files and add --version flag"
git push -u origin release/0.1.1
gh pr create --title "v0.1.1: group flat *_sample_<k> batch files and add --version" --body "..."
```
Report the PR link to the user and wait for them to merge it before proceeding to Task 2.

---

### Task 2: Cut `v0.1.1` GitHub Release and Verify PyPI Publication

**Files:**
- None (git tag + GitHub Release on `brineylab/ssrmsd`)

**Interfaces:**
- Consumes: merged `release/0.1.1` PR on `brineylab/ssrmsd` `main`, `.github/workflows/python-publish.yaml`.
- Produces: live `ssRMSD==0.1.1` wheel/sdist on PyPI.

- [ ] **Step 1: Sync `main` after user merges the PR**

```bash
git checkout main && git pull --ff-only origin main
```

- [ ] **Step 2: Tag `v0.1.1` and publish GitHub Release (pre-authorized)**

```bash
git tag -a v0.1.1 -m "v0.1.1"
git push origin v0.1.1
gh release create v0.1.1 --title "v0.1.1" --generate-notes
```

- [ ] **Step 3: Wait for `Publish to PyPI` workflow and verify `ssRMSD==0.1.1` on PyPI**

Run: `gh run watch` (on the `Publish to PyPI` workflow run) and verify `https://pypi.org/pypi/ssRMSD/0.1.1/json` returns HTTP 200 with wheel URLs.
Expected: workflow concludes `success` and `ssRMSD==0.1.1` is live on PyPI.

---

### Task 3: Author `tools/ssrmsd/` in `brineylab/toolshed`, Build Candidate, and Verify Offline

**Files:**
- Create (in `brineylab/toolshed` checkout on branch `add-ssrmsd`):
  - `tools/ssrmsd/Containerfile`
  - `tools/ssrmsd/LICENSE`
  - `tools/ssrmsd/SKILL.md`
  - `tools/ssrmsd/test/make_fixture.py`
  - `tools/ssrmsd/test/fixture/...` (generated kilobyte fixture + `expected.json`)
  - `tools/ssrmsd/test/run.sh`

**Interfaces:**
- Consumes: `ssRMSD==0.1.1` on PyPI, `brineylab/toolshed` `scripts/check.py` and `scripts/build`.
- Produces: committed `tools/ssrmsd/` passing `scripts/check.py`, local `linux/amd64` candidate image `ghcr.io/brineylab/toolshed/ssrmsd:src-<tree>` passing `podman run --rm --network none <candidate> bash /toolshed/test/run.sh`.

- [ ] **Step 1: Prepare `brineylab/toolshed` checkout and branch `add-ssrmsd`**

Clone `https://github.com/brineylab/toolshed.git` to `/home/bryanbriney/git/toolshed` (or fetch/branch if already present), then `git checkout -b add-ssrmsd`.

- [ ] **Step 2: Resolve exact pinned base digest and CPython 3.12 `linux/amd64` package versions for `ssRMSD==0.1.1`**

Inspect the pinned `docker.io/library/python:3.12-slim-bookworm@sha256:...` digest (matching `tools/datascience/Containerfile` or fresh digest) and resolve the exact `pip freeze` set for `ssRMSD==0.1.1` inside that base image (`podman run --rm --platform linux/amd64 ... pip install --no-cache-dir ssRMSD==0.1.1 && pip freeze`) so every direct and transitive package (`ssRMSD`, `gemmi`, `numpy`, `typer`, `click`, `typing-extensions`, etc.) is pinned with `==`.

- [ ] **Step 3: Write `tools/ssrmsd/LICENSE` and `tools/ssrmsd/Containerfile`**

- Copy `/home/bryanbriney/git/ssrmsd/LICENSE` to `tools/ssrmsd/LICENSE`.
- Write `tools/ssrmsd/Containerfile`:
  - `FROM docker.io/library/python:3.12-slim-bookworm@sha256:<resolved-digest>`
  - `RUN pip install --no-cache-dir ssRMSD==0.1.1 <every pinned dep==ver>`
  - `ENV PYTHONNOUSERSITE=1`
  - `RUN sed -i '1s|.*|#!/usr/local/bin/python3 -I|' /usr/local/bin/ssrmsd`
  - `COPY LICENSE /opt/ssrmsd/LICENSE`
  - `RUN ssrmsd --version | grep -Fx 'ssrmsd 0.1.1'`
  - `COPY test /toolshed/test` (must be last line)

- [ ] **Step 4: Write `tools/ssrmsd/test/make_fixture.py`, generate `tools/ssrmsd/test/fixture/`, and write `tools/ssrmsd/test/run.sh`**

- `make_fixture.py` deterministically builds a kilobyte-scale synthetic antibody–antigen fixture (3 distinct non-homomeric chains `A`, `B`, `C`, 25–30 residues each, exact rigid-body translations of `B`+`C` relative to `A` so analytical Cα RMSDs under `--align-chains A --score-chains B,C` equal the shift magnitudes identically):
  - `fixture/af3/mab{1,2,3}/seed-1_sample-{0,1,2}/model.cif` + a large-shift decoy `fixture/af3/mab{1,2,3}/mab{1,2,3}_model.cif` that MUST be excluded.
  - `fixture/esmfold2/mab{1,2,3}_sample_{0,1}.cif` (with permuted chain IDs `H,L,X` in swapped order) + sidecar `mab1_sample_0.json` and `mab1_sample_0.npz`.
  - `fixture/single/mab1_af3.tar.gz` + `fixture/single/mab1_linked.pdb` (chains joined by a 15-residue `G` linker).
  - `fixture/homomer/hom_0.cif` and `fixture/homomer/hom_1.cif` (two identical chains, triggers v1 homomer refusal).
  - `fixture/expected.json` recording the exact analytical `rms_rmsd`, `ssRMSD`, `SD`, `within_source_mean_rmsd`, and `cross_source_mean_rmsd`.
- `tools/ssrmsd/test/run.sh` (`set -euo pipefail`, `work=$(mktemp -d /tmp/ssrmsd-test.XXXXXX)`, `trap 'rm -rf "$work"' EXIT`):
  1. Verifies `ssrmsd --version` equals `ssrmsd 0.1.1`.
  2. Verifies Pyxis isolation: creates `$work/poison/ssrmsd.py` (`raise RuntimeError("poisoned")`) and runs `env -i PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$work/poison" ssrmsd --version` from a read-only dir (`cd /toolshed/test`) — must succeed and print `ssrmsd 0.1.1`.
  3. Runs `ssrmsd batch --dir af3=/toolshed/test/fixture/af3 --dir esmfold2=/toolshed/test/fixture/esmfold2 --join intersection --workers 2 --align-chains A --score-chains B,C --ref-source af3 --pairwise-out "$work/pairs.csv" -o "$work/scores.csv"` (crossing `ProcessPoolExecutor` threshold with 3 ensembles and 2 workers) and validates every row's `n_structures == 5`, `sources == af3:3,esmfold2:2`, analytical metrics within `1e-4`, and `30` pairwise rows (`10` per ensemble).
  4. Runs `ssrmsd score af3=/toolshed/test/fixture/single/mab1_af3.tar.gz esmfold=/toolshed/test/fixture/single/mab1_linked.pdb --id mab1_single --align-chains A --score-chains B,C --ref-source af3 --pairwise-out "$work/single_pairs.csv" -o "$work/single.csv"` and checks analytical metrics within `1e-4`.
  5. Writes `$work/manifest.csv` containing `mab1` (valid multi-source) and `hom1` (`/toolshed/test/fixture/homomer`), runs `ssrmsd batch --manifest "$work/manifest.csv" --format jsonl --workers 2 -o "$work/manifest.jsonl"`, asserts exit `0` with `mab1` `status == "ok"` and `hom1` `status == "error"` in order, and verifies `ssrmsd score /toolshed/test/fixture/homomer -o "$work/hom.csv"` exits `1` with `status == "error"`.

- [ ] **Step 5: Write initial `tools/ssrmsd/SKILL.md` (with placeholder `## Compute` to be measured in Task 4)**

Follow `CONTRIBUTING.md` frontmatter and section order (`Image:`, `## Commands`, `## Inputs and outputs`, `## Compute`, `## Running it`, `## Gotchas`), documenting all five shed structure predictors (`alphafold3-blackwell`, `boltz-2` `.../predictions`, `esmfold2`/`esmfold2-fast`, `openfold3`, `opendde`), every CLI subcommand, exit codes, homomer refusal, fixture threshold (`ProcessPoolExecutor`), and MIT license.

- [ ] **Step 6: Commit `tools/ssrmsd/`, validate with `check.py`, build candidate, and run offline test**

```bash
git add tools/ssrmsd
git commit -m "Add ssrmsd 1.0.0 tool"
python3 -I scripts/check.py tools/ssrmsd
scripts/build ssrmsd
candidate=$(python3 -I scripts/check.py --candidate tools/ssrmsd)
podman run --rm --network none "$candidate" bash /toolshed/test/run.sh
```
Expected: `check.py` exits `0`, `scripts/build` produces `ghcr.io/brineylab/toolshed/ssrmsd:src-<tree>`, and `test/run.sh` exits `0` under `--network none`.

---

### Task 4: Measure Compute, Finalize `SKILL.md`, and (After User Confirmation) Push, Record Test, and Open PR

**Files:**
- Modify: `tools/ssrmsd/SKILL.md` (`## Compute` measured timings)
- Create (after push): `results/ssrmsd/1.0.0.txt` (via `scripts/test ssrmsd podman`)

**Interfaces:**
- Consumes: built candidate from Task 3, `scripts/build`, `scripts/push`, `scripts/test`, `scripts/check.py`.
- Produces: final verified `tools/ssrmsd/` + `results/ssrmsd/1.0.0.txt` passing `python3 -I scripts/check.py tools/ssrmsd --registry ghcr.io/brineylab/toolshed`, and PR on `brineylab/toolshed`.

- [ ] **Step 1: Measure real CPU throughput inside the built candidate image**

Time `test/run.sh` and a realistic-sized antibody–antigen complex benchmark (e.g., ~500-residue 3-chain complex, 5-structure ensemble, single and batched across workers) inside `podman run --rm --network none "$candidate" ...` so `SKILL.md`'s `## Compute` reports real measured numbers on `linux/amd64`.

- [ ] **Step 2: Update `tools/ssrmsd/SKILL.md` with measured `## Compute` numbers, amend/commit, rebuild, and re-verify offline**

```bash
git add tools/ssrmsd/SKILL.md
git commit --amend --no-edit
python3 -I scripts/check.py tools/ssrmsd
scripts/build ssrmsd
candidate=$(python3 -I scripts/check.py --candidate tools/ssrmsd)
podman run --rm --network none "$candidate" bash /toolshed/test/run.sh
```
Expected: `check.py`, `scripts/build`, and offline `test/run.sh` all exit `0`.

- [ ] **Step 3: User confirmation gate before pushing to GHCR or opening PR**

Per `CONTRIBUTING.md`, pause and present the verified local build/test summary and ask the user to confirm:
1. Running `scripts/push ssrmsd` (triggers `seed.yml` to create the public `ssrmsd` package if new, and pushes the candidate to GHCR).
2. Running `scripts/test ssrmsd podman` to generate `results/ssrmsd/1.0.0.txt`.
3. Pushing branch `add-ssrmsd` and opening the PR on `brineylab/toolshed`.

- [ ] **Step 4: Push candidate, record test, run full registry `check.py`, and open PR (after user confirmation)**

```bash
scripts/push ssrmsd
scripts/test ssrmsd podman
git add results/ssrmsd/1.0.0.txt
git commit -m "Record ssrmsd 1.0.0 test result"
python3 -I scripts/check.py tools/ssrmsd --registry ghcr.io/brineylab/toolshed
git push -u origin add-ssrmsd
gh pr create --title "Add ssrmsd 1.0.0" --body "..."
```
Expected: `scripts/check.py --registry ghcr.io/brineylab/toolshed` exits `0`, `results/ssrmsd/1.0.0.txt` matches the pushed candidate digest with `exit: 0`, and PR is open for user review and squash-merge.
