# Design: `toolshed` Tool for `ssRMSD` (and `ssRMSD v0.1.1` Patch)

## Intent and Why

`ssRMSD` (`v0.1.0`) is now built and published on PyPI as a CPU-only Python library and CLI for self-consistency RMSD scoring of predicted antibody–antigen structure ensembles. To make `ssrmsd` directly callable by lab members, Slurm/Pyxis batch jobs, and autonomous campaigns (such as `flyloft`) alongside the lab's existing structure prediction containers (`alphafold3-blackwell`, `boltz-2`, `esmfold2`, `esmfold2-fast`, `openfold3`, `opendde`), we want to package it as a first-class tool (`tools/ssrmsd/`) in `github.com/brineylab/toolshed`.

The primary expected workflow is pipeline-style batch scoring: one or more structure prediction tools in the shed generate a batch of predicted ensembles, and those batch output directories are passed directly into `ssrmsd batch` (alongside one-off `ssrmsd score` calls).

Recon against all five structure predictors in `brineylab/toolshed` uncovered one compatibility gap in `ssRMSD v0.1.0`: `esmfold2` and `esmfold2-fast` write flat batch outputs named `<target>_sample_<i>.cif`, whereas `ssrmsd batch --dir` in `v0.1.0` only strips `_model_\d+` (treating every `esmfold2` sample as a separate 1-structure ensemble that fails with `"fewer than 2 structures"`). Cutting a small `ssRMSD v0.1.1` release first fixes that natively so every structure predictor in the shed plugs directly into `ssrmsd batch --dir`.

---

## Goals, Non-Goals, and Anti-Goals

### Goals
- **Seamless batch interoperability across all shed predictors:** Release `ssRMSD v0.1.1` adding flat `*_sample_<k>` batch grouping (alongside `*_model_<k>`) and `ssrmsd --version` / `-V` so `alphafold3-blackwell`, `boltz-2`, `esmfold2`/`esmfold2-fast`, `openfold3`, and `opendde` all feed cleanly into `ssrmsd batch --dir`.
- **Self-contained, reproducible CPU-only toolshed image (`ghcr.io/brineylab/toolshed/ssrmsd:1.0.0`):** Slim `python:3.12-slim-bookworm` image with pinned base digest, pinned PyPI packages (`ssRMSD==0.1.1` and all transitive dependencies), no `ENTRYPOINT`, `ssrmsd` on `/usr/local/bin`, and Pyxis isolation (`python3 -I` + `PYTHONNOUSERSITE=1`).
- **Agent- and human-ready `SKILL.md`:** Prose skill following `brineylab/toolshed` conventions, leading with batch scoring pipelines and concrete path recipes for every shed structure predictor, plus measured CPU throughput and clear gotchas.
- **Threshold-crossing in-image test (`test/run.sh`):** Offline (`--network none`), `/tmp`-only test exercising every documented command and crossing `ssrmsd`'s real runtime thresholds (`ProcessPoolExecutor` multiprocessing with `len(ensembles) > 1` and `--workers 2`, AF3 duplicate exclusion, flat `*_sample_*` grouping, archives, poly-G linker splitting, `--align-chains`/`--score-chains`, and `--pairwise-out`) with known analytical RMSDs checked to `1e-4`.
- **Full toolshed verification:** Pass `scripts/check.py`, build `linux/amd64` candidate via `scripts/build`, and (after user confirmation) push via `scripts/push` (seeding the public GHCR package via `seed.yml`), record `results/ssrmsd/1.0.0.txt` via `scripts/test ssrmsd podman`, and open a PR on `brineylab/toolshed`.

### Non-Goals
- Adding new scoring algorithms, homomer symmetry permutation, confidence-JSON parsing, or GPU code to `ssRMSD` (keep `v0.1.1` strictly scoped to `*_sample_<k>` grouping and `--version`).
- Writing a `tools/ssrmsd/bin/` wrapper script around `ssrmsd` (upstream already ships a full CLI).
- Packaging any structure predictor weights or adding a workflow orchestrator inside the container.
- Modifying `brineylab/toolshed`'s `docs/DESIGN.md`, `docs/VISION.md`, or CI scripts.

### Anti-Goals
- **Pyxis environment leakage:** Never let a Slurm caller's `PYTHONPATH` or `~/.local` site-packages shadow the container's pinned Python packages, and never rely on a custom `ENV PATH` directory that enroot drops.
- **Trivial test fixture that skips real code paths:** Never test only a single ensemble or `--workers 1`, which would skip `ProcessPoolExecutor` entirely (`score_batch` only forks workers when `workers > 1` and `len(ensembles) > 1`).
- **Silent unapproved public actions:** Never push directly to `main` on `brineylab/ssrmsd` (`v0.1.1` goes through a PR), and never run `scripts/push` (which triggers `seed.yml` and pushes to GHCR under the user's account) or open a PR on `brineylab/toolshed` without asking the user first, per `CONTRIBUTING.md`. Once the `v0.1.1` PR is merged by the user, tagging and publishing the `v0.1.1` GitHub Release (triggering PyPI upload) is pre-authorized with no extra checkpoint.
- **Unpinned or credential-carrying builds:** Never leave base image or Python dependencies unpinned, and never place tokens/credentials in files, `ARG`, `ENV`, or test output.

---

## Constraints

- **`brineylab/ssrmsd` (`v0.1.1`):**
  - Existing CI (`ci.yaml`): must pass `ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/`, and `pytest` across Python 3.11–3.13.
  - PyPI publishing (`python-publish.yaml`): triggers on `release: types: [published]` (so cutting `v0.1.1` requires pushing tag `v0.1.1` and publishing a GitHub Release, then confirming `ssRMSD==0.1.1` is available on PyPI).
- **`brineylab/toolshed` (`tools/ssrmsd/`):**
  - Directory name `ssrmsd` matches `SKILL.md` frontmatter `name: ssrmsd`; `metadata.version: 1.0.0` (semver `MAJOR.MINOR.PATCH`, immutable once promoted).
  - `Containerfile` must end with `COPY test /toolshed/test`, have no `ENTRYPOINT`, pin base image and all packages, and contain no credential `ARG`/`ENV`.
  - `SKILL.md` frontmatter must have `name`, `description` (double-quoted if containing `:`), and `metadata.version`; body sections in order: `Image:`, `## Commands`, `## Inputs and outputs`, `## Compute`, `## Running it`, `## Gotchas`. Every example is a full command line.
  - `test/run.sh` must exit `0` iff all checks pass, write only under `/tmp`, pass under `--network none`, and fit well under 30 minutes on CPU.
  - `scripts/build ssrmsd` requires all changes under `tools/ssrmsd/` to be committed first (candidate tag is `src-<git-tree-hash>`).
  - `results/ssrmsd/1.0.0.txt` must match the pushed candidate's digest and show `exit: 0`, and pass `python3 -I scripts/check.py tools/ssrmsd --registry ghcr.io/brineylab/toolshed`.

---

## Approaches Considered

| Approach | Summary | Trade-offs |
|---|---|---|
| **1. Thin PyPI-installed `python:3.12-slim-bookworm` image, no wrapper *(Picked)*** | Cut `ssRMSD v0.1.1` (published to PyPI), install pinned `ssRMSD==0.1.1` + pinned transitive deps from PyPI into digest-pinned `python:3.12-slim-bookworm`, patch `/usr/local/bin/ssrmsd` shebang to `python3 -I` + `PYTHONNOUSERSITE=1`, no `bin/` wrapper. | Smallest, cleanest image (~150 MB, no `git` or compilers), strictly follows toolshed's "wrapper only where upstream has none" rule, tests the exact PyPI package. Requires waiting ~1 min for `v0.1.1` PyPI publish before building. |
| **2. Git-commit install in `python:3.12-slim-bookworm`** | Install `"ssRMSD @ git+https://github.com/brineylab/ssrmsd@<40-char-sha>"` with pinned deps in `requirements.txt`. | Avoids waiting on PyPI and pins a git SHA directly, but requires `git` + `ca-certificates` in the image and a VCS wheel build (`hatch-vcs`) when official PyPI wheels already exist. |
| **3. Toolshed `bin/` adapter wrapper around `ssrmsd`** | Keep `ssrmsd` untouched and add predictor-specific CLI wrappers in `tools/ssrmsd/bin/`. | Violates toolshed's "wrapper only where upstream has none" rule and creates a second diverging CLI surface. |

**Decision:** Approach 1. It delivers the smallest, cleanest container with zero duplicate CLI code, while `v0.1.1` makes all five structure predictors in `toolshed` work directly with `ssrmsd batch --dir`.

---

## Design Details

### 1. Upstream `ssRMSD v0.1.1` Patch (`brineylab/ssrmsd`)

1. **Flat sample grouping (`src/ssrmsd/io.py`):**
   - Update `_MODEL_GROUP_RE` from `r"^(.+)_model_\d+$"` to `r"^(.+)_(?:model|sample)_\d+$"`.
   - Flat files like `mab1_sample_0.cif`, `mab1_sample_1.cif` (the exact batch output of `toolshed`'s `esmfold2` / `esmfold2-fast`) now group under `ensemble_id = "mab1"` in `discover_batch_dirs`, matching `af3`, `boltz-2`, `openfold3`, and `opendde` target IDs directly.
2. **CLI version flag (`src/ssrmsd/__init__.py`, `src/ssrmsd/cli.py`):**
   - Expose `__version__` in `ssrmsd/__init__.py` via `importlib.metadata.version("ssRMSD")` (with `PackageNotFoundError` fallback).
   - Add an eager `--version` / `-V` callback option on the root Typer `app` that prints `ssrmsd <version>` and exits `0`.
3. **Tests & docs (`tests/test_io.py`, `tests/test_cli.py`, `README.md`):**
   - Add unit tests for `*_sample_<k>` batch grouping (including cross-root join between AF3 subdir + flat `esmfold2` `*_sample_<k>.cif` root) and `ssrmsd --version` / `-V`.
   - Update `README.md` batch example comment and toolshed snippet to reference `ghcr.io/brineylab/toolshed/ssrmsd:1.0.0`.
4. **Release:**
   - Run `ruff check`, `ruff format --check`, `ty check`, and `pytest`.
   - Open a PR on `brineylab/ssrmsd` (e.g., branch `release/0.1.1`) for user review and merge.
   - Once merged by the user, pull `main`, tag `v0.1.1`, publish GitHub Release `v0.1.1` (pre-authorized; triggers `python-publish.yaml` to PyPI), and verify `ssRMSD==0.1.1` is live on PyPI before moving straight into building the `toolshed` image.

### 2. Toolshed Directory Layout (`brineylab/toolshed/tools/ssrmsd/`)

```text
tools/ssrmsd/
├── Containerfile
├── LICENSE                     # Upstream MIT license from brineylab/ssrmsd
├── SKILL.md                    # Prose skill (metadata.version: 1.0.0)
└── test/
    ├── run.sh                  # Offline test script exercising all documented commands
    └── fixture/                # Kilobyte-scale synthetic antibody-antigen ensemble fixture
        └── ...                 # AF3 dir (+ decoy top-level), flat esmfold2 dir, .tar.gz/.zip, poly-G pdb, manifest
```

### 3. `Containerfile`

- **Base image:** `FROM docker.io/library/python:3.12-slim-bookworm@sha256:<pinned-digest>` (CPU-only, no `NVIDIA_*` env vars, no `apt-get` needed).
- **Package install:** Single `RUN pip install --no-cache-dir` pinning `ssRMSD==0.1.1` and every direct + transitive dependency at exact `==` versions (`gemmi`, `numpy`, `typer`, `click`, `typing-extensions`, and any optional rich/shellingham deps if pulled; verify exact resolved freeze on CPython 3.12 linux/amd64).
- **Pyxis isolation:**
  - `ENV PYTHONNOUSERSITE=1`
  - `RUN sed -i '1s|.*|#!/usr/local/bin/python3 -I|' /usr/local/bin/ssrmsd`
  - `/usr/local/bin/ssrmsd` is on the distro default `PATH` (survives Pyxis/enroot, which drops custom `ENV PATH`).
- **License & build-time smoke check:**
  - `COPY LICENSE /opt/ssrmsd/LICENSE`
  - Build-time sanity check: `RUN ssrmsd --version | grep -Fx 'ssrmsd 0.1.1'`
- **Test layer last:**
  - `COPY test /toolshed/test`
- **No `ENTRYPOINT`**, no `HF_HUB_OFFLINE` (no Hugging Face models or weights; offline operation is verified by `scripts/test` with `--network none`).

### 4. `SKILL.md`

Frontmatter:
```yaml
---
name: ssrmsd
description: "Self-consistency RMSD (ssRMSD) scoring for predicted antibody-antigen and protein complex structure ensembles: scores single ensembles or parallel batches across heterogeneous predictors (AF3, Boltz-2, ESMFold2, OpenFold3, OpenDDE). Reach for it to rank or compare ensemble structural consistency after running one or more structure predictors."
metadata:
  version: 1.0.0
---
```

Body sections (strictly following `CONTRIBUTING.md` order):
1. **`Image:`** `` `ghcr.io/brineylab/toolshed/ssrmsd:1.0.0` `` + brief overview naming `ssRMSD` 0.1.1 (MIT, `github.com/brineylab/ssrmsd`), pinned `gemmi`/`numpy`/`typer`, CPU-only, no network, no weights.
2. **`## Commands`**:
   - Lead with **`ssrmsd batch`** (primary pipeline use case):
     - Single-predictor batch (`--dir af3=/work/af3_out --workers 8 -o /work/scores.csv`)
     - Multi-predictor cross-joined batch (`--dir af3=/work/af3_out --dir esmfold2=/work/esm_out --join intersection --align-chains A --score-chains B,C --workers 8 --pairwise-out /work/pairs.csv -o /work/scores.csv`)
     - Manifest batch (`--manifest /work/ensembles.csv --format jsonl --workers 8 -o /work/scores.jsonl`)
   - Concrete path table/prose showing how each of the five shed structure predictors feeds directly into `--dir`:
     - `alphafold3-blackwell`: `--dir af3=<output_dir>` (auto-detects `<target>/seed-<s>_sample-<i>/` and skips top-level `<target>_model.cif`)
     - `openfold3`: `--dir openfold3=<output_dir>` (`<target>/seed_<seed>/*.cif`)
     - `opendde`: `--dir opendde=<output_dir>` (`<target>/seed_<seed>/predictions/*.cif`)
     - `boltz-2`: `--dir boltz=<out_dir>/boltz_results_<batch>/predictions` (point at the `predictions/` directory whose children are `<target>/`)
     - `esmfold2` / `esmfold2-fast`: `--dir esmfold2=<out>` (auto-groups flat `<target>_sample_<i>.cif` files by `<target>`)
   - **`ssrmsd score`** for one-off ensembles (directory, `.tar.gz`/`.zip`, or labeled multi-predictor inputs).
   - **`ssrmsd --version`** for version checks.
3. **`## Inputs and outputs`**:
   - Supported structure formats (`.cif`, `.mmcif`, `.pdb`, `.ent`, plain or `.gz`) and in-memory archives (`.tar.gz`, `.tgz`, `.tar`, `.zip`).
   - Manifest schema (`ensemble_id,path[,source]`).
   - Output formats (`csv`, `tsv`, `jsonl`) and concise column definitions (`mean_sq_rmsd`, `rms_rmsd`, `mean_rmsd`, `median_rmsd`, `rmsd_sd`, legacy `ssRMSD` / `SD`, `within_source_mean_rmsd`, `cross_source_mean_rmsd`, `sources`, `status`, `error`, plus `--pairwise-out` columns).
4. **`## Compute`**:
   - CPU-only, small memory footprint; scales linearly with `--workers` across ensembles.
   - Include actual measured wall-time/throughput from running the container on this machine (single ensemble and batch rate).
5. **`## Running it`**:
   - Laptop (`podman run --rm -v "$PWD":/work -w /work ghcr.io/brineylab/toolshed/ssrmsd:1.0.0 ssrmsd batch ...` — no `--device` GPU flag).
   - Slurm with pyxis (`srun --cpus-per-task=8 --mem=8G --container-image=ghcr.io#brineylab/toolshed/ssrmsd:1.0.0 --container-mounts="$PWD":/work --container-workdir=/work ssrmsd batch ...`).
6. **`## Gotchas`**:
   - **Boltz-2 `--dir` path:** pass `boltz_results_<batch>/predictions`, not `--out_dir` itself.
   - **AF3 top-level duplicate exclusion:** when `seed-*_sample-*` directories exist, top-level `*_model.cif` is skipped automatically (`--include-top-level` or `--glob` overrides).
   - **`--align-chains` / `--score-chains` reference semantics:** chain IDs refer to the reference structure (use `--ref-source` when mixing predictors with different chain IDs; missing `--ref-source` fails the ensemble loudly).
   - **Homomers / identical chains refused in v1:** ensembles with two required chains at `>= --min-chain-id` (default `0.90`) fail loudly with `status=error` rather than risking a wrong chain pairing (e.g., HIV-1 protease homodimers from `esmfold2`/`openfold3`/`opendde` fixtures will error).
   - **Poly-G linkers & truncation:** poly-G runs (`>= --linker-min-len`, default `10`) are split automatically when a structure has fewer chains than `max_chains`; coverage is checked symmetrically against `max_chain_len` (`--min-coverage`, default `0.80`).
   - **Exit codes in batch vs single mode:** `score` exits `1` on `status=error`; `batch` writes error rows and exits `0` unless `--fail-fast` is passed or a fatal CLI/manifest error occurs.
   - **What the test covers & the threshold it crosses:** explicitly name `score_batch`'s `ProcessPoolExecutor` threshold (`len(ensembles) > 1` and `--workers > 1`) plus heterogeneous predictor discovery/alignment, and note what is not measured (very large 10k+ ensemble batches).
   - **License:** `ssRMSD` is MIT (`/opt/ssrmsd/LICENSE` in the image); no model weights.

### 5. In-Image Test (`tools/ssrmsd/test/run.sh` + Fixture)

A compact, deterministic synthetic fixture (a few tens of kilobytes total) built from 3-chain antibody–antigen-like complexes (distinct non-homomeric chains `A`, `B`, `C`, ~25–30 residues each) with rigid-body perturbations whose exact analytical Cα RMSDs are known by construction:

1. **Version & isolation check:**
   - `ssrmsd --version` matches `ssrmsd 0.1.1`.
   - Verify `PYTHONPATH=/tmp/poison ssrmsd --version` still works (proving `python3 -I` ignores caller `PYTHONPATH`).
2. **Threshold-crossing multi-ensemble batch (`ssrmsd batch --dir ... --workers 2`):**
   - At least 3 ensembles across two roots (`af3=...` with `<target>/seed-1_sample-{0,1,2}/model.cif` + an intentionally perturbed decoy top-level `<target>_model.cif` that MUST be excluded; `esmfold2=...` with flat `<target>_sample_{0,1}.cif` with permuted chain IDs `H,L,X`).
   - Run `ssrmsd batch --dir af3=... --dir esmfold2=... --join intersection --workers 2 --align-chains A --score-chains B,C --ref-source af3 --pairwise-out /tmp/.../pairs.csv -o /tmp/.../scores.csv`.
   - Verifies: `ProcessPoolExecutor` works inside the container, AF3 decoy top-level file was excluded (`n_structures == 5`, `sources == af3:3,esmfold2:2`), flat `esmfold2` `*_sample_*.cif` files grouped cleanly, sequence-based chain matching mapped permuted chain IDs, analytical `rms_rmsd`, `ssRMSD`, `SD`, `within_source_mean_rmsd`, `cross_source_mean_rmsd` match expected values within `1e-4`, and `--pairwise-out` has `10` pairs per ensemble.
3. **Single ensemble (`ssrmsd score`) with archive + poly-G linker `.pdb`:**
   - Score a `.tar.gz` AF3 archive combined with a poly-G-linked `.pdb` (`--pairwise-out`, `-o`), checking `status == ok` and exact analytical RMSD within `1e-4`.
4. **Manifest batch (`ssrmsd batch --manifest ... --format jsonl --workers 2`):**
   - Includes one valid ensemble and one intentionally broken ensemble (e.g., 1 structure or homomer) to verify `batch` writes both JSONL records (`ok` and `error`) and exits `0`, while `score` on the broken ensemble exits `1`.
5. **Hygiene:** all outputs written under `mktemp -d /tmp/ssrmsd-test.XXXXXX` and cleaned up on `EXIT`; passes under `podman run --rm --network none`.

### 6. Execution & Publication Workflow

1. Implement and test the `v0.1.1` patch on a branch in `/home/bryanbriney/git/ssrmsd` and open a PR for user merge.
2. Once the user merges the PR, tag `v0.1.1`, publish the GitHub Release (pre-authorized; triggers PyPI upload), and wait for `ssRMSD==0.1.1` to appear on PyPI.
3. Clone `brineylab/toolshed` to a working checkout (or use an existing checkout if the user has one), create branch `add-ssrmsd`, and author `tools/ssrmsd/`.
4. Commit `tools/ssrmsd/`, run `python3 -I scripts/check.py tools/ssrmsd`, run `scripts/build ssrmsd`, and run `podman run --rm --network none <candidate> bash /toolshed/test/run.sh`.
5. Measure throughput in the built image and update `SKILL.md`'s `## Compute` numbers if needed (re-committing and rebuilding so the candidate matches the committed tree).
6. **Ask the user before publishing** (per `CONTRIBUTING.md`): confirm `scripts/push ssrmsd` (which dispatches `seed.yml` via `gh` and pushes the candidate to GHCR), then run `scripts/test ssrmsd podman` to write `results/ssrmsd/1.0.0.txt`, verify `python3 -I scripts/check.py tools/ssrmsd --registry ghcr.io/brineylab/toolshed`, commit the record, and (with user confirmation) push the branch and open the PR on `brineylab/toolshed`.

---

## Left to the Builder

- Exact pinned `docker.io/library/python:3.12-slim-bookworm@sha256:...` digest (latest current digest or matching `tools/datascience/Containerfile`).
- Exact pinned transitive Python dependency versions resolved on CPython 3.12 `linux/amd64` for `ssRMSD==0.1.1`.
- Exact synthetic residue sequences and rigid-body translation/rotation parameters used to construct the kilobyte fixture (and whether to include the small generator script under `test/` for reproducibility, like `tools/igblast/test/make_fixture.py`).
- Exact local path for cloning `brineylab/toolshed` during implementation.
