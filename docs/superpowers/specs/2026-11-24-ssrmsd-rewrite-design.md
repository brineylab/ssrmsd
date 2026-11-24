# Design: `ssrmsd` Package Rewrite (and Toolshed Integration)

## Intent and Why

`ssRMSD` (ensemble self-consistency RMSD for predicted antibody–antigen structures) currently exists in two tightly coupled copies:
1. The original `brineylab/ssrmsd` repo (`ssRMSD.py`), a single-file BioPython pipeline tied to AlphaFold Server `.zip` archives, intermediate CIF→PDB conversion, hardcoded chains `A/B/C`, and exactly 5 `<mab>_model_{0..4}.pdb` files.
2. An inlined `compute_ssrmsd` helper in a persistent AlphaFold3 Slurm worker script, tied to local AF3 `seed-*_sample-*/*.cif` directories, hardcoded chains `A/B/C`, and positional Cα matching in file order.

Both compute the same unnormalized legacy metrics (`sum(mat**2)` over both triangles of the symmetric $n \times n$ matrix and `std(mat)` including diagonal zeros).

We are completely replacing the contents of `brineylab/ssrmsd` (on branch `rewrite`) with a clean, generalizable Python package + CLI that:
- Scores either a single ensemble or a batch of ensembles of predicted antibody–antigen structures.
- Combines predictions from multiple structure predictors (e.g., AlphaFold3 + ESMFold2, Boltz, Chai) into the same ensemble reliably.
- Serves both as an installable Python library/CLI (e.g., directly importable by the AF3 Slurm worker or notebooks) and as the upstream package wrapped by a slim CPU-only `brineylab/toolshed` container (`ghcr.io/brineylab/toolshed/ssrmsd:<version>`) + `SKILL.md`.

---

## Goals, Non-Goals, and Anti-Goals

### Goals
- **Single & batch scoring:** Score one ensemble directly from CLI arguments, or batches of 10,000+ ensembles from directory roots or a manifest, parallelized across CPU cores.
- **Cross-model ensembles:** Mix `.cif` and `.pdb` predictions (plain, `.gz`, or inside `.tar.gz`/`.zip` archives) from different predictors into the same ensemble without requiring matching chain IDs, chain order, or identical terminal residue coverage.
- **Comparable & backward-compatible metrics:** Emit size-normalized upper-triangle metrics as primary scores alongside legacy `ssRMSD` and `SD` columns with exact numerical parity against existing 5-sample AF3 runs.
- **Flexible superposition scope:** Full-complex global Cα Kabsch RMSD by default, with optional `--align-chains` and `--score-chains` (e.g., superpose on antigen, measure RMSD on antibody).
- **Toolshed-ready packaging:** Clean `src/` Python package (`pyproject.toml`, `ssrmsd` CLI entrypoint on `PATH`, CPU-only, only `gemmi` + `numpy` runtime deps) ready to be installed directly into a slim `brineylab/toolshed` container (`python:3.12-slim`, no `ENTRYPOINT`) paired with a `SKILL.md`.

### Non-Goals
- Running structure prediction, MSA search, CIF→PDB disk conversion pipelines, or reading/aggregating AF3 confidence JSONs (`iptm`, `ptm`, `ranking_score`).
- Full DockQ, interface contact Fnat, or all-atom / side-chain RMSD (Cα backbone only).
- Homomer / multi-copy symmetry permutation search in v1 (kept out of v1 for simplicity; chain-mapping and Kabsch internals remain cleanly separated so symmetry permutation can be added later without changing CLI, I/O, or metrics).
- Cloud storage transfers (S3/GCS); the tool operates on local or mounted filesystem paths (including local `.tar.gz` and `.zip` archives).
- Aligning structurally homologous proteins with unrelated sequences (all structures in an ensemble are assumed to represent the same complex/sequences).
- Incremental `run_cache.json` stateful directory pipelines (stateless CLI; callers pass the inputs they want scored).

### Anti-Goals
- **Silent misalignment:** Never compute a bogus RMSD by pairing mismatched chains or shifted residues (e.g., different chain ordering, ambiguous identical chains, or unstripped poly-G linkers). Match by sequence and fail loudly when alignment quality, uniqueness, or coverage falls below threshold.
- **Silent AF3 double-counting:** Never silently include local AF3's top-level `<job>_model.cif` (a duplicate copy of the top-ranked sample) alongside `seed-*_sample-*/*_model.cif`.
- **Heavy runtime bloat:** Avoid BioPython/pandas/conda/GPU dependencies for a pure CPU C++/NumPy workload.
- **Breaking legacy AF3 numbers:** Legacy `ssRMSD` and `SD` must match the direct-`.cif` AF3 worker calculation (`compute_ssrmsd`) to 4 decimal places on 5-sample AF3 ensembles.

---

## Constraints

- **Repository (`brineylab/ssrmsd`):** Complete replacement of repo contents on the `rewrite` branch (keeping `LICENSE` MIT). Standard modern Python `src/` layout (`pyproject.toml`, `src/ssrmsd/`, `tests/`, `README.md`).
- **Toolshed compatibility (`brineylab/toolshed`):**
  - The downstream `toolshed` image (`ghcr.io/brineylab/toolshed/ssrmsd:<version>`, immutable semver tags, no `:latest`) installs `brineylab/ssrmsd` into `python:3.12-slim` with no `ENTRYPOINT` and `ssrmsd` on `PATH`, running identically under `podman`/`docker` (`-v "$PWD":/work -w /work`) and Slurm/Pyxis.
  - Zero runtime network access; no external binaries or model weights.
- **Runtime:** Python 3.11+, CPU-only, only `gemmi` and `numpy` as runtime dependencies, deterministic output regardless of filesystem traversal order or worker process scheduling.

---

## Approaches Considered

| Approach | Summary | Trade-offs |
|---|---|---|
| **1. `gemmi` + `numpy` Python package & slim container *(Picked)*** | Installable Python package (`ssrmsd` CLI + library) using `gemmi` (C++ mmCIF/PDB/gzip parsing + built-in C++ Needleman–Wunsch alignment) and `numpy` (Kabsch SVD), packaged in `python:3.12-slim` (~150 MB) for `toolshed`. | Fast (~ms/ensemble), only 2 runtime deps, handles `.cif`/`.pdb` quirks reliably, directly importable in pipelines like the AF3 worker. Clean separation between chain mapping and Kabsch makes adding homomer permutation straightforward later if needed. |
| **2. Pure-Python (`numpy` only)** | Extend the AF3 worker's hand-rolled mmCIF parser to support PDB, archives, and pure-Python sequence alignment. | Smallest dep footprint, but fragile across predictor PDB/mmCIF quirks (`altloc`, insertion codes, quoted fields, non-standard residues) and slow on large batches. |
| **3. Heavyweight structural-bio wrapper (`OpenStructure` / `US-align`)** | Wrap a full structural biology suite inside the container and call it from a thin CLI wrapper. | Built-in general oligomeric symmetry and DockQ support, but 1–2 GB container image, slower cold start, and awkward to reuse as a lightweight Python dependency. |

**Decision:** Approach 1 (`gemmi` + `numpy`). It is fast, minimal, handles heterogeneous predictors cleanly, keeps `brineylab/ssrmsd` cleanly pip-installable, and drops straight into a tiny CPU-only `toolshed` container.

---

## Design Details

### 1. Input Formats & Structure Discovery

- **Supported structure files:** `.cif`, `.mmcif`, `.pdb`, `.ent`, and gzipped `.cif.gz`, `.mmcif.gz`, `.pdb.gz`, `.ent.gz`.
- **Archive support:** `.tar.gz`, `.tgz`, `.tar` (e.g., archived local AF3 S3 outputs) and `.zip` (e.g., AlphaFold Server downloads) are read directly in memory via Python stdlib `tarfile` / `zipfile` and parsed with `gemmi` string readers—no temporary extraction to disk.
- **AF3 layout auto-detection:**
  - **Local AF3 outputs:** When expanding a directory or archive, if subdirectories matching `seed-*_sample-*` exist, collect structure files only from those sample directories and ignore the top-level `<job>_model.cif` duplicate.
  - **AlphaFold Server `.zip` / flat directories:** When no `seed-*_sample-*` subdirectories exist, collect all supported structure files inside the directory/archive (e.g., `<mab>_model_0.cif` … `<mab>_model_4.cif`).
  - Users can override discovery via `--glob <pattern>` or `--include-top-level`.
- **Multi-model files *(designer call)*:** Structure files with multiple `MODEL` records default to model 0 (`structure[0]`) unless `--split-models` is explicitly passed, so unexpected extra models never silently change ensemble size $n$.
- **Atom extraction:** For each chain containing polymeric residues, extract the primary conformer (`altloc` `""` or `"A"`) `CA` coordinates (`float64`) and 1-letter amino acid sequence. Non-polymer chains (waters, ions, small-molecule ligands with no polymer `CA`) are ignored.

### 2. Cross-Model Chain Matching & Residue Correspondence

To reliably combine predictions from heterogeneous models (e.g., AF3 `.cif` + ESMFold2 `.pdb`), preprocessing runs in this exact order:

1. **Parse & poly-G linker splitting *(designer call)*:**
   - Parse all structures in the ensemble and determine `max_chains = max(len(s.chains) for s in structures)`.
   - Before selecting a reference structure, if `--linker-min-len > 0` (default `10`), for any structure with `len(s.chains) < max_chains` that contains a chain with one or more stretches of `G{linker_min_len,}`, split that chain on the linker spans (dropping the linker residues). This ensures ESMFold-style single-chain multimer outputs split cleanly regardless of filename sort order.
2. **Reference structure selection *(designer call)*:**
   - If `--ref-source` is specified: pick the first sorted structure matching `--ref-source`. If no structure in the ensemble matches `--ref-source`, fail the ensemble with `status=error` (`"ref-source '<label>' not present in ensemble"`) rather than silently falling back to another model's chain IDs.
   - Otherwise: prefer the first sorted structure that naturally had `max_chains` without linker splitting (so reference chain IDs reflect real file chain IDs like `A, B, C`), falling back to the first sorted structure if all structures were split (assigning deterministic synthetic IDs `A, B, C, ...` to split segments).
3. **Determine required chains (`--align-chains` / `--score-chains`) *(designer call)*:**
   - Both flags accept comma-separated chain IDs referring to the **reference structure** (propagated to every other structure via sequence-based chain matching).
   - Default (neither flag set): `align_chains = score_chains = all reference chains`.
   - `--align-chains` set without `--score-chains`: `score_chains = all reference chains`.
   - `--score-chains` set without `--align-chains`: `align_chains = all reference chains`.
   - Both set (e.g., `--align-chains A --score-chains B,C`): superpose on `align_chains` and measure RMSD over `score_chains`.
   - Let `required_chains = align_chains ∪ score_chains`:
     - When **no** chain-selection flags are passed: enforce strict 1-to-1 chain-count equality across all structures (extra unmatched polymer chains in any structure fail with `status=error`).
     - When `--align-chains` and/or `--score-chains` **are** passed: every chain in `required_chains` must match 1-to-1 in every structure, while extra unmatched "spectator" chains outside `required_chains` are ignored.
4. **Sequence-based chain matching & homomer guard:**
   - Compute pairwise sequence identity between each structure's chains and the reference's `required_chains` via `gemmi.align_string_sequences`.
   - **Identity denominator *(designer call)*:** `exact_matches / aligned_residue_pairs` (identity over the aligned non-gap overlap), so terminal truncations with high sequence identity over the shared region pass `--min-chain-id` (default `0.90`) independently of `--min-coverage` (default `0.80`).
   - **Ambiguity / homomer guard (v1):** If any two reference chains in `required_chains` have sequence identity `>= --min-chain-id` to each other (i.e., homomeric/identical copies), or if multiple candidate chains in a non-reference structure tie above `--min-chain-id` for the same reference chain, fail loudly with `status=error` (`"ambiguous/homomeric chains detected; homomer symmetry matching is not enabled"`). Keep the chain-matcher interface returning a canonical mapping so symmetry-aware homomer permutation can be plugged in later without refactoring I/O, CLI, or metrics.
   - Solve optimal 1-to-1 assignment maximizing total overlap identity; fail with `status=error` if any required chain fails to match at `>= --min-chain-id`.
5. **Residue alignment & ensemble-wide Cα intersection *(designer call)*:**
   - Align each matched chain of every structure to the corresponding reference chain via `gemmi.align_string_sequences`.
   - Take the **ensemble-wide intersection** of reference residue positions that have a valid aligned `CA` atom in *every* structure of the ensemble. Every pair $(i, j)$ is therefore evaluated over the exact same atom set.
   - **Symmetric coverage guard *(designer call)*:** For each required chain, verify `retained_ca / max_chain_len >= --min-coverage` (default `0.80`), where `max_chain_len` is the maximum length of that matched chain across all structures in the ensemble. Checking against `max_chain_len` (rather than only the reference) guarantees coverage gating is symmetric and cannot be bypassed if a truncated structure happens to be chosen as reference.

### 3. Pairwise Kabsch RMSD & Metrics

For each unordered pair $(i, j)$ with $0 \le i < j < n$:
1. Extract alignment coordinates $P_A, Q_A \in \mathbb{R}^{|A| \times 3}$ and scoring coordinates $P_S, Q_S \in \mathbb{R}^{|S| \times 3}$ in `float64`.
2. Compute centroids $\mu_P = \bar{P}_A$, $\mu_Q = \bar{Q}_A$ and cross-covariance $H = (P_A - \mu_P)^T (Q_A - \mu_Q)$.
3. Compute SVD $H = U \Sigma V^T$; if $\det(V^T U^T) < 0$, negate the last row of $V^T$; set $R = V^T U^T$.
4. Transform scoring atoms: $P'_S = (P_S - \mu_P) R^T + \mu_Q$.
5. Compute $d_{ij} = \sqrt{\frac{1}{|S|} \sum_{m=1}^{|S|} \| P'_{S,m} - Q_{S,m} \|^2}$.

Build the symmetric $n \times n$ matrix $M$ ($M_{ii} = 0$, $M_{ij} = M_{ji} = d_{ij}$, unrounded `float64`) and let $U = \{d_{ij} : i < j\}$ be the $K = \frac{n(n-1)}{2}$ upper-triangle values:

| Column | Definition | Purpose |
|---|---|---|
| `ensemble_id` | Ensemble name/ID | Identifier |
| `n_structures` | $n$ | Ensemble size |
| `n_ca_align` | $|A|$ | Cα atoms used for Kabsch superposition |
| `n_ca_score` | $|S|$ | Cα atoms measured in RMSD |
| `mean_sq_rmsd` | $\frac{1}{K}\sum_{d \in U} d^2 = \frac{\text{ssRMSD}}{n(n-1)}$ | Size-normalized ssRMSD ($\text{Å}^2$) |
| `rms_rmsd` | $\sqrt{\text{mean\_sq\_rmsd}}$ | Root-mean-square pairwise RMSD (Å) |
| `mean_rmsd` | $\frac{1}{K}\sum_{d \in U} d$ | Mean pairwise RMSD (Å) |
| `median_rmsd` | $\text{median}(U)$ | Outlier-resistant pairwise RMSD (Å) |
| `rmsd_sd` | $\text{std}(U, \text{ddof}=0)$ (upper triangle only) | Spread across pairs (Å) |
| `ssRMSD` | $\sum_{i,j} M_{ij}^2 = 2\sum_{d \in U} d^2$ | Legacy parity with AF3 worker |
| `SD` | $\text{std}(M, \text{ddof}=0)$ (all $n^2$ entries incl. diagonal) | Legacy parity with AF3 worker |
| `within_source_mean_rmsd` | Mean $d_{ij}$ where $\text{source}_i = \text{source}_j$ (empty if single/no source) | Multi-model internal agreement |
| `cross_source_mean_rmsd` | Mean $d_{ij}$ where $\text{source}_i \ne \text{source}_j$ (empty if single/no source) | Multi-model cross-predictor agreement |
| `sources` | Compact breakdown, e.g. `af3:5,esmfold:3` (empty if unlabeled) | Provenance |
| `status` | `ok` or `error` | Batch resilience |
| `error` | Short diagnostic message when `status=error` | Debugging |

- **Legacy parity baseline:** All metrics are computed from unrounded `float64` pairwise RMSDs directly from `.cif`/`.pdb` coordinates (without intermediate PDB 3-decimal coordinate truncation) and rounded to 4 decimal places on tabular output. Legacy `ssRMSD` (`np.sum(M ** 2)`) and `SD` (`np.std(M, ddof=0)`) match the AF3 worker script's `compute_ssrmsd` (`np.sum(mat ** 2)` and `np.std(mat)`) within `1e-4` on 5-sample AF3 ensembles.
- Ensembles with $n < 2$ produce `status=error` (`"fewer than 2 structures"`).
- **Optional pairwise matrix output (`--pairwise-out PATH`):** Writes one row per unordered pair (`ensemble_id, struct_i, struct_j, source_i, source_j, rmsd`) in CSV/TSV/JSONL matching `--format`.

### 4. CLI & Python API

#### CLI (`ssrmsd`)

1. **Single ensemble:**
   ```bash
   ssrmsd score <input>... [options]
   # Examples:
   ssrmsd score job_dir/
   ssrmsd score af3=af3_mAb1.tar.gz esmfold=esm_mAb1/ --align-chains A --score-chains B,C
   ssrmsd score pred1.cif pred2.cif pred3.pdb -o score.csv
   ```
   - Positional inputs may be structure files, directories, or `.tar.gz`/`.tar`/`.zip` archives, optionally prefixed with `label=` to tag their source model.
   - `--id NAME` sets `ensemble_id` (defaults to the common directory/archive stem or `"ensemble"`).

2. **Batch of ensembles:**
   ```bash
   ssrmsd batch --dir af3=af3_runs/ --dir esmfold=esm_runs/ --join intersection -o scores.csv
   ssrmsd batch --manifest ensembles.csv --workers 16 --pairwise-out pairs.csv -o scores.csv
   ```
   - `--dir [label=]ROOT` (repeatable): discovers ensembles as direct child directories, `.tar.gz`/`.tgz`/`.tar`/`.zip` archives, or flat structure files under each `ROOT`. Derives `ensemble_id` by stripping archive/structure extensions (and for flat `<mab>_model_<k>.{cif,pdb}` files, grouping by the prefix before `_model_`), and merges matching IDs across roots.
     - `--join {union,intersection}` (default `union`): `union` scores every `ensemble_id` found in any root; `intersection` restricts to IDs present in *all* `--dir` roots.
   - `--manifest PATH`: CSV or TSV (delimiter inferred from extension or `--delimiter`) with columns `ensemble_id,path` and optional `source`. Each `path` may be a structure file, directory, or archive; multiple rows with the same `ensemble_id` are combined into one ensemble.
   - `--dir` and `--manifest` are mutually exclusive to keep provenance unambiguous.

3. **Shared options:**
   - `--align-chains CHAINS`, `--score-chains CHAINS`, `--ref-source SOURCE`
   - `--glob PATTERN`, `--include-top-level`, `--split-models`
   - `--min-chain-id FLOAT` (default `0.90`), `--min-coverage FLOAT` (default `0.80`), `--linker-min-len INT` (default `10`)
   - `--format {csv,tsv,jsonl}` (default `csv`), `-o / --output PATH` (default stdout), `--pairwise-out PATH`
   - `--workers INT` (default `1` for `score`, `os.cpu_count()` for `batch`), `--fail-fast`

4. **Exit codes *(designer call)*:**
   - `ssrmsd score`: writes the output row; exits `0` if `status=ok`, exits `1` if `status=error`.
   - `ssrmsd batch`: writes all rows and exits `0` even if individual ensembles record `status=error` (preventing Slurm/workflow aborts over isolated failed predictions), unless `--fail-fast` is set. Fatal errors (invalid CLI args, unreadable manifest, zero ensembles discovered, or `--fail-fast` hit) exit non-zero.

#### Python API
```python
from ssrmsd import score_ensemble, score_batch, EnsembleSpec, EnsembleResult
```
- `score_ensemble(inputs, *, ensemble_id=..., align_chains=None, score_chains=None, ...) -> EnsembleResult`
- `score_batch(ensembles, *, workers=1, ...) -> list[EnsembleResult]` (returned deterministically in input order)

### 5. Repository & Toolshed Layout

**`brineylab/ssrmsd` (`rewrite` branch):**
```text
ssrmsd/
├── LICENSE
├── README.md
├── pyproject.toml                       # hatchling/setuptools; Python 3.11+; deps: gemmi, numpy; CLI `ssrmsd`
├── docs/superpowers/specs/
│   └── 2026-11-24-ssrmsd-rewrite-design.md
├── src/ssrmsd/
│   └── ...
└── tests/
    └── ...
```

**Downstream `brineylab/toolshed` wrapper (`tools/ssrmsd/`, separate repo/PR):**
- `Containerfile`: `FROM python:3.12-slim`, `pip install --no-cache-dir "git+https://github.com/brineylab/ssrmsd.git@v<version>"`, no `ENTRYPOINT` (`ssrmsd` on `PATH`).
- `SKILL.md`: prose skill documenting image tag, `ssrmsd score` / `ssrmsd batch` usage, CPU-only compute profile, and caveats (AF3 top-level duplicate handling, multi-model chain matching, `--align-chains` reference semantics).

---

## Left to the Builder

- Internal module decomposition under `src/ssrmsd/`.
- Build backend (`hatchling` vs. `setuptools`) and stdlib `argparse` vs. `click`/`typer` (stdlib `argparse` keeps runtime deps strictly to `gemmi` + `numpy`, preferred).
- Exact `gemmi` alignment scoring parameters and optimal linear-assignment implementation (small-N exact solver vs. Hungarian in pure Python/NumPy).
- Whether to also include a reference `Containerfile` / `SKILL.md` template in `brineylab/ssrmsd` or author those solely in `brineylab/toolshed`.
- Test fixtures (synthetic + realistic AF3 `.cif`, AlphaFold Server `.zip`, ESMFold `.pdb`, poly-G linker, and `.tar.gz` samples) verifying exact 4-decimal parity with the legacy AF3 worker's `compute_ssrmsd` formula.
