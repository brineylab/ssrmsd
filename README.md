# ssrmsd

Self-consistency RMSD (`ssRMSD`) scoring for predicted antibody–antigen structure ensembles.

`ssrmsd` scores single or batched ensembles across heterogeneous structure predictors (e.g., AlphaFold3 `.cif`/`.tar.gz`/`.zip` mixed with ESMFold2 `.pdb` predictions), matching chains by sequence and computing both size-normalized upper-triangle metrics and legacy 5-sample AF3 parity metrics.

## Features

- **Format & archive agnostic:** reads `.cif`, `.mmcif`, `.pdb`, `.ent` (plain or `.gz`), plus `.tar.gz`/`.tgz`/`.tar` and `.zip` archives directly in memory with zero temp-file extraction.
- **AF3-aware discovery:** auto-detects `seed-*_sample-*` subdirectories and excludes local AF3's duplicate top-level `<job>_model.cif` (override with `--include-top-level` or `--glob`).
- **Cross-model chain & residue matching:** matches chains by overlap sequence identity via `gemmi`, splits ESMFold-style poly-G linkers (`G{10,}`), aligns shared Cα atoms across the entire ensemble, and guards against silent truncation or ambiguous homomer matching.
- **Flexible superposition:** full-complex global Cα Kabsch by default, plus optional `--align-chains` / `--score-chains` (e.g., superpose on antigen, measure RMSD on antibody).
- **Comparable + legacy metrics:** emits size-normalized metrics (`mean_sq_rmsd`, `rms_rmsd`, `mean_rmsd`, `median_rmsd`, `rmsd_sd`), legacy AF3 parity columns (`ssRMSD`, `SD`), and optional within-source vs. cross-source breakdowns.

## Installation

Python 3.11+ (CPU-only; `gemmi`, `numpy`, and `typer`):

```bash
pip install ssRMSD
```

For local development:

```bash
pip install -e ".[dev]"
pytest
```

## CLI Usage

### Score a single ensemble

```bash
# Local AF3 job directory or archived .tar.gz / AlphaFold Server .zip
ssrmsd score /path/to/mab1_af3/
ssrmsd score /path/to/mab1.tar.gz

# Combine AF3 + ESMFold2 predictions into one ensemble, aligning on antigen (A) and scoring Fv (B,C)
ssrmsd score af3=af3_mab1.tar.gz esmfold=esm_mab1/ \
  --align-chains A --score-chains B,C \
  --pairwise-out pairs.csv -o score.csv
```

### Score a batch of ensembles

```bash
# Join multiple predictor roots by matching subfolder/archive name (or flat <mab>_model_<k> files)
ssrmsd batch \
  --dir af3=/data/af3_runs \
  --dir esmfold=/data/esm_runs \
  --join intersection \
  --workers 16 \
  -o scores.csv

# Arbitrary paths/sources via CSV or TSV manifest (columns: ensemble_id,path[,source])
ssrmsd batch --manifest ensembles.csv --workers 16 --format jsonl -o scores.jsonl
```

## Output Columns

| Column | Description |
|---|---|
| `ensemble_id` | Ensemble identifier |
| `n_structures` | Number of structures $n$ in the ensemble |
| `n_ca_align` | Common Cα atoms used for Kabsch superposition |
| `n_ca_score` | Common Cα atoms measured in RMSD |
| `mean_sq_rmsd` | Size-normalized ssRMSD: $\frac{1}{\binom{n}{2}} \sum_{i<j} d_{ij}^2 = \frac{\text{ssRMSD}}{n(n-1)}$ ($\text{Å}^2$) |
| `rms_rmsd` | Root-mean-square pairwise RMSD: $\sqrt{\text{mean\_sq\_rmsd}}$ (Å) |
| `mean_rmsd` | Mean upper-triangle pairwise RMSD (Å) |
| `median_rmsd` | Median upper-triangle pairwise RMSD (Å) |
| `rmsd_sd` | Population SD (`ddof=0`) of upper-triangle pairwise RMSDs (Å) |
| `ssRMSD` | Legacy parity: $\sum_{i,j} M_{ij}^2 = 2\sum_{i<j} d_{ij}^2$ |
| `SD` | Legacy parity: `np.std(M, ddof=0)` over all $n^2$ entries including diagonal |
| `within_source_mean_rmsd` | Mean pairwise RMSD for pairs from the same source (multi-source only) |
| `cross_source_mean_rmsd` | Mean pairwise RMSD for pairs from different sources (multi-source only) |
| `sources` | Per-source counts, e.g. `af3:5,esmfold:3` |
| `status` | `ok` or `error` |
| `error` | Diagnostic message when `status=error` |

## Python API

```python
from pathlib import Path
from ssrmsd import ScoringOptions, StructureInput, score_ensemble

result = score_ensemble(
    [
        StructureInput(Path("af3_mab1.tar.gz"), source="af3"),
        StructureInput(Path("esm_mab1/"), source="esmfold"),
    ],
    ensemble_id="mab1",
    options=ScoringOptions(align_chains=("A",), score_chains=("B", "C")),
    include_pairwise=True,
)
print(result.rms_rmsd, result.ssRMSD, result.cross_source_mean_rmsd)
```

## `brineylab/toolshed` Integration

To package `ssrmsd` as a CPU-only tool in `brineylab/toolshed` (`tools/ssrmsd/`):

**`Containerfile`** (no `ENTRYPOINT`; `ssrmsd` on `PATH`):
```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "git+https://github.com/brineylab/ssrmsd.git@v0.1.0"
```

Run locally with Podman or on Slurm with Pyxis:
```bash
podman run --rm -v "$PWD":/work -w /work \
  ghcr.io/brineylab/toolshed/ssrmsd:0.1.0 \
  ssrmsd batch --dir af3=af3_runs --dir esmfold=esm_runs -o scores.csv
```

## License

MIT.
