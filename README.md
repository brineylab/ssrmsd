# ssRMSD

## AF3 antibody structure pipeline (ZIP -> CIF -> PDB -> RMSD stats)

Small, single-file pipeline to:
1) extract `.cif` files from ZIP archives from AF3 server,
2) convert CIF -> PDB (selected chains),
3) compute per-antibody RMSD summary stats across multiple predicted models.

Outputs a CSV with RMSD-derived metrics (sum of squares and standard deviation) per mAb name.

## What it expects

Directory layout under `--base-dir` (defaults to `./preds/af3/`):

```

archived/        # input ZIPs live here
extracted/       # CIFs extracted here (auto-created)
pdbs/            # converted PDBs written here (auto-created)
output/          # results written here (auto-created)
logs/            # ssRMSD.log written here (auto-created)
run_cache.json   # cache file (auto-created)

```

ZIP archives should contain `.cif` files. PDB filenames are derived from the CIF basename.

RMSD computation assumes 5 models per antibody named like:
`<mab>_model_0.pdb ... <mab>_model_4.pdb`

## Installation

Python 3.9+ recommended.

```bash
pip install numpy pandas biopython
````

## Usage

Run with defaults:

```bash
python ssRMSD.py
```

Point at a different base directory:

```bash
python ssRMSD.py --base-dir /path/to/preds/af3
```

Tune concurrency:

```bash
python ssRMSD.py --threads 16 --processes 8
```

Force RMSD recomputation (ignores cached names):

```bash
python ssRMSD.py --force-recalc
```

## Outputs

* `output/rmsd_stats.csv`

  * `mab`: antibody name (basename before `_model_`)
  * `sum_of_squares`: sum of squared RMSDs across the model-vs-model matrix
  * `std_dev`: standard deviation of RMSDs in the matrix
* `logs/ssRMSD.log`
* `run_cache.json`

* tracks extracted ZIPs and mAbs already analyzed

## Notes / assumptions

* CIF -> PDB conversion writes only chains `A`, `B`, `C` (see `CHAINS_TO_WRITE`). For working with Fv (HC + LC) with monomeric antigen; can be changed.
* RMSD uses C-alpha atoms only (`atom.get_name() == "CA"`), aligned with `Bio.PDB.Superimposer`.
* If any of the expected model PDBs for an mAb are missing, that mAb is skipped for RMSD.

## License
MIT.
