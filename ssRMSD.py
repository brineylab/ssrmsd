import zipfile
import os
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Generator, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
import re

# -------------------------
# Configuration & Arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Unified pipeline for extracting, converting, and analyzing AF3 antibody structures."
)
parser.add_argument(
    "--base-dir",
    default="./preds/af3/",
    help="Root directory for predictions",
)
parser.add_argument(
    "--cache-file",
    default="run_cache.json",
    help="Name of cache JSON file (saved under base-dir)",
)
parser.add_argument(
    "--threads",
    type=int,
    default=12,
    help="Number of threads for CIF -> PDB conversion",
)
parser.add_argument(
    "--processes", type=int, default=6, help="Number of processes for RMSD computation"
)
parser.add_argument(
    "--force-recalc", action="store_true", help="Force recalculation of RMSD matrices"
)
args = parser.parse_args()

BASE_DIR = Path(args.base_dir)
ARCHIVE_DIR = BASE_DIR / "archived"
EXTRACT_DIR = BASE_DIR / "extracted"
PDB_DIR = BASE_DIR / "pdbs"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
CACHE_PATH = BASE_DIR / args.cache_file
CIF_EXT = ".cif"
PDB_EXT = ".pdb"
CHAINS_TO_WRITE = ["A", "B", "C"]
MODELS_PER_MAB = 5

# -------------------------
# Logging Setup
# -------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "ssRMSD.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# -------------------------
# Cache Management
# -------------------------
def load_cache() -> Dict[str, List[str]]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            log.info("Loading existing cache.")
            return json.load(f)
    return {"extracted_zips": [], "rmsd_names": []}


def save_cache(cache: Dict[str, List[str]]) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# -------------------------
# Extraction
# -------------------------
def extract_new_zips(cache: Dict[str, List[str]]) -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    all_zips = sorted([f for f in ARCHIVE_DIR.iterdir() if f.suffix == ".zip"])
    new_zips = [z for z in all_zips if z.name not in cache["extracted_zips"]]
    if not new_zips:
        log.info("No new ZIP archives to extract.")
        return
    for zip_path in new_zips:
        extracted = False
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(CIF_EXT):
                    target = EXTRACT_DIR / zip_path.stem / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        with open(target, "wb") as out:
                            out.write(zf.read(name))
                        extracted = True
        if extracted:
            log.info(f"Extracted archive: {zip_path.name}")
            cache["extracted_zips"].append(zip_path.name)
    save_cache(cache)


# -------------------------
# CIF to PDB Conversion
# -------------------------
def convert_cif_to_pdb(
    filepath: str, chainids: Optional[List[str]] = None
) -> Generator[str, None, None]:
    template = (
        "{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   "
        "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:<4s}{:<2s}{:2s}\n"
    )
    try:
        with open(filepath, "r") as fhandle:
            in_section = False
            read_atom = False
            label_pos = 0
            labels = {}
            serial = 0
            empty = {".", "?"}
            for line in fhandle:
                if line.startswith("loop_"):
                    in_section = True
                elif line.startswith("#"):
                    in_section, read_atom = False, False
                elif in_section and line.startswith("_atom_site."):
                    labels[line.strip()] = label_pos
                    label_pos += 1
                    read_atom = True
                elif read_atom:
                    fields = re.findall(r'[^"\s]\S*|".+?"', line)
                    record = fields[labels["_atom_site.group_PDB"]]
                    current_chainid = fields[
                        labels.get(
                            "_atom_site.auth_asym_id",
                            labels.get("_atom_site.label_asym_id"),
                        )
                    ]
                    if chainids and current_chainid not in chainids:
                        continue
                    serial += 1
                    atname = fields[
                        labels.get(
                            "_atom_site.auth_atom_id",
                            labels.get("_atom_site.label_atom_id"),
                        )
                    ]
                    element = fields[labels["_atom_site.type_symbol"]]
                    element = element if element not in empty else " "
                    if len(atname) < 4 and atname[0].isalpha() and len(element) < 2:
                        atname = " " + atname
                    altloc = fields[labels["_atom_site.label_alt_id"]]
                    altloc = altloc if altloc not in empty else " "
                    resname = fields[
                        labels.get(
                            "_atom_site.auth_comp_id",
                            labels.get("_atom_site.label_comp_id"),
                        )
                    ]
                    resnum = int(
                        fields[
                            labels.get(
                                "_atom_site.auth_seq_id",
                                labels.get("_atom_site.label_seq_id"),
                            )
                        ]
                    )
                    icode = fields[labels["_atom_site.pdbx_PDB_ins_code"]]
                    icode = icode if icode not in empty else " "
                    x = float(fields[labels["_atom_site.Cartn_x"]])
                    y = float(fields[labels["_atom_site.Cartn_y"]])
                    z = float(fields[labels["_atom_site.Cartn_z"]])
                    occ = float(fields[labels["_atom_site.occupancy"]])
                    bfactor = float(fields[labels["_atom_site.B_iso_or_equiv"]])
                    charge = (
                        fields[labels.get("_atom_site.pdbx_formal_charge")]
                        if "_atom_site.pdbx_formal_charge" in labels
                        else "  "
                    )
                    yield template.format(
                        record,
                        serial,
                        atname,
                        altloc,
                        resname,
                        current_chainid,
                        resnum,
                        icode,
                        x,
                        y,
                        z,
                        occ,
                        bfactor,
                        current_chainid,
                        element,
                        charge,
                    )
        yield "END\n"
    except Exception as e:
        log.error(f"Failed to convert {filepath}: {e}")


def convert_single_cif(cif_file: Path) -> str:
    pdb_file = PDB_DIR / cif_file.with_suffix(PDB_EXT).name
    if pdb_file.exists():
        return f"Exists: {pdb_file}"
    with open(pdb_file, "w") as f:
        for line in convert_cif_to_pdb(str(cif_file), CHAINS_TO_WRITE):
            f.write(line)
    return f"Converted: {pdb_file}"


def convert_all_cif(cache: Dict[str, List[str]]) -> None:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    cif_files = list(EXTRACT_DIR.rglob(f"*{CIF_EXT}"))
    total = len(cif_files)
    pdb_files = set(p.name for p in PDB_DIR.glob("*.pdb"))
    to_convert = [f for f in cif_files if f.with_suffix(PDB_EXT).name not in pdb_files]

    log.info(f"Total CIF files found: {total}")
    log.info(f"Previously converted PDBs: {total - len(to_convert)}")
    log.info(f"New CIFs to convert in this run: {len(to_convert)}")

    if not to_convert:
        log.info("No new CIF files to convert.")
        return

    progress_checkpoints = set(
        int(len(to_convert) * i / 5) for i in range(1, 6)
    )  # Roughly every 20%

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(convert_single_cif, f): idx
            for idx, f in enumerate(to_convert)
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if i in progress_checkpoints:
                percent = int((i / len(to_convert)) * 100)
                log.info(f"CIF conversion progress: {percent}%")


# -------------------------
# RMSD Analysis
# -------------------------
parser = PDBParser(QUIET=True)


def extract_ca(pdb_path: Path) -> List:
    structure = parser.get_structure("model", str(pdb_path))
    return [atom for atom in structure.get_atoms() if atom.get_name() == "CA"]


def rmsd_pair(a: List, b: List) -> float:
    if len(a) != len(b):
        raise ValueError("Atom count mismatch")
    sup = Superimposer()
    sup.set_atoms(a, b)
    return sup.rms


def compute_for_name(name: str) -> Optional[Tuple[str, float, float]]:
    models = [PDB_DIR / f"{name}_model_{i}{PDB_EXT}" for i in range(MODELS_PER_MAB)]
    if not all(m.exists() for m in models):
        return None
    ca_list = [extract_ca(m) for m in models]
    size = len(ca_list)
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mat[i, j] = rmsd_pair(ca_list[i], ca_list[j])
    sos = round(float(np.sum(mat**2)), 2)
    std = round(float(np.std(mat)), 2)
    return name, sos, std


def analyze_all(cache: Dict[str, List[str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_file = OUTPUT_DIR / "rmsd_stats.csv"
    if stats_file.exists() and not args.force_recalc:
        existing_df = pd.read_csv(stats_file)
    else:
        existing_df = pd.DataFrame(columns=["mab", "sum_of_squares", "std_dev"])
    all_pdbs = list(PDB_DIR.glob(f"*{PDB_EXT}"))
    names = sorted(set(p.stem.split("_model_")[0] for p in all_pdbs))
    to_compute = [n for n in names if n not in cache["rmsd_names"] or args.force_recalc]
    if not to_compute:
        log.info("No new models for RMSD calculation.")
        log.info("No new calculation made!")  # Added log message
        return
    log.info(f"Computing RMSD for {len(to_compute)} mAbs...")
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {executor.submit(compute_for_name, n): n for n in to_compute}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                name, sos, std = res
                results.append((name, sos, std))
                cache["rmsd_names"].append(name)
    new_df = pd.DataFrame(results, columns=["mab", "sum_of_squares", "std_dev"])
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(stats_file, index=False)
    save_cache(cache)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    cache = load_cache()
    extract_new_zips(cache)
    convert_all_cif(cache)
    analyze_all(cache)
    log.info("Pipeline completed successfully!")
