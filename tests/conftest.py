from __future__ import annotations

import gzip
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import gemmi
import numpy as np

AA1_TO_AA3: dict[str, str] = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
}

AG_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALE"
HC_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSL"
LC_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATY"


def make_helix_coords(
    n: int,
    *,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 2.3,
    rise: float = 1.5,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate deterministic non-collinear 3D C-alpha helix coordinates."""
    t = np.arange(n, dtype=np.float64) * 1.7453 + phase
    x = radius * np.cos(t) + offset[0]
    y = radius * np.sin(t) + offset[1]
    z = rise * np.arange(n, dtype=np.float64) + offset[2]
    return np.column_stack([x, y, z])


def rotation_matrix(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around a unit axis."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    omc = 1.0 - c
    return np.array(
        [
            [c + ux * ux * omc, ux * uy * omc - uz * s, ux * uz * omc + uy * s],
            [uy * ux * omc + uz * s, c + uy * uy * omc, uy * uz * omc - ux * s],
            [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz * uz * omc],
        ],
        dtype=np.float64,
    )


def apply_rigid_transform(
    coords: np.ndarray,
    *,
    rot: np.ndarray | None = None,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply a rigid rotation and translation to `(N, 3)` coordinates."""
    out = np.asarray(coords, dtype=np.float64)
    if rot is not None:
        out = out @ rot.T
    return out + np.asarray(shift, dtype=np.float64)


def build_gemmi_structure(
    models: Sequence[Sequence[tuple[str, str, np.ndarray]]],
    *,
    add_water: bool = False,
    add_altloc_b: bool = False,
    resname_overrides: dict[tuple[str, int], str] | None = None,
    omit_ca: set[tuple[str, int]] | None = None,
) -> gemmi.Structure:
    """Build a valid gemmi Structure from models -> chains -> (chain_id, seq, coords)."""
    st = gemmi.Structure()
    st.name = "TEST"
    for m_idx, chains in enumerate(models, start=1):
        model = gemmi.Model(m_idx)
        for chain_id, seq, coords in chains:
            chain = gemmi.Chain(chain_id)
            for idx, (aa, xyz) in enumerate(zip(seq, coords, strict=True), start=1):
                res = gemmi.Residue()
                res.name = (
                    resname_overrides[(chain_id, idx)]
                    if resname_overrides and (chain_id, idx) in resname_overrides
                    else AA1_TO_AA3.get(aa, "UNK")
                )
                res.seqid = gemmi.SeqId(str(idx))
                res.entity_type = gemmi.EntityType.Polymer
                if not omit_ca or (chain_id, idx) not in omit_ca:
                    atom = gemmi.Atom()
                    atom.name = "CA"
                    atom.element = gemmi.Element("C")
                    atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    atom.occ = 0.6 if add_altloc_b else 1.0
                    atom.b_iso = 20.0
                    if add_altloc_b:
                        atom.altloc = "A"
                    res.add_atom(atom)
                    if add_altloc_b:
                        atom_b = gemmi.Atom()
                        atom_b.name = "CA"
                        atom_b.element = gemmi.Element("C")
                        atom_b.pos = gemmi.Position(
                            float(xyz[0]) + 5.0, float(xyz[1]) + 5.0, float(xyz[2]) + 5.0
                        )
                        atom_b.occ = 0.4
                        atom_b.b_iso = 20.0
                        atom_b.altloc = "B"
                        res.add_atom(atom_b)
                else:
                    atom_n = gemmi.Atom()
                    atom_n.name = "N"
                    atom_n.element = gemmi.Element("N")
                    atom_n.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    atom_n.occ = 1.0
                    atom_n.b_iso = 20.0
                    res.add_atom(atom_n)
                chain.add_residue(res)
            if add_water:
                hoh = gemmi.Residue()
                hoh.name = "HOH"
                hoh.seqid = gemmi.SeqId("999")
                hoh.entity_type = gemmi.EntityType.Water
                o_atom = gemmi.Atom()
                o_atom.name = "O"
                o_atom.element = gemmi.Element("O")
                o_atom.pos = gemmi.Position(50.0, 50.0, 50.0)
                o_atom.occ = 1.0
                o_atom.b_iso = 30.0
                hoh.add_atom(o_atom)
                chain.add_residue(hoh)
            model.add_chain(chain)
        st.add_model(model)
    st.setup_entities()
    return st


def write_cif(
    path: Path,
    chains: Sequence[tuple[str, str, np.ndarray]],
    *,
    extra_models: Sequence[Sequence[tuple[str, str, np.ndarray]]] = (),
    gzipped: bool = False,
    **kwargs: object,
) -> Path:
    """Write a synthetic mmCIF file (plain or gzipped)."""
    st = build_gemmi_structure([chains, *extra_models], **kwargs)  # type: ignore[arg-type]
    text = st.make_mmcif_document().as_string()
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        path.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def write_pdb(
    path: Path,
    chains: Sequence[tuple[str, str, np.ndarray]],
    *,
    extra_models: Sequence[Sequence[tuple[str, str, np.ndarray]]] = (),
    gzipped: bool = False,
    **kwargs: object,
) -> Path:
    """Write a synthetic PDB file (plain or gzipped)."""
    st = build_gemmi_structure([chains, *extra_models], **kwargs)  # type: ignore[arg-type]
    text = st.make_pdb_string()
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        path.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def make_complex_chains(
    *,
    ag_seq: str = AG_SEQ[:30],
    hc_seq: str = HC_SEQ[:25],
    lc_seq: str = LC_SEQ[:25],
    ab_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise_seed: int | None = None,
    noise_scale: float = 0.0,
) -> list[tuple[str, str, np.ndarray]]:
    """Build a synthetic 3-chain (A=Ag, B=HC, C=LC) complex."""
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    ag = make_helix_coords(len(ag_seq), offset=(0.0, 0.0, 0.0), phase=0.0)
    hc = make_helix_coords(len(hc_seq), offset=(15.0, 5.0, 0.0), phase=1.0)
    lc = make_helix_coords(len(lc_seq), offset=(15.0, -5.0, 0.0), phase=2.0)
    shift = np.asarray(ab_shift, dtype=np.float64)
    hc = hc + shift
    lc = lc + shift
    if rng is not None and noise_scale > 0.0:
        ag = ag + rng.normal(0.0, noise_scale, size=ag.shape)
        hc = hc + rng.normal(0.0, noise_scale, size=hc.shape)
        lc = lc + rng.normal(0.0, noise_scale, size=lc.shape)
    return [("A", ag_seq, ag), ("B", hc_seq, hc), ("C", lc_seq, lc)]


def make_af3_job_dir(
    root: Path,
    job_name: str,
    *,
    n_samples: int = 5,
    include_top_level_dup: bool = True,
    ab_shifts: Sequence[tuple[float, float, float]] | None = None,
) -> Path:
    """Create a synthetic local AF3 job directory with sample subdirs + top-level duplicate."""
    job_dir = root / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    shifts = (
        list(ab_shifts)
        if ab_shifts is not None
        else [(float(i), 0.0, 0.0) for i in range(n_samples)]
    )
    first_chains: list[tuple[str, str, np.ndarray]] | None = None
    for idx in range(n_samples):
        chains = make_complex_chains(ab_shift=shifts[idx], noise_seed=100 + idx, noise_scale=0.1)
        if first_chains is None:
            first_chains = chains
        sample_dir = job_dir / f"seed-1_sample-{idx}"
        write_cif(sample_dir / f"{job_name}_seed-1_sample-{idx}_model.cif", chains)
    if include_top_level_dup and first_chains is not None:
        write_cif(job_dir / f"{job_name}_model.cif", first_chains)
    return job_dir


def make_tar_gz(src_dir: Path, tar_path: Path) -> Path:
    """Archive `src_dir` into a `.tar.gz` file."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)
    return tar_path


def make_zip(files: Sequence[tuple[str, Path]], zip_path: Path) -> Path:
    """Create a `.zip` archive from `(arcname, file_path)` pairs."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arcname, fp in files:
            zf.write(fp, arcname=arcname)
    return zip_path
