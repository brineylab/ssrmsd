from __future__ import annotations

import csv
import fnmatch
import gzip
import re
import tarfile
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import gemmi
import numpy as np

from ssrmsd.types import ChainData, EnsembleSpec, ScoringOptions, SsrmsdError, StructureData, StructureInput

SUPPORTED_STRUCTURE_SUFFIXES: tuple[str, ...] = (
    ".cif.gz",
    ".mmcif.gz",
    ".pdb.gz",
    ".ent.gz",
    ".cif",
    ".mmcif",
    ".pdb",
    ".ent",
)

SUPPORTED_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tgz",
    ".tar",
    ".zip",
)

_AF3_SAMPLE_RE = re.compile(r"(^|/)seed-[^/]+_sample-\d+/")
_MODEL_GROUP_RE = re.compile(r"^(.+)_model_\d+$")


def _match_suffix(name: str, suffixes: tuple[str, ...]) -> str | None:
    lower = name.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return suffix
    return None


def is_structure_filename(name: str) -> bool:
    """Return True if filename has a supported structure extension."""
    return _match_suffix(name, SUPPORTED_STRUCTURE_SUFFIXES) is not None


def is_archive_filename(name: str) -> bool:
    """Return True if filename has a supported archive extension."""
    return _match_suffix(name, SUPPORTED_ARCHIVE_SUFFIXES) is not None


def strip_known_suffix(name: str) -> str:
    """Strip a supported archive or structure suffix from a filename."""
    for suffixes in (SUPPORTED_ARCHIVE_SUFFIXES, SUPPORTED_STRUCTURE_SUFFIXES):
        matched = _match_suffix(name, suffixes)
        if matched is not None:
            return name[: -len(matched)]
    return Path(name).stem


def parse_labeled_path(raw: str) -> StructureInput:
    """Parse an optional `label=path` CLI token into a `StructureInput`.

    If the raw string already exists on disk or has a path separator before `=`,
    treat the entire string as an unlabeled path so directories like `run_lr=1e-4/job`
    are never misparsed as source labels.
    """
    if "=" in raw:
        candidate_path = Path(raw)
        if not candidate_path.exists():
            label, rest = raw.split("=", 1)
            if label and "/" not in label and "\\" not in label and rest:
                return StructureInput(path=Path(rest), source=label)
    return StructureInput(path=Path(raw), source=None)


def _extract_chains(model: gemmi.Model) -> tuple[ChainData, ...]:
    chains: list[ChainData] = []
    for chain in model:
        seq_chars: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for res in chain:
            if res.entity_type in (gemmi.EntityType.Water, gemmi.EntityType.NonPolymer):
                continue
            tab = gemmi.find_tabulated_residue(res.name)
            if tab.is_water():
                continue
            ca_atom: gemmi.Atom | None = None
            for atom in res.first_conformer():
                if atom.name == "CA":
                    ca_atom = atom
                    break
            if ca_atom is None:
                continue
            code = tab.one_letter_code.upper()
            if not code or code in (" ", "?", "."):
                code = "X"
            seq_chars.append(code)
            coords.append((ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z))
        if coords:
            chains.append(
                ChainData(
                    chain_id=chain.name,
                    sequence="".join(seq_chars),
                    ca_coords=np.asarray(coords, dtype=np.float64),
                )
            )
    return tuple(chains)


def _parse_gemmi_structure(text: str, filename: str) -> gemmi.Structure:
    lower = filename.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    try:
        if lower.endswith((".cif", ".mmcif")):
            doc = gemmi.cif.read_string(text)
            st = gemmi.make_structure_from_block(doc.sole_block())
        elif lower.endswith((".pdb", ".ent")):
            st = gemmi.read_pdb_string(text)
        else:
            raise SsrmsdError(f"Unsupported structure format: {filename}")
    except SsrmsdError:
        raise
    except Exception as exc:
        raise SsrmsdError(f"Failed to parse structure '{filename}': {exc}") from exc
    st.setup_entities()
    return st


def read_structures_from_bytes(
    data: bytes,
    filename: str,
    *,
    source: str | None = None,
    split_models: bool = False,
) -> list[StructureData]:
    """Parse one structure file payload into one (or more, if split_models) StructureData."""
    try:
        raw_bytes = gzip.decompress(data) if filename.lower().endswith(".gz") else data
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise SsrmsdError(f"Failed to decompress structure '{filename}': {exc}") from exc

    st = _parse_gemmi_structure(text, filename)
    if len(st) == 0:
        raise SsrmsdError(f"Structure '{filename}' contains no models")

    models = list(st) if split_models else [st[0]]
    results: list[StructureData] = []
    for idx, model in enumerate(models):
        chains = _extract_chains(model)
        if not chains:
            raise SsrmsdError(f"Structure '{filename}' model {idx} contains no polymer CA atoms")
        name = f"{filename}#model{idx}" if split_models and len(models) > 1 else filename
        results.append(StructureData(name=name, source=source, chains=chains))
    return results


def _filter_member_paths(
    relpaths: Sequence[str],
    options: ScoringOptions,
) -> list[str]:
    cleaned = [p.lstrip("./") for p in relpaths]
    candidates = [
        p
        for p in cleaned
        if not p.startswith("__MACOSX/") and is_structure_filename(p)
    ]
    if options.glob_pattern is not None:
        matched = [
            p
            for p in candidates
            if fnmatch.fnmatch(p, options.glob_pattern)
            or fnmatch.fnmatch(Path(p).name, options.glob_pattern)
        ]
        return sorted(matched)

    if not options.include_top_level:
        af3_samples = [p for p in candidates if _AF3_SAMPLE_RE.search(p)]
        if af3_samples:
            return sorted(af3_samples)

    return sorted(candidates)


def _load_from_tar(path: Path, source: str | None, options: ScoringOptions) -> list[StructureData]:
    try:
        with tarfile.open(path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            by_name = {m.name.lstrip("./"): m for m in members}
            selected = _filter_member_paths(list(by_name.keys()), options)
            out: list[StructureData] = []
            for rel in selected:
                extracted = tf.extractfile(by_name[rel])
                if extracted is None:
                    continue
                payload = extracted.read()
                out.extend(
                    read_structures_from_bytes(
                        payload,
                        rel,
                        source=source,
                        split_models=options.split_models,
                    )
                )
            return out
    except SsrmsdError:
        raise
    except Exception as exc:
        raise SsrmsdError(f"Failed to read tar archive '{path}': {exc}") from exc


def _load_from_zip(path: Path, source: str | None, options: ScoringOptions) -> list[StructureData]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            by_name = {info.filename.lstrip("./"): info for info in infos}
            selected = _filter_member_paths(list(by_name.keys()), options)
            out: list[StructureData] = []
            for rel in selected:
                payload = zf.read(by_name[rel])
                out.extend(
                    read_structures_from_bytes(
                        payload,
                        rel,
                        source=source,
                        split_models=options.split_models,
                    )
                )
            return out
    except SsrmsdError:
        raise
    except Exception as exc:
        raise SsrmsdError(f"Failed to read zip archive '{path}': {exc}") from exc


def _load_from_directory(
    directory: Path,
    source: str | None,
    options: ScoringOptions,
) -> list[StructureData]:
    all_files = [p for p in directory.rglob("*") if p.is_file()]
    rel_to_abs = {p.relative_to(directory).as_posix(): p for p in all_files}
    selected = _filter_member_paths(list(rel_to_abs.keys()), options)
    out: list[StructureData] = []
    for rel in selected:
        abs_path = rel_to_abs[rel]
        out.extend(
            read_structures_from_bytes(
                abs_path.read_bytes(),
                rel,
                source=source,
                split_models=options.split_models,
            )
        )
    return out


def load_ensemble_structures(
    inputs: Sequence[StructureInput],
    options: ScoringOptions,
) -> list[StructureData]:
    """Expand files, directories, and archives into parsed `StructureData` entries."""
    structures: list[StructureData] = []
    for inp in inputs:
        path = inp.path
        if not path.exists():
            raise SsrmsdError(f"Input path does not exist: {path}")
        if path.is_dir():
            structures.extend(_load_from_directory(path, inp.source, options))
        elif is_archive_filename(path.name):
            if path.name.lower().endswith(".zip"):
                structures.extend(_load_from_zip(path, inp.source, options))
            else:
                structures.extend(_load_from_tar(path, inp.source, options))
        elif is_structure_filename(path.name):
            structures.extend(
                read_structures_from_bytes(
                    path.read_bytes(),
                    path.name,
                    source=inp.source,
                    split_models=options.split_models,
                )
            )
        else:
            raise SsrmsdError(f"Unsupported input file: {path}")
    return structures


def _derive_batch_ensemble_id(child: Path) -> str | None:
    if child.is_dir():
        return child.name
    if is_archive_filename(child.name):
        return strip_known_suffix(child.name)
    if is_structure_filename(child.name):
        stem = strip_known_suffix(child.name)
        m = _MODEL_GROUP_RE.match(stem)
        return m.group(1) if m else stem
    return None


def discover_batch_dirs(
    roots: Sequence[StructureInput],
    *,
    join: Literal["union", "intersection"] = "union",
) -> list[EnsembleSpec]:
    """Discover ensembles across one or more batch root directories."""
    if not roots:
        raise SsrmsdError("At least one --dir root is required")

    per_root: list[dict[str, list[StructureInput]]] = []
    for root in roots:
        if not root.path.is_dir():
            raise SsrmsdError(f"Batch root is not a directory: {root.path}")
        bucket: dict[str, list[StructureInput]] = {}
        for child in sorted(root.path.iterdir(), key=lambda p: p.name):
            if child.name.startswith("."):
                continue
            eid = _derive_batch_ensemble_id(child)
            if eid is None:
                continue
            bucket.setdefault(eid, []).append(StructureInput(path=child, source=root.source))
        per_root.append(bucket)

    if join == "intersection":
        shared_ids = set(per_root[0].keys())
        for bucket in per_root[1:]:
            shared_ids &= set(bucket.keys())
        ensemble_ids = sorted(shared_ids)
    elif join == "union":
        all_ids: set[str] = set()
        for bucket in per_root:
            all_ids.update(bucket.keys())
        ensemble_ids = sorted(all_ids)
    else:
        raise SsrmsdError(f"Unsupported join mode: {join}")

    specs: list[EnsembleSpec] = []
    for eid in ensemble_ids:
        merged: list[StructureInput] = []
        for bucket in per_root:
            merged.extend(bucket.get(eid, ()))
        specs.append(EnsembleSpec(ensemble_id=eid, inputs=tuple(merged)))
    return specs


def load_manifest(path: Path, *, delimiter: str | None = None) -> list[EnsembleSpec]:
    """Load batch `EnsembleSpec` definitions from a CSV or TSV manifest file."""
    if not path.is_file():
        raise SsrmsdError(f"Manifest file does not exist: {path}")
    sep = delimiter if delimiter is not None else ("\t" if path.suffix.lower() in (".tsv", ".tab") else ",")
    grouped: OrderedDict[str, list[StructureInput]] = OrderedDict()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=sep)
            fieldnames = set(reader.fieldnames or ())
            missing = {"ensemble_id", "path"} - fieldnames
            if missing:
                raise SsrmsdError(
                    f"Manifest '{path}' missing required columns: {', '.join(sorted(missing))}"
                )
            for row_num, row in enumerate(reader, start=2):
                eid = (row.get("ensemble_id") or "").strip()
                raw_path = (row.get("path") or "").strip()
                raw_source = (row.get("source") or "").strip() or None
                if not eid or not raw_path:
                    raise SsrmsdError(
                        f"Manifest '{path}' row {row_num} has empty ensemble_id or path"
                    )
                grouped.setdefault(eid, []).append(
                    StructureInput(path=Path(raw_path), source=raw_source)
                )
    except SsrmsdError:
        raise
    except Exception as exc:
        raise SsrmsdError(f"Failed to read manifest '{path}': {exc}") from exc

    return [EnsembleSpec(ensemble_id=eid, inputs=tuple(inps)) for eid, inps in grouped.items()]
