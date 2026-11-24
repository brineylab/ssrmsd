from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

import gemmi
import numpy as np

from ssrmsd.types import ChainData, ScoringOptions, SsrmsdError, StructureData


@dataclass(frozen=True)
class AlignedEnsemble:
    """Ensemble structures reduced to common aligned/scored C-alpha coordinates."""

    names: tuple[str, ...]
    sources: tuple[str | None, ...]
    align_coords: tuple[np.ndarray, ...]
    score_coords: tuple[np.ndarray, ...]
    chain_mapping: tuple[dict[str, str], ...]


def _synthetic_chain_id(index: int) -> str:
    letters = string.ascii_uppercase
    if index < len(letters):
        return letters[index]
    return f"C{index}"


def _split_poly_g_linkers(
    structures: Sequence[StructureData],
    linker_min_len: int,
) -> list[StructureData]:
    if linker_min_len <= 0 or not structures:
        return list(structures)

    max_chains = max(len(s.chains) for s in structures)
    pattern = re.compile(f"G{{{linker_min_len},}}")
    result: list[StructureData] = []

    for s in structures:
        if len(s.chains) >= max_chains:
            result.append(s)
            continue

        new_chains: list[ChainData] = []
        did_split = False
        for chain in s.chains:
            spans = [m.span() for m in pattern.finditer(chain.sequence)]
            if not spans:
                new_chains.append(chain)
                continue
            segments: list[tuple[int, int]] = []
            cursor = 0
            for start, end in spans:
                if start > cursor:
                    segments.append((cursor, start))
                cursor = end
            if cursor < len(chain.sequence):
                segments.append((cursor, len(chain.sequence)))
            if len(segments) >= 2:
                did_split = True
                for seg_idx, (start, end) in enumerate(segments):
                    new_chains.append(
                        ChainData(
                            chain_id=f"{chain.chain_id}_{seg_idx}",
                            sequence=chain.sequence[start:end],
                            ca_coords=chain.ca_coords[start:end],
                        )
                    )
            else:
                new_chains.append(chain)

        if did_split:
            result.append(
                StructureData(
                    name=s.name,
                    source=s.source,
                    chains=tuple(new_chains),
                    was_split=True,
                )
            )
        else:
            result.append(s)

    return result


def _select_reference(
    structures: Sequence[StructureData],
    ref_source: str | None,
) -> tuple[int, StructureData]:
    if ref_source is not None:
        for idx, s in enumerate(structures):
            if s.source == ref_source:
                return idx, s
        raise SsrmsdError(f"ref-source '{ref_source}' not present in ensemble")

    for idx, s in enumerate(structures):
        if not s.was_split:
            return idx, s

    # Fallback: all structures were linker-split -> normalize chain IDs to A, B, C, ...
    first = structures[0]
    normalized_chains = tuple(
        ChainData(
            chain_id=_synthetic_chain_id(i),
            sequence=c.sequence,
            ca_coords=c.ca_coords,
        )
        for i, c in enumerate(first.chains)
    )
    return 0, StructureData(
        name=first.name,
        source=first.source,
        chains=normalized_chains,
        was_split=first.was_split,
    )


def _align_sequences(ref_seq: str, target_seq: str) -> tuple[float, dict[int, int]]:
    """Globally align two 1-letter sequences via gemmi; return overlap identity and index map."""
    if not ref_seq or not target_seq:
        return 0.0, {}
    res = gemmi.align_string_sequences(list(ref_seq), list(target_seq), [])
    g1 = res.add_gaps(ref_seq, 1)
    g2 = res.add_gaps(target_seq, 2)
    i = 0
    j = 0
    matches = 0
    aligned_pairs = 0
    idx_map: dict[int, int] = {}
    for c1, c2 in zip(g1, g2, strict=True):
        if c1 != "-" and c2 != "-":
            aligned_pairs += 1
            if c1 == c2:
                matches += 1
            idx_map[i] = j
            i += 1
            j += 1
        elif c1 != "-":
            i += 1
        elif c2 != "-":
            j += 1
    identity = (matches / aligned_pairs) if aligned_pairs > 0 else 0.0
    return identity, idx_map


def _resolve_chain_sets(
    ref: StructureData,
    options: ScoringOptions,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool]:
    ref_ids = tuple(c.chain_id for c in ref.chains)
    ref_id_set = set(ref_ids)
    has_selection = options.align_chains is not None or options.score_chains is not None

    if options.align_chains is not None:
        missing = [cid for cid in options.align_chains if cid not in ref_id_set]
        if missing:
            raise SsrmsdError(
                f"Unknown --align-chains ID(s) {missing} not in reference chains {list(ref_ids)}"
            )
        align_ids = tuple(dict.fromkeys(options.align_chains))
    else:
        align_ids = ref_ids

    if options.score_chains is not None:
        missing = [cid for cid in options.score_chains if cid not in ref_id_set]
        if missing:
            raise SsrmsdError(
                f"Unknown --score-chains ID(s) {missing} not in reference chains {list(ref_ids)}"
            )
        score_ids = tuple(dict.fromkeys(options.score_chains))
    else:
        score_ids = ref_ids

    required_set = set(align_ids) | set(score_ids)
    required_ids = tuple(cid for cid in ref_ids if cid in required_set)
    return align_ids, score_ids, required_ids, has_selection


def align_ensemble(
    structures: Sequence[StructureData],
    options: ScoringOptions,
) -> AlignedEnsemble:
    """Match chains by sequence across structures and extract common C-alpha coordinates."""
    if len(structures) < 2:
        raise SsrmsdError("fewer than 2 structures")

    prepped = _split_poly_g_linkers(structures, options.linker_min_len)
    ref_idx, ref = _select_reference(prepped, options.ref_source)
    if ref_idx == 0 and prepped[0] is not ref:
        prepped[0] = ref

    align_ids, score_ids, required_ids, has_selection = _resolve_chain_sets(ref, options)
    ref_chains_by_id = {c.chain_id: c for c in ref.chains}

    # Strict chain-count equality when no chain-selection flags are set.
    if not has_selection:
        expected_n = len(ref.chains)
        for s in prepped:
            if len(s.chains) != expected_n:
                raise SsrmsdError(
                    f"Chain count mismatch: '{s.name}' has {len(s.chains)} chains, "
                    f"reference '{ref.name}' has {expected_n}"
                )

    # Homomer guard on reference required chains.
    for i in range(len(required_ids)):
        for j in range(i + 1, len(required_ids)):
            c1 = ref_chains_by_id[required_ids[i]]
            c2 = ref_chains_by_id[required_ids[j]]
            ident, _ = _align_sequences(c1.sequence, c2.sequence)
            if ident >= options.min_chain_id:
                raise SsrmsdError(
                    f"Ambiguous/homomeric chains detected in reference ('{c1.chain_id}' and "
                    f"'{c2.chain_id}', identity={ident:.2f}); "
                    "homomer symmetry matching is not enabled"
                )

    # Match each structure's chains to required reference chains and record residue alignments.
    chain_mappings: list[dict[str, str]] = []
    # per_chain_maps[ref_cid][struct_idx] = (target_chain, {ref_pos: target_pos})
    per_chain_maps: dict[str, list[tuple[ChainData, dict[int, int]]]] = {
        cid: [] for cid in required_ids
    }

    for s_idx, s in enumerate(prepped):
        if s_idx == ref_idx:
            mapping = {cid: cid for cid in required_ids}
            chain_mappings.append(mapping)
            for cid in required_ids:
                rc = ref_chains_by_id[cid]
                per_chain_maps[cid].append((rc, {k: k for k in range(len(rc.sequence))}))
            continue

        # Compute overlap identity matrix between required ref chains and structure chains.
        ident_mat: list[list[float]] = []
        map_mat: list[list[dict[int, int]]] = []
        for r_cid in required_ids:
            r_seq = ref_chains_by_id[r_cid].sequence
            row_id: list[float] = []
            row_map: list[dict[int, int]] = []
            for t_chain in s.chains:
                ident, idx_map = _align_sequences(r_seq, t_chain.sequence)
                row_id.append(ident)
                row_map.append(idx_map)
            ident_mat.append(row_id)
            map_mat.append(row_map)

        # Check row/column uniqueness above threshold (homomer/ambiguity guard).
        mapping = {}
        used_targets: set[int] = set()
        for r_i, r_cid in enumerate(required_ids):
            hits = [t_j for t_j, val in enumerate(ident_mat[r_i]) if val >= options.min_chain_id]
            if len(hits) > 1:
                hit_ids = [s.chains[t_j].chain_id for t_j in hits]
                raise SsrmsdError(
                    f"Ambiguous/homomeric chains in '{s.name}': multiple chains {hit_ids} "
                    f"match reference chain '{r_cid}' above min-chain-id={options.min_chain_id}"
                )
            if not hits:
                best = max(ident_mat[r_i]) if ident_mat[r_i] else 0.0
                raise SsrmsdError(
                    f"No matching chain in '{s.name}' for reference chain '{r_cid}' "
                    f"(best identity={best:.2f} < {options.min_chain_id})"
                )
            t_j = hits[0]
            if t_j in used_targets:
                raise SsrmsdError(
                    f"Ambiguous/homomeric chain assignment in '{s.name}': "
                    f"chain '{s.chains[t_j].chain_id}' matches multiple reference chains"
                )
            used_targets.add(t_j)
            t_chain = s.chains[t_j]
            mapping[r_cid] = t_chain.chain_id
            per_chain_maps[r_cid].append((t_chain, map_mat[r_i][t_j]))

        chain_mappings.append(mapping)

    # Compute ensemble-wide C-alpha intersection and symmetric coverage check per required chain.
    common_coords_by_chain: dict[str, list[np.ndarray]] = {}
    for cid in required_ids:
        entries = per_chain_maps[cid]
        common_positions: set[int] = set(entries[0][1].keys())
        for _, idx_map in entries[1:]:
            common_positions &= set(idx_map.keys())
        sorted_pos = sorted(common_positions)
        max_chain_len = max(len(chain.sequence) for chain, _ in entries)
        coverage = (len(sorted_pos) / max_chain_len) if max_chain_len > 0 else 0.0
        if coverage < options.min_coverage or len(sorted_pos) == 0:
            raise SsrmsdError(
                f"Aligned CA coverage for chain '{cid}' is {coverage:.2f} "
                f"({len(sorted_pos)}/{max_chain_len}), below min-coverage={options.min_coverage}"
            )
        chain_coords_per_struct: list[np.ndarray] = []
        for chain, idx_map in entries:
            target_indices = [idx_map[p] for p in sorted_pos]
            chain_coords_per_struct.append(chain.ca_coords[target_indices])
        common_coords_by_chain[cid] = chain_coords_per_struct

    # Assemble concatenated alignment and scoring coordinate arrays per structure.
    n = len(prepped)
    align_coords: list[np.ndarray] = []
    score_coords: list[np.ndarray] = []
    for s_idx in range(n):
        a_arr = np.vstack([common_coords_by_chain[cid][s_idx] for cid in align_ids])
        s_arr = np.vstack([common_coords_by_chain[cid][s_idx] for cid in score_ids])
        align_coords.append(a_arr)
        score_coords.append(s_arr)

    return AlignedEnsemble(
        names=tuple(s.name for s in prepped),
        sources=tuple(s.source for s in prepped),
        align_coords=tuple(align_coords),
        score_coords=tuple(score_coords),
        chain_mapping=tuple(chain_mappings),
    )
