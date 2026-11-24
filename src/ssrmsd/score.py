from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ssrmsd.align import align_ensemble
from ssrmsd.io import load_ensemble_structures, parse_labeled_path, strip_known_suffix
from ssrmsd.kabsch import aggregate_metrics, compute_pairwise_matrix
from ssrmsd.types import (
    EnsembleResult,
    EnsembleSpec,
    ScoringOptions,
    SsrmsdError,
    StructureData,
    StructureInput,
)


def _normalize_inputs(
    inputs: Sequence[StructureInput | Path | str],
) -> tuple[StructureInput, ...]:
    normalized: list[StructureInput] = []
    for item in inputs:
        if isinstance(item, StructureInput):
            normalized.append(item)
        elif isinstance(item, Path):
            normalized.append(StructureInput(path=item, source=None))
        elif isinstance(item, str):
            normalized.append(parse_labeled_path(item))
        else:
            raise SsrmsdError(f"Unsupported input type: {type(item).__name__}")
    return tuple(normalized)


def _default_ensemble_id(inputs: Sequence[StructureInput]) -> str:
    if len(inputs) == 1:
        p = inputs[0].path
        return p.name if p.is_dir() else strip_known_suffix(p.name)
    stems = {p.path.name if p.path.is_dir() else strip_known_suffix(p.path.name) for p in inputs}
    if len(stems) == 1:
        return next(iter(stems))
    return "ensemble"


def _format_sources(structures: Sequence[StructureData]) -> str:
    labeled = [s.source for s in structures if s.source is not None]
    if not labeled:
        return ""
    counts = Counter(labeled)
    return ",".join(f"{k}:{counts[k]}" for k in sorted(counts))


def score_ensemble(
    inputs: Sequence[StructureInput | Path | str],
    *,
    ensemble_id: str | None = None,
    options: ScoringOptions | None = None,
    include_pairwise: bool = False,
) -> EnsembleResult:
    """Score a single ensemble from files, directories, and/or archives."""
    opts = options or ScoringOptions()
    try:
        norm_inputs = _normalize_inputs(inputs)
    except Exception as exc:
        return EnsembleResult.from_error(ensemble_id or "ensemble", error=str(exc))

    eid = ensemble_id if ensemble_id is not None else _default_ensemble_id(norm_inputs)
    structures: list[StructureData] = []
    try:
        structures = load_ensemble_structures(norm_inputs, opts)
        if len(structures) < 2:
            return EnsembleResult.from_error(
                eid,
                n_structures=len(structures),
                sources=_format_sources(structures),
                error=f"fewer than 2 structures ({len(structures)} found)",
            )
        aligned = align_ensemble(structures, opts)
        matrix = compute_pairwise_matrix(aligned)
        return aggregate_metrics(eid, aligned, matrix, include_pairwise=include_pairwise)
    except SsrmsdError as exc:
        return EnsembleResult.from_error(
            eid,
            n_structures=len(structures),
            sources=_format_sources(structures),
            error=str(exc),
        )
    except Exception as exc:
        return EnsembleResult.from_error(
            eid,
            n_structures=len(structures),
            sources=_format_sources(structures),
            error=f"unexpected error: {exc}",
        )


def _score_spec_worker(
    args: tuple[EnsembleSpec, ScoringOptions, bool],
) -> EnsembleResult:
    spec, opts, include_pairwise = args
    return score_ensemble(
        spec.inputs,
        ensemble_id=spec.ensemble_id,
        options=opts,
        include_pairwise=include_pairwise,
    )


def score_batch(
    ensembles: Sequence[EnsembleSpec],
    *,
    options: ScoringOptions | None = None,
    workers: int = 1,
    include_pairwise: bool = False,
    fail_fast: bool = False,
) -> list[EnsembleResult]:
    """Score a batch of ensembles deterministically, optionally across CPU workers."""
    opts = options or ScoringOptions()
    if not ensembles:
        return []

    if workers <= 1 or len(ensembles) == 1:
        results: list[EnsembleResult] = []
        for spec in ensembles:
            res = score_ensemble(
                spec.inputs,
                ensemble_id=spec.ensemble_id,
                options=opts,
                include_pairwise=include_pairwise,
            )
            if fail_fast and res.status == "error":
                raise SsrmsdError(f"{res.ensemble_id}: {res.error}")
            results.append(res)
        return results

    payloads = [(spec, opts, include_pairwise) for spec in ensembles]
    results = []
    executor = ProcessPoolExecutor(max_workers=workers)
    cancelled = False
    try:
        futures = [executor.submit(_score_spec_worker, p) for p in payloads]
        for fut in futures:
            res = fut.result()
            if fail_fast and res.status == "error":
                cancelled = True
                executor.shutdown(wait=False, cancel_futures=True)
                raise SsrmsdError(f"{res.ensemble_id}: {res.error}")
            results.append(res)
    finally:
        if not cancelled:
            executor.shutdown(wait=True)
    return results
