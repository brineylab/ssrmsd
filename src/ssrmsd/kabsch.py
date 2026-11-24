from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ssrmsd.align import AlignedEnsemble
from ssrmsd.types import EnsembleResult, PairwiseDistance, SsrmsdError


def kabsch_rmsd(
    p_align: np.ndarray,
    q_align: np.ndarray,
    p_score: np.ndarray | None = None,
    q_score: np.ndarray | None = None,
) -> float:
    """Compute float64 Kabsch RMSD aligning `p_align` onto `q_align`."""
    p_a = np.asarray(p_align, dtype=np.float64)
    q_a = np.asarray(q_align, dtype=np.float64)
    if p_a.shape != q_a.shape or p_a.ndim != 2 or p_a.shape[1] != 3 or len(p_a) == 0:
        raise SsrmsdError(f"Invalid alignment coordinate shapes: {p_a.shape} vs {q_a.shape}")

    p_s = p_a if p_score is None else np.asarray(p_score, dtype=np.float64)
    q_s = q_a if q_score is None else np.asarray(q_score, dtype=np.float64)
    if p_s.shape != q_s.shape or p_s.ndim != 2 or p_s.shape[1] != 3 or len(p_s) == 0:
        raise SsrmsdError(f"Invalid scoring coordinate shapes: {p_s.shape} vs {q_s.shape}")

    if np.array_equal(p_a, q_a) and np.array_equal(p_s, q_s):
        return 0.0

    mu_p = p_a.mean(axis=0)
    mu_q = q_a.mean(axis=0)
    pc = p_a - mu_p
    qc = q_a - mu_q

    cov = pc.T @ qc
    u, _, vt = np.linalg.svd(cov)
    if np.linalg.det(vt.T @ u.T) < 0.0:
        vt[-1, :] *= -1.0
    rot = vt.T @ u.T

    p_s_aligned = (p_s - mu_p) @ rot.T + mu_q
    diff = p_s_aligned - q_s
    sq_sum = max(0.0, float(np.sum(diff * diff)))
    return math.sqrt(sq_sum / len(p_s))


def compute_pairwise_matrix(aligned: AlignedEnsemble) -> np.ndarray:
    """Compute symmetric `(n, n)` float64 pairwise Kabsch RMSD matrix with zero diagonal."""
    n = len(aligned.names)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = kabsch_rmsd(
                aligned.align_coords[i],
                aligned.align_coords[j],
                aligned.score_coords[i],
                aligned.score_coords[j],
            )
            mat[i, j] = d
            mat[j, i] = d
    return mat


def aggregate_metrics(
    ensemble_id: str,
    aligned: AlignedEnsemble,
    matrix: np.ndarray,
    *,
    include_pairwise: bool = False,
) -> EnsembleResult:
    """Aggregate normalized, legacy, and multi-source ensemble RMSD metrics."""
    n = len(aligned.names)
    if n < 2:
        raise SsrmsdError("fewer than 2 structures")

    iu, ju = np.triu_indices(n, k=1)
    upper = matrix[iu, ju]

    mean_sq = float(np.mean(upper * upper))
    rms = math.sqrt(max(0.0, mean_sq))
    mean_val = float(np.mean(upper))
    median_val = float(np.median(upper))
    rmsd_sd = float(np.std(upper, ddof=0))
    legacy_ss = float(np.sum(matrix * matrix))
    legacy_sd = float(np.std(matrix, ddof=0))

    labeled = [s for s in aligned.sources if s is not None]
    if labeled:
        counts = Counter(labeled)
        sources_str = ",".join(f"{k}:{counts[k]}" for k in sorted(counts))
    else:
        sources_str = ""

    within_mean: float | None = None
    cross_mean: float | None = None
    distinct_sources = set(labeled)
    if len(distinct_sources) >= 2:
        within_vals: list[float] = []
        cross_vals: list[float] = []
        for i, j in zip(iu.tolist(), ju.tolist(), strict=True):
            si = aligned.sources[i]
            sj = aligned.sources[j]
            if si is None or sj is None:
                continue
            val = float(matrix[i, j])
            if si == sj:
                within_vals.append(val)
            else:
                cross_vals.append(val)
        if within_vals:
            within_mean = float(np.mean(within_vals))
        if cross_vals:
            cross_mean = float(np.mean(cross_vals))

    pairwise: tuple[PairwiseDistance, ...] = ()
    if include_pairwise:
        pairwise = tuple(
            PairwiseDistance(
                ensemble_id=ensemble_id,
                struct_i=aligned.names[i],
                struct_j=aligned.names[j],
                source_i=aligned.sources[i],
                source_j=aligned.sources[j],
                rmsd=float(matrix[i, j]),
            )
            for i, j in zip(iu.tolist(), ju.tolist(), strict=True)
        )

    return EnsembleResult(
        ensemble_id=ensemble_id,
        n_structures=n,
        n_ca_align=int(len(aligned.align_coords[0])),
        n_ca_score=int(len(aligned.score_coords[0])),
        mean_sq_rmsd=mean_sq,
        rms_rmsd=rms,
        mean_rmsd=mean_val,
        median_rmsd=median_val,
        rmsd_sd=rmsd_sd,
        ssRMSD=legacy_ss,
        SD=legacy_sd,
        within_source_mean_rmsd=within_mean,
        cross_source_mean_rmsd=cross_mean,
        sources=sources_str,
        status="ok",
        error="",
        pairwise=pairwise,
    )
