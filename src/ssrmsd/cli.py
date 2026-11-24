from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from ssrmsd.format import write_pairwise, write_results
from ssrmsd.io import discover_batch_dirs, load_manifest, parse_labeled_path
from ssrmsd.score import score_batch, score_ensemble
from ssrmsd.types import EnsembleResult, ScoringOptions, SsrmsdError


class OutputFormat(StrEnum):
    """Supported tabular/structured output formats."""

    csv = "csv"
    tsv = "tsv"
    jsonl = "jsonl"


class JoinMode(StrEnum):
    """How to combine ensemble IDs across multiple `--dir` roots."""

    union = "union"
    intersection = "intersection"


app = typer.Typer(
    name="ssrmsd",
    help="Compute normalized and legacy ssRMSD for predicted structure ensembles.",
    no_args_is_help=True,
    add_completion=False,
)


def _parse_chain_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    chains = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not chains:
        raise SsrmsdError("Chain list option must contain at least one chain ID")
    return chains


def _emit_outputs(
    results: Sequence[EnsembleResult],
    *,
    output: Path | None,
    pairwise_out: Path | None,
    fmt: OutputFormat,
) -> None:
    fmt_lit = fmt.value
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            write_results(results, handle, fmt=fmt_lit)
    else:
        write_results(results, sys.stdout, fmt=fmt_lit)

    if pairwise_out is not None:
        pairwise_out.parent.mkdir(parents=True, exist_ok=True)
        with pairwise_out.open("w", encoding="utf-8", newline="") as handle:
            write_pairwise(results, handle, fmt=fmt_lit)


@app.command("score")
def score_cmd(
    inputs: Annotated[
        list[str],
        typer.Argument(
            help="Structure files, directories, or archives (optionally prefixed with label=).",
        ),
    ],
    ensemble_id: Annotated[
        str | None,
        typer.Option("--id", help="Ensemble identifier for output rows."),
    ] = None,
    align_chains: Annotated[
        str | None,
        typer.Option(
            "--align-chains",
            help="Comma-separated reference chain IDs used for Kabsch superposition.",
        ),
    ] = None,
    score_chains: Annotated[
        str | None,
        typer.Option(
            "--score-chains",
            help="Comma-separated reference chain IDs over which RMSD is measured.",
        ),
    ] = None,
    ref_source: Annotated[
        str | None,
        typer.Option(
            "--ref-source",
            help="Source label providing reference chain IDs (strict: errors if absent).",
        ),
    ] = None,
    glob_pattern: Annotated[
        str | None,
        typer.Option(
            "--glob",
            help="Override structure discovery inside directories/archives with a glob pattern.",
        ),
    ] = None,
    include_top_level: Annotated[
        bool,
        typer.Option(
            "--include-top-level",
            help="Include top-level AF3 *_model.cif files alongside seed-*_sample-* outputs.",
        ),
    ] = False,
    split_models: Annotated[
        bool,
        typer.Option(
            "--split-models",
            help="Treat each MODEL block in a multi-model file as a separate ensemble member.",
        ),
    ] = False,
    min_chain_id: Annotated[
        float,
        typer.Option(
            "--min-chain-id",
            help="Minimum overlap sequence identity for chain matching.",
        ),
    ] = 0.90,
    min_coverage: Annotated[
        float,
        typer.Option(
            "--min-coverage",
            help="Minimum aligned CA coverage relative to max chain length.",
        ),
    ] = 0.80,
    linker_min_len: Annotated[
        int,
        typer.Option(
            "--linker-min-len",
            help="Minimum poly-G run length to split linked chains (0 disables).",
        ),
    ] = 10,
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format."),
    ] = OutputFormat.csv,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output path (defaults to stdout)."),
    ] = None,
    pairwise_out: Annotated[
        Path | None,
        typer.Option("--pairwise-out", help="Optional path to write all pairwise RMSD rows."),
    ] = None,
) -> None:
    """Score a single ensemble of predicted structures."""
    try:
        options = ScoringOptions(
            align_chains=_parse_chain_list(align_chains),
            score_chains=_parse_chain_list(score_chains),
            ref_source=ref_source,
            glob_pattern=glob_pattern,
            include_top_level=include_top_level,
            split_models=split_models,
            min_chain_id=min_chain_id,
            min_coverage=min_coverage,
            linker_min_len=linker_min_len,
        )
        res = score_ensemble(
            inputs,
            ensemble_id=ensemble_id,
            options=options,
            include_pairwise=pairwise_out is not None,
        )
        _emit_outputs([res], output=output, pairwise_out=pairwise_out, fmt=fmt)
    except SsrmsdError as exc:
        typer.echo(f"ssrmsd: error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if res.status != "ok":
        raise typer.Exit(code=1)


@app.command("batch")
def batch_cmd(
    dirs: Annotated[
        list[str] | None,
        typer.Option(
            "--dir",
            help="Batch root directory (repeatable; optionally prefixed with label=).",
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help="CSV/TSV manifest with columns ensemble_id,path[,source].",
        ),
    ] = None,
    join: Annotated[
        JoinMode,
        typer.Option("--join", help="How to combine ensemble IDs across multiple --dir roots."),
    ] = JoinMode.union,
    delimiter: Annotated[
        str | None,
        typer.Option("--delimiter", help="Explicit manifest delimiter (inferred by default)."),
    ] = None,
    align_chains: Annotated[
        str | None,
        typer.Option(
            "--align-chains",
            help="Comma-separated reference chain IDs used for Kabsch superposition.",
        ),
    ] = None,
    score_chains: Annotated[
        str | None,
        typer.Option(
            "--score-chains",
            help="Comma-separated reference chain IDs over which RMSD is measured.",
        ),
    ] = None,
    ref_source: Annotated[
        str | None,
        typer.Option(
            "--ref-source",
            help="Source label providing reference chain IDs (strict: errors if absent).",
        ),
    ] = None,
    glob_pattern: Annotated[
        str | None,
        typer.Option(
            "--glob",
            help="Override structure discovery inside directories/archives with a glob pattern.",
        ),
    ] = None,
    include_top_level: Annotated[
        bool,
        typer.Option(
            "--include-top-level",
            help="Include top-level AF3 *_model.cif files alongside seed-*_sample-* outputs.",
        ),
    ] = False,
    split_models: Annotated[
        bool,
        typer.Option(
            "--split-models",
            help="Treat each MODEL block in a multi-model file as a separate ensemble member.",
        ),
    ] = False,
    min_chain_id: Annotated[
        float,
        typer.Option(
            "--min-chain-id",
            help="Minimum overlap sequence identity for chain matching.",
        ),
    ] = 0.90,
    min_coverage: Annotated[
        float,
        typer.Option(
            "--min-coverage",
            help="Minimum aligned CA coverage relative to max chain length.",
        ),
    ] = 0.80,
    linker_min_len: Annotated[
        int,
        typer.Option(
            "--linker-min-len",
            help="Minimum poly-G run length to split linked chains (0 disables).",
        ),
    ] = 10,
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format."),
    ] = OutputFormat.csv,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output path (defaults to stdout)."),
    ] = None,
    pairwise_out: Annotated[
        Path | None,
        typer.Option("--pairwise-out", help="Optional path to write all pairwise RMSD rows."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Worker processes for batch execution."),
    ] = os.cpu_count() or 1,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Abort immediately if any ensemble fails."),
    ] = False,
) -> None:
    """Score a batch of ensembles from directories or a manifest."""
    if dirs and manifest is not None:
        typer.echo("ssrmsd: error: --dir and --manifest are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    if not dirs and manifest is None:
        typer.echo("ssrmsd: error: one of --dir or --manifest is required", err=True)
        raise typer.Exit(code=2)

    try:
        options = ScoringOptions(
            align_chains=_parse_chain_list(align_chains),
            score_chains=_parse_chain_list(score_chains),
            ref_source=ref_source,
            glob_pattern=glob_pattern,
            include_top_level=include_top_level,
            split_models=split_models,
            min_chain_id=min_chain_id,
            min_coverage=min_coverage,
            linker_min_len=linker_min_len,
        )
        if manifest is not None:
            specs = load_manifest(manifest, delimiter=delimiter)
        else:
            assert dirs is not None
            roots = [parse_labeled_path(raw) for raw in dirs]
            specs = discover_batch_dirs(roots, join=join.value)
        if not specs:
            typer.echo("ssrmsd: no ensembles discovered", err=True)
            raise typer.Exit(code=1)
        results = score_batch(
            specs,
            options=options,
            workers=workers,
            include_pairwise=pairwise_out is not None,
            fail_fast=fail_fast,
        )
        _emit_outputs(results, output=output, pairwise_out=pairwise_out, fmt=fmt)
    except SsrmsdError as exc:
        typer.echo(f"ssrmsd: error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
