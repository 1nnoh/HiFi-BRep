from __future__ import annotations

import json
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

from src.data.brep_dataset import validate_brep_record
from src.preprocessing.brep import discover_step_files, output_relative_path, parse_step_file


@dataclass(frozen=True)
class PreprocessJob:
    input_path: Path
    output_path: Path
    input_relative: str
    output_relative: str
    uid: str
    max_face: int


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_job(job: PreprocessJob) -> dict[str, object]:
    try:
        record = parse_step_file(job.input_path, uid=job.uid, max_face=job.max_face)
        _atomic_pickle(job.output_path, record)
        return {
            "input": job.input_relative,
            "output": job.output_relative,
            "status": "written",
        }
    except Exception as exc:
        return {
            "input": job.input_relative,
            "output": job.output_relative,
            "status": "failed",
            "failure_type": type(exc).__name__,
        }


def _validate_existing(path: Path, *, max_face: int) -> None:
    with path.open("rb") as stream:
        record = pickle.load(stream)
    validate_brep_record(record, max_face=max_face)


def run_preprocessing(
    *,
    input_root: str | Path,
    output_root: str | Path,
    layout: str,
    workers: int,
    max_face: int,
    resume: bool,
    limit: int | None = None,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    input_root_path = Path(input_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    discovered = discover_step_files(input_root_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        discovered = discovered[:limit]
    jobs: list[PreprocessJob] = []
    skipped: list[dict[str, object]] = []
    planned_outputs: set[Path] = set()
    for relative_input in discovered:
        relative_output = output_relative_path(relative_input, layout=layout)
        output_path = (output_root_path / relative_output).resolve()
        if not output_path.is_relative_to(output_root_path):
            raise ValueError(f"Output path escapes output root: {relative_output}")
        if output_path in planned_outputs:
            raise ValueError(f"Multiple STEP files map to the same output: {relative_output}")
        planned_outputs.add(output_path)
        if output_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Output sample already exists; use --resume to validate and skip it: {relative_output}"
                )
            _validate_existing(output_path, max_face=max_face)
            skipped.append(
                {
                    "input": relative_input.as_posix(),
                    "output": relative_output.as_posix(),
                    "status": "skipped_valid",
                }
            )
            continue
        uid = relative_output.stem
        jobs.append(
            PreprocessJob(
                input_path=input_root_path / relative_input,
                output_path=output_path,
                input_relative=relative_input.as_posix(),
                output_relative=relative_output.as_posix(),
                uid=uid,
                max_face=max_face,
            )
        )

    results: list[dict[str, object]] = []
    if workers == 1:
        results = [_run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {executor.submit(_run_job, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    records = sorted(skipped + results, key=lambda item: str(item["input"]))
    counts = {
        status: sum(item["status"] == status for item in records)
        for status in ("written", "skipped_valid", "failed")
    }
    summary = {
        "format_version": 1,
        "layout": layout,
        "max_face": max_face,
        "workers": workers,
        "discovered": len(discovered),
        "counts": counts,
        "records": records,
    }
    output_root_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_root_path / "preprocess_summary.json"
    temporary_path = output_root_path / ".preprocess_summary.json.tmp"
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, summary_path)
    return summary
