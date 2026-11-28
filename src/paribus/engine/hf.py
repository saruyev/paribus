"""Hugging Face Hub datasets as a source of sample records (needs the `paribus[hf]` extra).

Rows are streamed, so only the rows that are read get downloaded, and audio is not
decoded: each row's original audio file is written to a local cache once, byte for byte,
and records point at those files. The revision is pinned, so a cached file never goes
stale and a second run reads nothing from the network but the row list.
"""

import itertools
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import datasets
from pydantic import ValidationError

from paribus.engine.data import (
    MAX_ISSUES,
    HFConfig,
    RawRow,
    SourceError,
    SourceIssue,
    _describe,
)
from paribus.engine.types import SampleRecord

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "paribus" / "hf"


class HFDatasetSource:
    """Reads one split of a Hub dataset at a pinned revision.

    Audio files are cached under
    `<cache_dir>/<dataset>/<revision>/<subset>[/<data_dir>]/<split>/<row index>.<ext>`;
    the extension comes from the file name the dataset stores with the audio. Ids come from
    `id_column`, or are `<split>/<row index>`. The source is named after the dataset
    unless the config names it.
    """

    def __init__(self, config: HFConfig) -> None:
        """Keep the config; nothing is downloaded until `records()` is called."""
        self.config = config
        self.name = config.name or config.dataset
        self.license = config.license
        cache = (
            (config.cache_dir or DEFAULT_CACHE_DIR)
            / config.dataset
            / _safe(config.revision)
            / _safe(config.subset or "default")
        )
        if config.data_dir is not None:  # two folders of one dataset must not share a cache
            cache = cache / _safe(config.data_dir)
        self._cache = (cache / config.split).resolve()

    def records(self) -> Iterator[SampleRecord]:
        """Read and cache every row (up to `limit`), then yield the records in row order.

        Raises:
            SourceError: If a row lacks a configured column, its audio cannot be saved,
                or an id repeats.
        """
        records, issues = self._validate()
        if issues:
            raise SourceError(self.name, issues[:MAX_ISSUES], len(issues) > MAX_ISSUES)
        yield from records

    def _validate(self) -> tuple[list[SampleRecord], list[SourceIssue]]:
        """Build a record per row, collecting problems instead of stopping at the first."""
        records: list[SampleRecord] = []
        issues: list[SourceIssue] = []
        first_row_of: dict[str, int] = {}
        for index, row in enumerate(self._rows()):
            if len(issues) > MAX_ISSUES:
                break
            where = f"{self.config.dataset}[{self.config.split}]#{index}"
            try:
                record = self._to_record(index, row)
            except (ValidationError, ValueError, TypeError) as error:
                issues.append(SourceIssue(where, _describe(error)))
                continue
            if record.id in first_row_of:
                message = f"duplicate id {record.id!r}, first in row {first_row_of[record.id]}"
                issues.append(SourceIssue(where, message))
                continue
            first_row_of[record.id] = index
            records.append(record)
        return records, issues

    def _rows(self) -> Iterator[Mapping[str, Any]]:
        """Stream the split's rows with the audio column left undecoded, up to `limit`."""
        config = self.config
        stream = datasets.load_dataset(
            config.dataset,
            name=config.subset,
            data_dir=config.data_dir,
            split=config.split,
            revision=config.revision,
            streaming=True,
        )
        stream = stream.cast_column(config.audio_column, datasets.Audio(decode=False))
        return itertools.islice(stream, config.limit)

    def _to_record(self, index: int, row: Mapping[str, Any]) -> SampleRecord:
        """Build the record for one row: cache its audio, take the configured columns."""
        config = self.config
        record_id = (
            str(_column(row, config.id_column)) if config.id_column else f"{config.split}/{index}"
        )
        record: RawRow = {
            "id": record_id,
            "audio": self._cache_audio(index, _column(row, config.audio_column)),
            "language": config.language,
        }
        if config.transcripts != "none" and config.text_column is not None:
            text = _column(row, config.text_column)
            if config.transcripts == "reference":
                record["reference"] = text
            else:
                record["output"] = {"text": text}
        if config.meta_columns:
            record["meta"] = {name: _column(row, name) for name in config.meta_columns}
        if config.speech_segments_column is not None:
            spans = _column(row, config.speech_segments_column)
            record["speech_segments"] = _spans(spans, config.speech_segments_column)
        return SampleRecord.model_validate(record)

    def _cache_audio(self, index: int, audio: Any) -> Path:
        """Write the row's original audio file to the cache, unless it is there already.

        The file goes through a `.part` name and is renamed when complete, so a run that
        is interrupted never leaves a truncated file that a later run would trust.

        Raises:
            ValueError: If the value is not undecoded audio, has no file name to take the
                format from, or has neither bytes nor a readable local file.
        """
        if not isinstance(audio, Mapping):
            raise ValueError(f"audio column must hold audio, got {type(audio).__name__}")
        name: str = audio.get("path") or ""
        suffix = Path(name).suffix.lower()
        if not suffix:
            raise ValueError("cannot tell the audio format: the audio has no file name")
        target = self._cache / f"{index:06d}{suffix}"
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        if audio.get("bytes") is not None:
            partial.write_bytes(audio["bytes"])
        elif Path(name).is_file():
            shutil.copyfile(name, partial)
        else:
            raise ValueError(f"audio has no bytes and {name} is not a local file")
        partial.replace(target)
        return target


def _column(row: Mapping[str, Any], name: str) -> Any:
    """Return a row's value in column `name`.

    Raises:
        ValueError: If the row has no such column, listing the columns it has.
    """
    if name not in row:
        raise ValueError(f"no column {name!r}, columns are: {', '.join(row)}")
    return row[name]


def _spans(value: Any, column: str) -> list[tuple[Any, Any]]:
    """Read speech spans from a column value as (start, end) pairs.

    Accepted shapes: a list of `{start, end}` items, a list of `[start, end]` pairs, or
    the column-wise form `{start: [...], end: [...]}` that `datasets` uses for sequences
    of structs. Values are checked by `SampleRecord`, not here.

    Raises:
        ValueError: If the value has none of these shapes.
    """
    if isinstance(value, Mapping) and "start" in value and "end" in value:
        return list(zip(value["start"], value["end"], strict=True))
    if isinstance(value, list):
        if all(isinstance(item, Mapping) and "start" in item and "end" in item for item in value):
            return [(item["start"], item["end"]) for item in value]
        if all(isinstance(item, list | tuple) and len(item) == 2 for item in value):
            return [(start, end) for start, end in value]
    raise ValueError(f"column {column!r} must hold spans with start and end")


def _safe(part: str) -> str:
    """Make a revision or subset name usable as one folder name (`refs/pr/1` -> `refs--pr--1`)."""
    return part.replace("/", "--")
