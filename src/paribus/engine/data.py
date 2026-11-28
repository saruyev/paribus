"""Data loading: sources of sample records (manifests, audio folders, Hub datasets).

A source yields `SampleRecord`s and is built from one config object; the config models
validate what users write in YAML, so a typo in a key fails like a typo in a manifest.
`ManifestSource` reads a manifest file the user (or a dataset builder) wrote;
`FolderSource` turns a folder of audio files into records; `HFConfig` describes a
Hugging Face dataset, read by `paribus.engine.hf.HFDatasetSource`. Every source is
validated as a whole before the first record is yielded, so broken input fails at once
with every problem listed, not halfway through a run.
"""

import csv
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from paribus.engine.types import MetaValue, SampleRecord, normalize_language

type TimeUnit = Literal["s", "ms"]
type MetaType = Literal["str", "int", "float", "bool"]
type RawRow = dict[str, Any]
# A reader yields (line number, raw line content); a parser turns that content into a RawRow.
type Reader = Callable[[Path], Iterator[tuple[int, Any]]]
type Parser = Callable[[Any], RawRow]

# Stop collecting problems after this many; the rest of the file is still not trusted.
MAX_ISSUES = 20

_TIME_FACTORS: dict[TimeUnit, float] = {"s": 1.0, "ms": 0.001}
_CSV_META_PREFIX = "meta."
_CSV_OUTPUT_TEXT = "output_text"

# Audio formats picked up by a folder source unless its config says otherwise; matched
# case-insensitively, so `.WAV` counts.
AUDIO_PATTERNS = (
    "**/*.wav",
    "**/*.flac",
    "**/*.mp3",
    "**/*.ogg",
    "**/*.opus",
    "**/*.m4a",
    "**/*.webm",
)


class Source(Protocol):
    """Anything that yields sample records, with a name and a declared license."""

    name: str
    license: str

    def records(self) -> Iterator[SampleRecord]:
        """Yield every record of the source."""
        ...


@dataclass(frozen=True)
class SourceIssue:
    """One problem found in a source.

    `where` points at it: `file:line` for a manifest, the file path for a folder.
    """

    where: str
    message: str


class SourceError(ValueError):
    """A source failed validation; `issues` lists every problem found (up to a cap)."""

    def __init__(self, source: str, issues: list[SourceIssue], truncated: bool) -> None:
        """Build the error message from the issues, one per line."""
        self.source = source
        self.issues = issues
        self.truncated = truncated
        lines = [f"{issue.where}: {issue.message}" for issue in issues]
        if truncated:
            lines.append(f"... stopped after {len(issues)} problems")
        super().__init__(f"invalid source {source!r}\n" + "\n".join(lines))


class SourceConfig(BaseModel):
    """Settings every source has: its type, the license of its data and an optional name.

    `license` is required on purpose: every dataset paribus touches says under which terms,
    and `private` marks the user's own data, which never leaves their machine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    license: str = Field(min_length=1)
    name: str | None = None


class ManifestConfig(SourceConfig):
    """Settings of a `.jsonl` or `.csv` manifest source.

    Attributes:
        path: The manifest file; its suffix picks the format.
        time_unit: Unit of every time in the file (offset, speech segments, transcript
            timestamps). Records always come out in seconds.
        meta_types: Declared types of metadata keys. Values of declared keys are parsed
            (from CSV text) or checked (in JSONL); undeclared keys stay as they are, which
            in CSV means strings.
    """

    type: Literal["manifest"] = "manifest"
    path: Path
    time_unit: TimeUnit = "s"
    meta_types: dict[str, MetaType] = Field(default_factory=dict)


class ManifestSource:
    """Reads sample records from a `.jsonl` or `.csv` manifest.

    JSONL holds one record per line in the full `SampleRecord` shape, including a nested
    `output` transcript. CSV is flat: the record's scalar fields as columns, metadata as
    `meta.<key>` columns, and the pipeline's output text as `output_text`; an empty cell
    means "not set". Relative audio paths are resolved against the manifest's folder.
    The source is named after the file (without suffix) unless the config names it.
    """

    def __init__(self, config: ManifestConfig) -> None:
        """Keep the config; the file is read only when `records()` is called.

        Raises:
            ValueError: If the file is neither `.jsonl` nor `.csv`.
        """
        if config.path.suffix not in _FORMATS:
            raise ValueError(f"manifest must be .jsonl or .csv, got {config.path.name}")
        self.config = config
        self.name = config.name or config.path.stem
        self.license = config.license
        self._read, self._parse = _FORMATS[config.path.suffix]

    def records(self) -> Iterator[SampleRecord]:
        """Validate the whole manifest, then yield its records in file order.

        Raises:
            SourceError: If any line is invalid, an id repeats or an audio file is missing.
        """
        records, issues = self._validate()
        if issues:
            raise SourceError(self.name, issues[:MAX_ISSUES], len(issues) > MAX_ISSUES)
        yield from records

    def _validate(self) -> tuple[list[SampleRecord], list[SourceIssue]]:
        """Read and check every line, collecting problems instead of stopping at the first.

        All problems take one path: parsing, conversion and schema validation raise, and
        the error becomes an issue for that line. Collection stops one problem past
        `MAX_ISSUES`, so the caller can tell the list was cut short.
        """
        records: list[SampleRecord] = []
        issues: list[SourceIssue] = []
        first_line_of: dict[str, int] = {}
        for line, raw in self._read(self.config.path):
            if len(issues) > MAX_ISSUES:
                break
            where = f"{self.config.path}:{line}"
            try:
                record = self._to_record(self._parse(raw))
            except (ValidationError, ValueError, TypeError) as error:
                issues.append(SourceIssue(where, _describe(error)))
                continue
            problem = _cross_record_problem(record, line, first_line_of)
            if problem:
                issues.append(SourceIssue(where, problem))
                continue
            records.append(record)
        return records, issues

    def _to_record(self, row: RawRow) -> SampleRecord:
        """Apply unit conversion, metadata types and path resolution, then validate."""
        row = _scale_times(row, _TIME_FACTORS[self.config.time_unit])
        if "meta" in row:
            row["meta"] = _coerce_meta(row["meta"], self.config.meta_types)
        if "audio" in row:
            row["audio"] = _resolve(Path(row["audio"]), self.config.path.parent)
        return SampleRecord.model_validate(row)


class FolderConfig(SourceConfig):
    """Settings of a folder of audio files.

    Attributes:
        root: The folder; it is searched recursively.
        patterns: Glob patterns of audio files, relative to `root`.
        transcripts: What a text file next to an audio file is: a gold `reference`, the
            pipeline's `output`, or `none` (ignored). There is no guessing, because
            mistaking pipeline output for gold would compare the pipeline with itself.
        transcript_suffix: Suffix of those text files: `a/001.wav` pairs with `a/001.txt`.
        language: Expected language of every record in the folder, if known.
    """

    type: Literal["folder"] = "folder"
    root: Path
    patterns: tuple[str, ...] = AUDIO_PATTERNS
    transcripts: Literal["none", "reference", "output"] = "none"
    transcript_suffix: str = ".txt"
    language: str | None = None

    @field_validator("language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        """Validate the folder's language once here, not once per file."""
        return None if value is None else normalize_language(value)


class FolderSource:
    """Turns every audio file under a folder into a record.

    A record's id is the file's path relative to the root without its suffix
    (`calls/2026/001`): stable across runs and unique, except when two files differ only
    in suffix (`a.wav` and `a.mp3`), which is reported as an error. Files are yielded in
    path order. Hidden files and folders (names starting with `.`, which includes the
    `._*` files macOS leaves on external drives) are skipped, and symlinked folders are not
    followed. The source is named after the folder unless the config names it.
    """

    def __init__(self, config: FolderConfig) -> None:
        """Keep the config; the folder is searched only when `records()` is called."""
        self.config = config
        self.name = config.name or config.root.name
        self.license = config.license

    def records(self) -> Iterator[SampleRecord]:
        """Validate every file under the folder, then yield the records in path order.

        Raises:
            SourceError: If the folder has no audio, two files map to one id, or a
                transcript cannot be read.
        """
        records, issues = self._validate()
        if issues:
            raise SourceError(self.name, issues[:MAX_ISSUES], len(issues) > MAX_ISSUES)
        yield from records

    def _validate(self) -> tuple[list[SampleRecord], list[SourceIssue]]:
        """Build a record per audio file, collecting problems instead of stopping."""
        root = self.config.root
        if not root.is_dir():
            return [], [SourceIssue(str(root), "folder not found")]
        files = _find_audio(root, self.config.patterns)
        if not files:
            patterns = ", ".join(self.config.patterns)
            return [], [SourceIssue(str(root), f"no audio files match {patterns}")]
        records: list[SampleRecord] = []
        issues: list[SourceIssue] = []
        first_file_of: dict[str, Path] = {}
        for path in files:
            if len(issues) > MAX_ISSUES:
                break
            try:
                record = self._to_record(path)
            except (ValidationError, ValueError) as error:
                issues.append(SourceIssue(str(path), _describe(error)))
                continue
            if record.id in first_file_of:
                message = f"same id {record.id!r} as {first_file_of[record.id]}"
                issues.append(SourceIssue(str(path), message))
                continue
            first_file_of[record.id] = path
            records.append(record)
        return records, issues

    def _to_record(self, path: Path) -> SampleRecord:
        """Build the record for one audio file, with its transcript if configured."""
        row: RawRow = {
            "id": path.relative_to(self.config.root).with_suffix("").as_posix(),
            "audio": path.resolve(),
            "language": self.config.language,
        }
        if self.config.transcripts != "none":
            text = _read_transcript(path.with_suffix(self.config.transcript_suffix))
            if text is not None and self.config.transcripts == "reference":
                row["reference"] = text
            elif text is not None:
                row["output"] = {"text": text}
        return SampleRecord.model_validate(row)


class HFConfig(SourceConfig):
    """Settings of a Hugging Face Hub dataset; the source itself is in `paribus.engine.hf`.

    The config lives here, without the `datasets` dependency, so that a config with
    `type: hf` is validated even where the `paribus[hf]` extra is not installed.

    Attributes:
        dataset: Repository id on the Hub, e.g. `openslr/librispeech_asr`.
        revision: Commit sha or tag to read. Required, because a dataset can change under
            a branch name, and then two runs would quietly read different data.
        subset: The dataset's configuration (`name` in `datasets.load_dataset`), if any.
        data_dir: A folder inside the dataset to read, for datasets laid out as folders
            rather than configurations, such as the parquet conversion of a script
            dataset (`google/fleurs` at `refs/convert/parquet` keeps `ru_ru/test/...`).
        split: Split to read, e.g. `test`.
        audio_column: Column holding the audio.
        text_column: Column holding a transcript; needed when `transcripts` is not `none`.
        transcripts: What `text_column` is: a gold `reference` or a pipeline `output`.
        id_column: Column holding record ids; without it ids are `<split>/<index>`.
        language: Expected language of every record, if known.
        meta_columns: Columns copied into each record's `meta` (speaker, sex, a clip's
            own license), so slice search can use them. Values must be scalars.
        speech_segments_column: Column holding ground-truth speech spans, e.g. forced
            alignments of words: a list of `{start, end}` items (other keys ignored) or
            of `[start, end]` pairs, in seconds. It becomes `speech_segments`.
        limit: Read only the first `limit` rows, so a big dataset is not read in full.
        cache_dir: Where audio files are written; defaults to `~/.cache/paribus/hf`.
    """

    type: Literal["hf"] = "hf"
    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    subset: str | None = None
    data_dir: str | None = None
    split: str = Field(min_length=1)
    audio_column: str = "audio"
    text_column: str | None = None
    transcripts: Literal["none", "reference", "output"] = "none"
    id_column: str | None = None
    language: str | None = None
    meta_columns: list[str] = Field(default_factory=list)
    speech_segments_column: str | None = None
    limit: int | None = Field(default=None, ge=1)
    cache_dir: Path | None = None

    @field_validator("language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        """Validate the dataset's language once here, not once per row."""
        return None if value is None else normalize_language(value)

    @model_validator(mode="after")
    def _transcripts_need_a_column(self) -> Self:
        """Reject a transcript role without the column that holds the transcripts."""
        if self.transcripts != "none" and self.text_column is None:
            raise ValueError(f"transcripts={self.transcripts!r} needs text_column")
        return self


def _hf_source(config: HFConfig) -> Source:
    """Build a `HFDatasetSource`, importing it only now, since it needs `paribus[hf]`.

    Raises:
        ImportError: If the `datasets` package is not installed.
    """
    try:
        from paribus.engine.hf import HFDatasetSource
    except ModuleNotFoundError as error:
        if error.name != "datasets":
            raise
        raise ImportError("source type 'hf' needs the extra: pip install 'paribus[hf]'") from None
    return HFDatasetSource(config)


# Source types by the `type` key of their config: the config class that validates the
# settings, and the factory that builds the source from a validated config.
_SOURCE_TYPES: dict[str, tuple[type[SourceConfig], Callable[[Any], Source]]] = {}


def register_source[C: SourceConfig](
    source_type: str, config_cls: type[C], factory: Callable[[C], Source]
) -> None:
    """Make `source_type` buildable by `build_source`.

    The registry is open so that new kinds of sources (a database, an object store) can be
    added without changing paribus. A type can be registered only once, so a plugin cannot
    silently replace a built-in source.

    Raises:
        ValueError: If `source_type` is already registered.
    """
    if source_type in _SOURCE_TYPES:
        raise ValueError(f"source type {source_type!r} is already registered")
    _SOURCE_TYPES[source_type] = (config_cls, factory)


def build_source(raw: Mapping[str, Any]) -> Source:
    """Build a source from its settings as written in YAML, picking the class by `type`.

    The whole mapping, `type` included, is validated by the registered config class, so
    unknown keys and wrong values fail here, before any data is read.

    Raises:
        ValueError: If `type` is missing or not registered.
        ValidationError: If the settings do not match the type's config.
    """
    source_type = raw.get("type")
    known = ", ".join(sorted(_SOURCE_TYPES))
    if source_type is None:
        raise ValueError(f"source config needs a 'type', one of: {known}")
    if source_type not in _SOURCE_TYPES:
        raise ValueError(f"unknown source type {source_type!r}, known: {known}")
    config_cls, factory = _SOURCE_TYPES[source_type]
    return factory(config_cls.model_validate(raw))


register_source("manifest", ManifestConfig, ManifestSource)
register_source("folder", FolderConfig, FolderSource)
register_source("hf", HFConfig, _hf_source)


def _find_audio(root: Path, patterns: Sequence[str]) -> list[Path]:
    """Return the audio files under `root` matching any pattern, sorted, without duplicates.

    Matching ignores case. Files inside hidden folders and hidden files are left out.
    `Path.glob` does not follow symlinked folders for `**`, which keeps the search from
    looping or wandering off the folder.
    """
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern, case_sensitive=False):
            hidden = any(part.startswith(".") for part in path.relative_to(root).parts)
            if path.is_file() and not hidden:
                found.add(path)
    return sorted(found)


def _read_transcript(path: Path) -> str | None:
    """Return the stripped text of a transcript file, or None if there is none.

    A missing file is not an error: gold transcripts often exist for only some files.

    Raises:
        ValueError: If the file exists but is not UTF-8 text.
    """
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        raise ValueError(f"transcript {path.name} is not UTF-8 text") from None


def _cross_record_problem(
    record: SampleRecord, line: int, first_line_of: dict[str, int]
) -> str | None:
    """Return what is wrong with a valid record in the context of the file, if anything.

    Checks what one record cannot check on its own: that its id is not taken by an earlier
    line, and that its audio file exists. Records the id's first line in `first_line_of`.
    """
    if record.id in first_line_of:
        return f"duplicate id {record.id!r}, first on line {first_line_of[record.id]}"
    first_line_of[record.id] = line
    if not record.audio.is_file():
        return f"audio file not found: {record.audio}"
    return None


def _read_jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (line number, text) for every non-blank line; parsing is left to the caller."""
    with path.open(encoding="utf-8") as lines:
        for number, text in enumerate(lines, start=1):
            if text.strip():
                yield number, text


def _parse_jsonl_line(text: str) -> RawRow:
    """Parse one JSONL line into a dict.

    Raises:
        ValueError: If the line is not JSON or not a JSON object.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not valid JSON: {error.msg}") from None
    if not isinstance(value, dict):
        raise ValueError("each line must be a JSON object")
    return value


def _read_csv_rows(path: Path) -> Iterator[tuple[int, dict[str | None, Any]]]:
    """Yield (line number, cells by column) for every CSV data row.

    The line number is where the row ends, which differs from its start only for quoted
    cells that contain line breaks.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for cells in reader:
            yield reader.line_num, cells


def _csv_row_to_record_shape(cells: Mapping[str | None, str | list[str] | None]) -> RawRow:
    """Map one CSV row to the nested shape `SampleRecord` expects.

    Empty cells are dropped, `meta.<key>` columns are gathered into `meta`, and
    `output_text` becomes a transcript with text only. Extra cells beyond the header
    (DictReader puts them under the key None) are kept under an invalid key, so that
    validation reports them instead of silently dropping them.
    """
    row: RawRow = {}
    meta: dict[str, str] = {}
    for column, value in cells.items():
        if column is None:
            row["<cells beyond header>"] = value
        elif value is None or value == "":
            continue
        elif column.startswith(_CSV_META_PREFIX):
            meta[column.removeprefix(_CSV_META_PREFIX)] = str(value)
        elif column == _CSV_OUTPUT_TEXT:
            row["output"] = {"text": value}
        else:
            row[column] = value
    if meta:
        row["meta"] = meta
    return row


_FORMATS: dict[str, tuple[Reader, Parser]] = {
    ".jsonl": (_read_jsonl_lines, _parse_jsonl_line),
    ".csv": (_read_csv_rows, _csv_row_to_record_shape),
}


def _scale_times(row: RawRow, factor: float) -> RawRow:
    """Return a copy of the row with every time field multiplied by `factor`.

    Covers `offset`, `speech_segments` and the start/end of transcript segments and words.
    Values that are not numbers are left for validation to reject.
    """
    if factor == 1.0:
        return dict(row)
    scaled = dict(row)
    if _is_number(scaled.get("offset")):
        scaled["offset"] = scaled["offset"] * factor
    if isinstance(scaled.get("speech_segments"), list):
        scaled["speech_segments"] = [
            _scale_pair(pair, factor) for pair in scaled["speech_segments"]
        ]
    output = scaled.get("output")
    if isinstance(output, dict) and isinstance(output.get("segments"), list):
        scaled["output"] = {
            **output,
            "segments": [_scale_span(s, factor) for s in output["segments"]],
        }
    return scaled


def _scale_span(item: Any, factor: float) -> Any:
    """Scale `start`/`end` of a segment or word dict, and of its nested words."""
    if not isinstance(item, dict):
        return item
    scaled = dict(item)
    for key in ("start", "end"):
        if _is_number(scaled.get(key)):
            scaled[key] = scaled[key] * factor
    if isinstance(scaled.get("words"), list):
        scaled["words"] = [_scale_span(word, factor) for word in scaled["words"]]
    return scaled


def _scale_pair(pair: Any, factor: float) -> Any:
    """Scale a [start, end] pair; anything else is returned unchanged for validation."""
    if isinstance(pair, list | tuple) and all(_is_number(v) for v in pair):
        return [v * factor for v in pair]
    return pair


def _is_number(value: Any) -> bool:
    """True for int and float, but not bool (which is an int subclass in Python)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _coerce_meta(meta: Any, types: Mapping[str, MetaType]) -> Any:
    """Parse or check the values of declared metadata keys; leave the rest untouched."""
    if not isinstance(meta, dict):
        return meta
    return {
        key: _coerce(key, value, types[key]) if key in types else value
        for key, value in meta.items()
    }


def _coerce(key: str, value: Any, meta_type: MetaType) -> MetaValue:
    """Return `value` as `meta_type`: parse it if it is text, check it otherwise.

    Raises:
        ValueError: If the value cannot be read as the declared type.
    """
    try:
        if isinstance(value, str):
            return _parse(value, meta_type)
        return _check(value, meta_type)
    except ValueError:
        raise ValueError(f"meta.{key}: {value!r} is not a valid {meta_type}") from None


def _parse(text: str, meta_type: MetaType) -> MetaValue:
    """Parse metadata text as the declared type; booleans accept only true/false."""
    match meta_type:
        case "str":
            return text
        case "int":
            return int(text)
        case "float":
            return float(text)
        case "bool":
            lowered = text.strip().lower()
            if lowered not in ("true", "false"):
                raise ValueError(text)
            return lowered == "true"


def _check(value: Any, meta_type: MetaType) -> MetaValue:
    """Check a native JSON value against the declared type; an int is accepted as float."""
    ok = {
        "str": isinstance(value, str),
        "int": _is_number(value) and isinstance(value, int),
        "float": _is_number(value),
        "bool": isinstance(value, bool),
    }[meta_type]
    if not ok:
        raise ValueError(value)
    return float(value) if meta_type == "float" else value


def _resolve(audio: Path, base_dir: Path) -> Path:
    """Make a relative audio path absolute against the manifest's folder."""
    return audio if audio.is_absolute() else (base_dir / audio).resolve()


def _describe(error: Exception) -> str:
    """Turn a validation error into one readable line: `field: message; field: message`."""
    if not isinstance(error, ValidationError):
        return str(error)
    parts = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "record"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)
