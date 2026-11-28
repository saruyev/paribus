"""Tests for paribus.engine.hf, with a local dataset standing in for the Hub."""

import sys
from pathlib import Path
from typing import Any

import datasets
import pytest
from pydantic import ValidationError

from paribus.engine.data import HFConfig, SourceError, build_source
from paribus.engine.hf import HFDatasetSource

_DATASET = "org/speech"
_REVISION = "a1b2c3"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `datasets.load_dataset` with a stream over rows a test puts in `hub["rows"]`.

    The fake keeps the arguments of its last call in `hub["call"]`, so tests can check
    what would have been asked of the Hub.
    """
    hub: dict[str, Any] = {"rows": {}, "call": None}

    def load_dataset(path: str, **kwargs: Any) -> Any:
        """Return the test's rows as a stream with an encoded audio column."""
        hub["call"] = {"path": path, **kwargs}
        local = datasets.Dataset.from_dict(hub["rows"]).cast_column("audio", datasets.Audio())
        return local.to_iterable_dataset()

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return hub


def _wav(folder: Path, *names: str) -> list[str]:
    """Create placeholder `.wav` files and return their paths; audio is never decoded here."""
    paths = []
    for name in names:
        path = folder / f"{name}.wav"
        path.write_bytes(b"RIFF" + name.encode())
        paths.append(str(path))
    return paths


def _source(tmp_path: Path, **settings: Any) -> HFDatasetSource:
    """A source for the fake dataset, caching under `tmp_path/cache`."""
    base = {
        "dataset": _DATASET,
        "revision": _REVISION,
        "split": "test",
        "license": "CC-BY-4.0",
        "cache_dir": tmp_path / "cache",
    }
    return HFDatasetSource(HFConfig(**{**base, **settings}))


def test_rows_become_records_with_cached_original_audio(
    tmp_path: Path, hub: dict[str, Any]
) -> None:
    """Each row's audio file is copied to the cache unchanged; ids default to split/index."""
    hub["rows"] = {"audio": _wav(tmp_path, "x", "y")}

    records = list(_source(tmp_path, language="en").records())

    assert [r.id for r in records] == ["test/0", "test/1"]
    cache = (tmp_path / "cache" / _DATASET / _REVISION / "default" / "test").resolve()
    assert records[0].audio == cache / "000000.wav"
    assert records[1].audio.read_bytes() == b"RIFFy"
    assert records[0].language == "en"


def test_hub_is_asked_for_the_pinned_revision_as_a_stream(
    tmp_path: Path, hub: dict[str, Any]
) -> None:
    """Dataset, subset, split and revision are passed through, and rows are streamed."""
    hub["rows"] = {"audio": _wav(tmp_path, "x")}

    list(_source(tmp_path, subset="clean").records())

    assert hub["call"] == {
        "path": _DATASET,
        "name": "clean",
        "data_dir": None,
        "split": "test",
        "revision": _REVISION,
        "streaming": True,
    }


def test_audio_given_as_bytes_is_written_to_the_cache(tmp_path: Path, hub: dict[str, Any]) -> None:
    """Hub parquet datasets carry audio as bytes plus a file name; both are used."""
    hub["rows"] = {"audio": [{"bytes": b"fLaC-data", "path": "utt.FLAC"}]}

    (record,) = _source(tmp_path).records()

    assert record.audio.name == "000000.flac"
    assert record.audio.read_bytes() == b"fLaC-data"


def test_ids_and_transcripts_come_from_configured_columns(
    tmp_path: Path, hub: dict[str, Any]
) -> None:
    """`id_column` gives ids; `text_column` gives the reference or the pipeline output."""
    hub["rows"] = {"audio": _wav(tmp_path, "x"), "key": ["spk1-utt7"], "text": ["hello"]}
    gold = _source(tmp_path, id_column="key", text_column="text", transcripts="reference")

    (gold_record,) = gold.records()
    (output_record,) = _source(tmp_path, text_column="text", transcripts="output").records()

    assert gold_record.id == "spk1-utt7" and gold_record.reference == "hello"
    assert output_record.output is not None and output_record.output.text == "hello"


def test_limit_reads_only_the_first_rows(tmp_path: Path, hub: dict[str, Any]) -> None:
    """With `limit`, the stream is cut, so a big dataset is not read in full."""
    hub["rows"] = {"audio": _wav(tmp_path, "x", "y", "z")}

    assert [r.id for r in _source(tmp_path, limit=2).records()] == ["test/0", "test/1"]


def test_cached_audio_is_not_written_again(tmp_path: Path, hub: dict[str, Any]) -> None:
    """A file already in the cache is trusted, since the revision is pinned."""
    hub["rows"] = {"audio": _wav(tmp_path, "x")}
    (first,) = _source(tmp_path).records()
    first.audio.write_bytes(b"already cached")

    (second,) = _source(tmp_path).records()

    assert second.audio == first.audio
    assert second.audio.read_bytes() == b"already cached"


def test_missing_column_is_reported_per_row(tmp_path: Path, hub: dict[str, Any]) -> None:
    """A misnamed column fails on every row, naming the columns the dataset has."""
    hub["rows"] = {"audio": _wav(tmp_path, "x", "y")}
    source = _source(tmp_path, text_column="txt", transcripts="reference")

    with pytest.raises(SourceError) as error:
        list(source.records())

    issues = [(issue.where, issue.message) for issue in error.value.issues]
    assert issues == [
        (f"{_DATASET}[test]#0", "no column 'txt', columns are: audio"),
        (f"{_DATASET}[test]#1", "no column 'txt', columns are: audio"),
    ]


def test_duplicate_ids_are_reported(tmp_path: Path, hub: dict[str, Any]) -> None:
    """Two rows with one id would be indistinguishable in results."""
    hub["rows"] = {"audio": _wav(tmp_path, "x", "y"), "key": ["same", "same"]}

    with pytest.raises(SourceError) as error:
        list(_source(tmp_path, id_column="key").records())

    (issue,) = error.value.issues
    assert issue.message == "duplicate id 'same', first in row 0"


def test_transcript_role_needs_a_text_column(tmp_path: Path) -> None:
    """Asking for reference transcripts without saying where they are fails in the config."""
    with pytest.raises(ValidationError, match="needs text_column"):
        HFConfig(
            dataset=_DATASET, revision=_REVISION, split="test", license="x", transcripts="reference"
        )


def test_revision_is_required() -> None:
    """Without a pinned revision two runs could read different data."""
    with pytest.raises(ValidationError, match="revision"):
        HFConfig.model_validate({"dataset": _DATASET, "split": "test", "license": "x"})


def test_build_source_without_the_extra_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where `datasets` is missing, `type: hf` fails with the install command, not obscurely."""
    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.delitem(sys.modules, "paribus.engine.hf")

    with pytest.raises(ImportError, match=r"paribus\[hf\]"):
        build_source(
            {
                "type": "hf",
                "dataset": _DATASET,
                "revision": _REVISION,
                "split": "test",
                "license": "x",
            }
        )


def test_build_source_builds_an_hf_source() -> None:
    """With the extra installed, `type: hf` builds a `HFDatasetSource` named after the dataset."""
    source = build_source(
        {"type": "hf", "dataset": _DATASET, "revision": _REVISION, "split": "test", "license": "x"}
    )

    assert isinstance(source, HFDatasetSource)
    assert source.name == _DATASET


def test_meta_columns_are_copied_into_meta(tmp_path: Path, hub: dict[str, Any]) -> None:
    """Configured scalar columns become metadata that slice search can use."""
    hub["rows"] = {"audio": _wav(tmp_path, "x"), "sex": ["F"], "speaker": [1089]}

    (record,) = _source(tmp_path, meta_columns=["sex", "speaker"]).records()

    assert record.meta == {"sex": "F", "speaker": 1089}


@pytest.mark.parametrize(
    "words",
    [
        [{"word": "hi", "start": 0.5, "end": 0.9}, {"word": "there", "start": 1.0, "end": 1.4}],
        [[0.5, 0.9], [1.0, 1.4]],
    ],
)
def test_speech_segments_come_from_aligned_spans(
    tmp_path: Path, hub: dict[str, Any], words: list[Any]
) -> None:
    """Word alignments become ground-truth speech spans; extra keys like `word` are ignored."""
    hub["rows"] = {"audio": _wav(tmp_path, "x"), "words": [words]}

    (record,) = _source(tmp_path, speech_segments_column="words").records()

    assert record.speech_segments == [(0.5, 0.9), (1.0, 1.4)]


def test_speech_segments_of_the_wrong_shape_are_reported(
    tmp_path: Path, hub: dict[str, Any]
) -> None:
    """A column that is not spans fails on its row instead of being guessed at."""
    hub["rows"] = {"audio": _wav(tmp_path, "x"), "words": [["hi", "there"]]}

    with pytest.raises(SourceError) as error:
        list(_source(tmp_path, speech_segments_column="words").records())

    assert error.value.issues[0].message == "column 'words' must hold spans with start and end"


def test_a_data_dir_is_read_and_cached_on_its_own(tmp_path: Path, hub: dict[str, Any]) -> None:
    """A folder of a dataset is passed to the Hub and gets its own cache folder."""
    hub["rows"] = {"audio": _wav(tmp_path, "x")}

    (record,) = _source(tmp_path, data_dir="ru_ru").records()

    assert hub["call"]["data_dir"] == "ru_ru"
    cache = tmp_path / "cache" / _DATASET / _REVISION / "default" / "ru_ru" / "test"
    assert record.audio == cache.resolve() / "000000.wav"
