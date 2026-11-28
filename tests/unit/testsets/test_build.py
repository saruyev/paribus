"""Tests for testsets.build: a whole build on small folders of synthetic tones."""

import json
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from paribus.engine.data import ManifestConfig, ManifestSource
from testsets import Clip, OpContext, Operation, Range, register_op
from testsets.build import build
from testsets.recipe import Concat, Recipe

_RATE = 16000


def _write_tone(
    path: Path, seconds: float, hz: float, level: float = 0.1, rate: int = _RATE
) -> None:
    """Write a mono 16-bit wav tone; a stand-in for a phrase or a background clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(rate * seconds)) / rate
    pcm = np.round(level * np.sin(2 * np.pi * hz * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """Folders `speech/` (four phrases with transcripts) and `music/` (two clips)."""
    for number, seconds in enumerate([1.0, 1.5, 2.0, 6.0]):
        _write_tone(tmp_path / "speech" / f"p{number}.wav", seconds, 300 + 50 * number)
        (tmp_path / "speech" / f"p{number}.txt").write_text(f"phrase {number}", encoding="utf-8")
    _write_tone(tmp_path / "music" / "m0.wav", 3.0, 1000)
    _write_tone(tmp_path / "music" / "m1.wav", 0.5, 2000)
    return tmp_path


def _recipe(
    root: Path, *conditions: dict[str, Any], take: int = 3, sample_rate: int | str = _RATE
) -> Recipe:
    """A recipe over the fixture folders with the given conditions."""
    return Recipe.model_validate(
        {
            "name": "tiny",
            "seed": 7,
            "sample_rate": sample_rate,
            "speech": [
                {
                    "source": {
                        "type": "folder",
                        "root": str(root / "speech"),
                        "license": "CC0-1.0",
                        "transcripts": "reference",
                        "language": "en",
                    },
                    "take": take,
                    "duration": [0.5, 3.0],
                }
            ],
            "backgrounds": {
                "music": {
                    "source": {
                        "type": "folder",
                        "root": str(root / "music"),
                        "license": "CC-BY-3.0",
                    }
                }
            },
            "conditions": list(conditions) or [{"name": "clean"}],
        }
    )


def _files(root: Path) -> dict[str, bytes]:
    """Every file under a built set, by relative path."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


_PADDED = {"name": "padded", "ops": [{"pad": {"before": [1, 2], "after": 0.5, "fill": "floor"}}]}
_MUSIC = {"name": "music", "ops": [{"mix": {"background": "music", "snr_db": [0, 10]}}]}
_ONLY_MUSIC = {
    "name": "only_music",
    "speech": "none",
    "count": 2,
    "ops": [{"generate": {"fill": "music", "seconds": 1}}],
}


def test_built_set_reads_back_as_a_manifest(inputs: Path, tmp_path: Path) -> None:
    """The output is an ordinary manifest: paribus reads it like user data."""
    manifest = build(_recipe(inputs, {"name": "clean"}, _MUSIC, _ONLY_MUSIC), tmp_path / "out")

    records = list(ManifestSource(ManifestConfig(path=manifest, license="mixed")).records())

    assert len(records) == 3 + 3 + 2
    clean = [r for r in records if r.meta["condition"] == "clean"]
    assert {r.reference for r in clean} <= {"phrase 0", "phrase 1", "phrase 2"}
    assert all(r.meta["speech_truth"] == "utterance_bounds" for r in clean)
    assert all(r.language == "en" for r in clean)
    music = [r for r in records if r.meta["condition"] == "music"]
    assert all(0 <= float(r.meta["snr_db"]) <= 10 for r in music)
    assert all(str(r.meta["background_id"]).startswith("music/") for r in music)
    no_speech = [r for r in records if r.meta["condition"] == "only_music"]
    assert all(r.speech_segments == [] and r.reference is None for r in no_speech)


def test_duration_filter_leaves_out_long_phrases(inputs: Path, tmp_path: Path) -> None:
    """The 6-second phrase is outside [0.5, 3] and is never chosen."""
    manifest = build(_recipe(inputs), tmp_path / "out")

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]

    assert all("p3" not in row["id"] for row in rows)


def test_too_few_phrases_is_an_error(inputs: Path, tmp_path: Path) -> None:
    """Asking for more phrases than pass the filter fails instead of building a smaller set."""
    with pytest.raises(ValueError, match=r"has 3 phrases .* takes 4"):
        build(_recipe(inputs, take=4), tmp_path / "out")


def test_padding_moves_the_speech_spans(inputs: Path, tmp_path: Path) -> None:
    """After padding, speech starts at `pad_before` and the clip is longer by both pads."""
    manifest = build(_recipe(inputs, _PADDED), tmp_path / "out")

    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        ((start, end),) = row["speech_segments"]
        assert start == pytest.approx(row["meta"]["pad_before"], abs=1e-3)
        with wave.open(str(manifest.parent / row["audio"])) as clip:
            seconds = clip.getnframes() / clip.getframerate()
        assert seconds == pytest.approx(end + row["meta"]["pad_after"], abs=1e-3)


@register_op("test_mute_start")
class _MuteStart(Operation):
    """Silence the first `seconds` of a clip: a stand-in for a user's own operation."""

    seconds: Range

    shorthand: ClassVar[str | None] = "seconds"

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Zero the drawn number of leading frames, recording the length in `meta`."""
        frames = int(ctx.value("muted", self.seconds, 3) * clip.sample_rate)
        samples = clip.samples.copy()
        samples[:, :frames] = 0
        return replace(clip, samples=samples)


def test_an_operation_from_outside_runs_like_a_built_in(inputs: Path, tmp_path: Path) -> None:
    """A registered operation is used from YAML, draws through the context, lands in meta."""
    condition = {"name": "muted", "ops": [{"test_mute_start": [0.1, 0.3]}]}

    rows = _rows(build(_recipe(inputs, condition), tmp_path / "out"))

    for row in rows:
        muted = row["meta"]["muted"]
        assert 0.1 <= muted <= 0.3
        with wave.open(str(tmp_path / "out" / "tiny" / row["audio"])) as clip:
            pcm = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
        edge = int(muted * _RATE)  # `meta` is rounded to 1 ms, so allow 16 frames
        assert not pcm[: edge - 16].any()
        assert pcm[edge + 16 :].any()


def test_rebuild_is_byte_identical(inputs: Path, tmp_path: Path) -> None:
    """The same recipe gives the same files, down to the noise and the drawn settings."""
    recipe = _recipe(inputs, _PADDED, _MUSIC, _ONLY_MUSIC)

    first = build(recipe, tmp_path / "a").parent
    second = build(recipe, tmp_path / "b").parent

    assert _files(first) == _files(second)


def test_adding_a_condition_changes_no_existing_clip(inputs: Path, tmp_path: Path) -> None:
    """Each clip's randomness depends on its own condition, not on the conditions before it."""
    before = build(_recipe(inputs, _MUSIC), tmp_path / "a").parent
    after = build(_recipe(inputs, _PADDED, _MUSIC), tmp_path / "b").parent

    old = {k: v for k, v in _files(before).items() if k.startswith("audio/music/")}
    new = {k: v for k, v in _files(after).items() if k.startswith("audio/music/")}
    assert old == new


def test_existing_set_needs_force(inputs: Path, tmp_path: Path) -> None:
    """A built set is not overwritten by accident; with force it is rebuilt."""
    recipe = _recipe(inputs)
    build(recipe, tmp_path / "out")

    with pytest.raises(FileExistsError, match="--force"):
        build(recipe, tmp_path / "out")
    assert build(recipe, tmp_path / "out", force=True).is_file()
    assert not (tmp_path / "out" / ".tiny.partial").exists()


def test_sources_json_lists_inputs_with_licenses(inputs: Path, tmp_path: Path) -> None:
    """Every input is recorded with its config, and backgrounds with their license counts."""
    manifest = build(_recipe(inputs, _MUSIC), tmp_path / "out")

    sources = json.loads((manifest.parent / "sources.json").read_text())

    speech, music = sources["inputs"]
    assert speech["role"] == "speech" and speech["source"]["license"] == "CC0-1.0"
    assert music["role"] == "background:music"
    assert music["clips_in_pool_by_license"] == {"CC-BY-3.0": 2}
    assert music["clips_used"] == 3


@pytest.fixture
def chapters(tmp_path: Path) -> Path:
    """`chapters/`: chapter 1-1 with five 2-second phrases, chapter 1-2 with one."""
    for name in ["1-1-0003", "1-1-0001", "1-1-0002", "1-1-0004", "1-1-0005", "1-2-0001"]:
        _write_tone(tmp_path / "chapters" / f"{name}.wav", 2.0, 400)
        (tmp_path / "chapters" / f"{name}.txt").write_text(name, encoding="utf-8")
    return tmp_path / "chapters"


def _long_recipe(root: Path, take: int, **concat: Any) -> Recipe:
    """A recipe that joins the chapter phrases into long clips."""
    return Recipe.model_validate(
        {
            "name": "long",
            "seed": 3,
            "speech": [
                {
                    "source": {
                        "type": "folder",
                        "root": str(root),
                        "license": "CC0-1.0",
                        "transcripts": "reference",
                    },
                    "take": take,
                    "concat": {"seconds": [4, 5], "gap": 0.5, **concat},
                }
            ],
            "conditions": [{"name": "clean"}],
        }
    )


def _rows(manifest: Path) -> list[dict[str, Any]]:
    """The manifest's rows as plain dicts."""
    return [json.loads(line) for line in manifest.read_text().splitlines()]


def test_concat_joins_consecutive_phrases_of_a_chapter(chapters: Path, tmp_path: Path) -> None:
    """Parts follow id order within a chapter; chapters are not mixed; leftovers are dropped.

    Two 2-second phrases and a 0.5 s gap make 4.5 s, inside [4, 5]; a third would pass 5.
    Chapter 1-1 gives two such clips and a dropped leftover; chapter 1-2 is too short.
    """
    manifest = build(_long_recipe(chapters, take=2), tmp_path / "out")

    rows = _rows(manifest)

    assert [row["reference"] for row in rows] == ["1-1-0001 1-1-0002", "1-1-0003 1-1-0004"]
    assert all(row["meta"]["concat_parts"] == 2 for row in rows)
    assert all(row["meta"]["concat_group"] == "1-1" for row in rows)
    assert rows[0]["id"] == "clean/chapters/1-1-0001+2"


def test_concat_shifts_speech_by_parts_and_gaps(chapters: Path, tmp_path: Path) -> None:
    """The second part's speech starts after the first part and the silence between them."""
    manifest = build(_long_recipe(chapters, take=2), tmp_path / "out")

    row = _rows(manifest)[0]

    assert row["speech_segments"] == [[0.0, 2.0], [2.5, 4.5]]
    with wave.open(str(manifest.parent / row["audio"])) as clip:
        assert clip.getnframes() == int(4.5 * _RATE)


def test_concat_with_too_few_long_clips_is_an_error(chapters: Path, tmp_path: Path) -> None:
    """Only two long clips can be made; asking for three fails."""
    with pytest.raises(ValueError, match="has 2 long clips, the recipe takes 3"):
        build(_long_recipe(chapters, take=3), tmp_path / "out")


def test_concat_groups_by_a_meta_key(tmp_path: Path) -> None:
    """With `group_by: speaker`, phrases join by the speaker in their metadata."""
    speakers = {"a": "s1", "b": "s2", "c": "s1", "d": "s2"}
    for name in speakers:
        _write_tone(tmp_path / f"{name}.wav", 2.0, 400)
    lines = [
        json.dumps({"id": n, "audio": f"{n}.wav", "reference": n, "meta": {"speaker": s}})
        for n, s in speakers.items()
    ]
    (tmp_path / "m.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    recipe = Recipe.model_validate(
        {
            "name": "long",
            "seed": 3,
            "speech": [
                {
                    "source": {
                        "type": "manifest",
                        "path": str(tmp_path / "m.jsonl"),
                        "license": "x",
                    },
                    "take": 2,
                    "concat": {"seconds": [4, 5], "gap": 0.5, "group_by": "speaker"},
                }
            ],
            "conditions": [{"name": "clean"}],
        }
    )

    rows = _rows(build(recipe, tmp_path / "out"))

    assert sorted(row["reference"] for row in rows) == ["a c", "b d"]
    assert all(row["meta"]["speaker"] == row["meta"]["concat_group"] for row in rows)


def _rate_of(path: Path) -> int:
    """Sample rate of a wav file."""
    with wave.open(str(path)) as clip:
        return clip.getframerate()


def test_native_clean_copies_the_original_files(inputs: Path, tmp_path: Path) -> None:
    """With `native`, a condition without operations is the source file, byte for byte."""
    manifest = build(_recipe(inputs, sample_rate="native"), tmp_path / "out")

    for row in _rows(manifest):
        source = inputs / "speech" / Path(row["meta"]["source_id"]).with_suffix(".wav").name
        assert (manifest.parent / row["audio"]).read_bytes() == source.read_bytes()


def test_native_keeps_each_rate_until_resample(tmp_path: Path) -> None:
    """An 8 kHz phrase stays 8 kHz under a 16 kHz background, and `resample` changes it."""
    _write_tone(tmp_path / "speech" / "p0.wav", 1.0, 300, rate=8000)
    _write_tone(tmp_path / "music" / "m0.wav", 3.0, 1000)
    recipe = _recipe(
        tmp_path,
        {"name": "music", "ops": [{"mix": {"background": "music", "snr_db": 10}}]},
        {"name": "wide", "ops": [{"resample": 16000}]},
        take=1,
        sample_rate="native",
    )

    rows = {row["meta"]["condition"]: row for row in _rows(build(recipe, tmp_path / "out"))}

    out = tmp_path / "out" / "tiny"
    assert _rate_of(out / rows["music"]["audio"]) == 8000
    assert _rate_of(out / rows["wide"]["audio"]) == 16000


def test_generate_has_its_own_rate(inputs: Path, tmp_path: Path) -> None:
    """Clips made from nothing are 16 kHz under `native`, unless `generate` says otherwise."""
    recipe = _recipe(
        inputs,
        {
            "name": "z16",
            "speech": "none",
            "count": 1,
            "ops": [{"generate": {"fill": "zero", "seconds": 1}}],
        },
        {
            "name": "z8",
            "speech": "none",
            "count": 1,
            "ops": [{"generate": {"fill": "music", "seconds": 1, "sample_rate": 8000}}],
        },
        sample_rate="native",
    )

    rows = {row["meta"]["condition"]: row for row in _rows(build(recipe, tmp_path / "out"))}

    out = tmp_path / "out" / "tiny"
    assert _rate_of(out / rows["z16"]["audio"]) == 16000
    assert _rate_of(out / rows["z8"]["audio"]) == 8000


def test_concat_needs_one_sample_rate(tmp_path: Path) -> None:
    """Under `native`, phrases of different rates cannot be joined; a fixed rate joins them."""
    _write_tone(tmp_path / "speech" / "b-1-0.wav", 1.0, 300, rate=8000)
    _write_tone(tmp_path / "speech" / "b-1-1.wav", 1.0, 400)
    _write_tone(tmp_path / "music" / "m0.wav", 1.0, 1000)
    recipe = _recipe(tmp_path, take=1, sample_rate="native")
    speech = recipe.speech[0].model_copy(
        update={"duration": None, "concat": Concat(seconds=(1.5, 3), gap=0.5)}
    )
    recipe = recipe.model_copy(update={"speech": [speech]})

    with pytest.raises(ValueError, match=r"different sample rates \(8000, 16000 Hz\)"):
        build(recipe, tmp_path / "out")

    fixed = recipe.model_copy(update={"sample_rate": 16000})
    (row,) = _rows(build(fixed, tmp_path / "out"))
    assert row["meta"]["concat_parts"] == 2


def _write_stereo(path: Path, seconds: float) -> None:
    """Write a stereo 16-bit wav: 300 Hz left, 600 Hz right, like two sides of a call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(_RATE * seconds)) / _RATE
    left, right = (0.1 * np.sin(2 * np.pi * hz * t) for hz in (300, 600))
    pcm = np.round(np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(_RATE)
        out.writeframes(pcm.tobytes())


def _channels_of(path: Path) -> int:
    """Channel count of a wav file."""
    with wave.open(str(path)) as clip:
        return clip.getnchannels()


def test_stereo_stays_stereo_unless_the_recipe_says_mono(tmp_path: Path) -> None:
    """`keep` copies a stereo clean file as is and keeps both channels under a mix."""
    _write_stereo(tmp_path / "speech" / "call.wav", 2.0)
    _write_tone(tmp_path / "music" / "m0.wav", 3.0, 1000)
    conditions = [
        {"name": "clean"},
        {"name": "music", "ops": [{"mix": {"background": "music", "snr_db": 10}}]},
    ]
    keep = _recipe(tmp_path, *conditions, take=1, sample_rate="native")
    mono = keep.model_copy(update={"channels": "mono", "name": "mono"})

    rows = {row["meta"]["condition"]: row for row in _rows(build(keep, tmp_path / "out"))}
    out = tmp_path / "out" / "tiny"
    assert (out / rows["clean"]["audio"]).read_bytes() == (
        tmp_path / "speech" / "call.wav"
    ).read_bytes()
    assert _channels_of(out / rows["music"]["audio"]) == 2

    for row in _rows(build(mono, tmp_path / "out")):
        assert _channels_of(tmp_path / "out" / "mono" / row["audio"]) == 1


def test_concat_needs_one_channel_count(tmp_path: Path) -> None:
    """Under `keep`, a mono and a stereo phrase cannot be joined; `mono` joins them."""
    _write_tone(tmp_path / "speech" / "b-1-0.wav", 1.0, 300)
    _write_stereo(tmp_path / "speech" / "b-1-1.wav", 1.0)
    _write_tone(tmp_path / "music" / "m0.wav", 1.0, 1000)
    recipe = _recipe(tmp_path, take=1)
    speech = recipe.speech[0].model_copy(
        update={"duration": None, "concat": Concat(seconds=(1.5, 3), gap=0.5)}
    )
    recipe = recipe.model_copy(update={"speech": [speech]})

    with pytest.raises(ValueError, match=r"different channel counts \(1, 2\)"):
        build(recipe, tmp_path / "out")

    (row,) = _rows(build(recipe.model_copy(update={"channels": "mono"}), tmp_path / "out"))
    assert row["meta"]["concat_parts"] == 2


def test_split_writes_pieces_of_a_group(inputs: Path, tmp_path: Path) -> None:
    """Each piece is a record with `group_id`, `offset`, local speech and its own meta."""
    condition = {"name": "cut", "ops": [{"split": 0.5}, {"gain": [-6, 0]}]}

    rows = _rows(build(_recipe(inputs, condition, take=1), tmp_path / "out"))

    assert len(rows) > 1
    assert {row["group_id"] for row in rows} == {"cut/" + rows[0]["meta"]["source_id"]}
    assert [row["offset"] for row in rows] == [0.5 * part for part in range(len(rows))]
    assert [row["meta"]["split_part"] for row in rows] == list(range(len(rows)))
    assert all("gain_db_2" not in row["meta"] for row in rows)  # each piece has its own meta
    assert all(end <= 0.5 for row in rows for _, end in row["speech_segments"])


def test_resample_before_or_after_split_gives_the_same_pieces(inputs: Path, tmp_path: Path) -> None:
    """The two orders differ in the samples only, so they can be compared piece by piece."""
    whole = {"name": "whole", "ops": [{"resample": 8000}, {"split": 0.5}]}
    pieces = {"name": "pieces", "ops": [{"split": 0.5}, {"resample": 8000}]}

    rows = _rows(build(_recipe(inputs, whole, pieces), tmp_path / "out"))

    by_order: dict[str, list[tuple[float, list[Any]]]] = {"whole": [], "pieces": []}
    for row in rows:
        by_order[row["meta"]["condition"]].append((row["offset"], row["speech_segments"]))
    assert by_order["whole"] == by_order["pieces"]
    out = tmp_path / "out" / "tiny"
    assert all(_rate_of(out / row["audio"]) == 8000 for row in rows)
