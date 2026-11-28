"""Build a test set from a recipe: phrases x conditions -> wav files + a paribus manifest.

Usage: `python -m testsets.build RECIPE --out DIR [--force]`

The set is written to `DIR/<recipe name>/`: `audio/<condition>/...wav`, `manifest.jsonl`
(read it with a `manifest` source) and `sources.json` (every input with its revision and
license). It is built in a temporary folder next to the target and moved into place at
the end, so an interrupted build never leaves a half-written set behind.

Every clip has its own random generator seeded from (recipe seed, condition, clip
number), so a rebuild gives byte-identical files and adding a condition changes no
existing clip.
"""

import argparse
import json
import shutil
import wave
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from paribus.engine.audio import load_audio
from paribus.engine.data import build_source
from paribus.engine.types import MetaValue, SampleRecord
from testsets import ops
from testsets.ops import Clip, Samples
from testsets.recipe import (
    Background,
    Codec,
    Condition,
    Gain,
    Generate,
    Mix,
    Op,
    Pad,
    Range,
    Recipe,
    Resample,
    SpeechInput,
    load_recipe,
)


@dataclass(frozen=True)
class Phrase:
    """A source phrase, decoded once and corrupted by every speech condition.

    Attributes:
        key: `<source name>/<record id>`, unique across the recipe's inputs.
        record: The source record (language, reference, id).
        samples: Mono audio at the recipe's sample rate.
        speech: Ground-truth speech spans in seconds.
        truth: `alignment` if the spans come from the source, `utterance_bounds` if
            they are just the whole phrase.
    """

    key: str
    record: SampleRecord
    samples: Samples
    speech: list[ops.Span]
    truth: str


@dataclass(frozen=True)
class Pool:
    """Background clips of one recipe background, decoded at the recipe's sample rate."""

    clips: list[tuple[str, Samples]]
    licenses: Counter[str] = field(default_factory=Counter)


@dataclass
class _Draw:
    """One clip's random generator, and the record in `meta` of what it drew.

    Every drawn setting is written down, so slice search can later tell which SNR or
    pad length a clip got.
    """

    rng: np.random.Generator
    meta: dict[str, MetaValue]

    def value(self, key: str, setting: Range, digits: int) -> float:
        """Draw a setting (fixed or a range), record it rounded under `key`, return it."""
        drawn = ops.draw(setting, self.rng)
        self.note(key, round(drawn, digits))
        return drawn

    def note(self, key: str, value: MetaValue) -> None:
        """Set `meta[key]`, or `key_2`, `key_3`... when an operation repeats in a condition."""
        name, number = key, 1
        while name in self.meta:
            number += 1
            name = f"{key}_{number}"
        self.meta[name] = value


@dataclass(frozen=True)
class _Made:
    """A finished clip and what is known about it, ready to be written."""

    condition: Condition
    number: int
    phrase: Phrase | None
    clip: Clip
    meta: dict[str, MetaValue]


def build(recipe: Recipe, out_dir: Path, force: bool = False) -> Path:
    """Build the recipe's test set under `out_dir` and return its manifest path.

    Raises:
        FileExistsError: If the set already exists and `force` is not set.
        ValueError: If an input has fewer usable phrases than the recipe takes.
    """
    target = out_dir / recipe.name
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild it")
    partial = out_dir / f".{recipe.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    phrases = _load_phrases(recipe)
    pools = {
        name: _load_pool(background, recipe.sample_rate)
        for name, background in recipe.backgrounds.items()
    }
    run = _Build(recipe, pools, Counter())
    records = [
        _write_clip(partial, made)
        for condition in recipe.conditions
        for made in run.clips(condition, phrases)
    ]
    _write_manifest(partial / "manifest.jsonl", records)
    _write_sources(partial / "sources.json", recipe, pools, run.used)
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    return target / "manifest.jsonl"


def _load_phrases(recipe: Recipe) -> list[Phrase]:
    """Decode every input's phrases and take `take` of each, chosen by the seed.

    Raises:
        ValueError: If two inputs share a source name, or an input has too few phrases
            within its duration filter.
    """
    phrases: list[Phrase] = []
    for number, speech in enumerate(recipe.speech):
        rng = np.random.default_rng([recipe.seed, number])
        candidates = _candidates(speech, recipe.sample_rate)
        if len(candidates) < speech.take:
            raise ValueError(
                f"speech input {number} has {len(candidates)} phrases within its duration "
                f"filter, the recipe takes {speech.take}"
            )
        chosen = sorted(rng.choice(len(candidates), size=speech.take, replace=False))
        phrases.extend(candidates[index] for index in chosen)
    keys = [phrase.key for phrase in phrases]
    if len(set(keys)) != len(keys):
        raise ValueError("two speech inputs share a source name; give them distinct `name`s")
    return phrases


def _candidates(speech: SpeechInput, sample_rate: int) -> list[Phrase]:
    """Decode an input's records and keep those within its duration filter."""
    source = build_source(speech.source)
    candidates = []
    for record in source.records():
        samples = load_audio(record, sample_rate).mono()
        duration = samples.shape[0] / sample_rate
        if speech.duration is not None and not speech.duration[0] <= duration <= speech.duration[1]:
            continue
        spans = record.speech_segments
        candidates.append(
            Phrase(
                key=f"{source.name}/{record.id}",
                record=record,
                samples=samples,
                speech=list(spans) if spans is not None else [(0.0, duration)],
                truth="alignment" if spans is not None else "utterance_bounds",
            )
        )
    return candidates


def _load_pool(background: Background, sample_rate: int) -> Pool:
    """Decode a background's clips, keeping only those whose license is allowed.

    Raises:
        ValueError: If no clip is left after the license filter.
    """
    source = build_source(background.source)
    clips: list[tuple[str, Samples]] = []
    licenses: Counter[str] = Counter()
    for record in source.records():
        license_name = (
            str(record.meta.get(background.license_column, ""))
            if background.license_column
            else source.license
        )
        if background.licenses is not None and license_name not in background.licenses:
            continue
        clips.append((f"{source.name}/{record.id}", load_audio(record, sample_rate).mono()))
        licenses[license_name] += 1
    if not clips:
        raise ValueError(f"background source {source.name!r} has no clip with an allowed license")
    return Pool(clips=clips, licenses=licenses)


@dataclass(frozen=True)
class _Build:
    """What stays the same for a whole build: the recipe, the backgrounds, their use.

    Attributes:
        recipe: The recipe being built.
        pools: Decoded background clips by background name.
        used: How many clips took each background clip, by its id, for `sources.json`.
    """

    recipe: Recipe
    pools: Mapping[str, Pool]
    used: Counter[str]

    def clips(self, condition: Condition, phrases: list[Phrase]) -> list[_Made]:
        """Make every clip of one condition: one per phrase, or `count` without speech."""
        rate = self.recipe.sample_rate
        starts: list[tuple[Phrase | None, Clip | None]]
        if condition.speech == "none":
            assert condition.count is not None  # guaranteed by Condition's validator
            starts = [(None, None)] * condition.count
        else:
            starts = [(phrase, Clip(phrase.samples, rate, phrase.speech)) for phrase in phrases]
        salt = zlib.crc32(condition.name.encode())
        made = []
        for number, (phrase, clip) in enumerate(starts):
            draw = _Draw(np.random.default_rng([self.recipe.seed, salt, number]), {})
            draw.meta["condition"] = condition.name
            for op in condition.ops:
                clip = self.apply(op, clip, draw)
            assert clip is not None  # a no-speech condition starts with generate
            made.append(_Made(condition, number, phrase, clip, draw.meta))
        return made

    def apply(self, op: Op, clip: Clip | None, draw: _Draw) -> Clip:
        """Apply one operation, drawing its settings through `draw`."""
        if isinstance(op, Generate):
            rate = self.recipe.sample_rate
            frames = int(draw.value("seconds", op.seconds, 3) * rate)
            return ops.generate(self.fill(op.fill, frames, op.floor_dbfs, draw), rate)
        assert clip is not None  # only generate starts from nothing
        if isinstance(op, Mix):
            snr_db = draw.value("snr_db", op.snr_db, 2)
            return ops.mix(clip, self.fill(op.background, clip.frames, 0.0, draw), snr_db)
        if isinstance(op, Pad):
            rate = clip.sample_rate
            before = int(ops.draw(op.before, draw.rng) * rate)
            after = int(ops.draw(op.after, draw.rng) * rate)
            # Recorded from the frame counts, so the meta matches the audio exactly.
            draw.note("pad_before", round(before / rate, 3))
            draw.note("pad_after", round(after / rate, 3))
            filler = self.fill(op.fill, before + after, op.floor_dbfs, draw)
            return ops.pad(clip, filler[:before], filler[before:])
        if isinstance(op, Resample):
            draw.note("sample_rate_out", op.rate)
            return ops.resample(clip, op.rate)
        if isinstance(op, Codec):
            draw.note("codec", op.name if op.bitrate is None else f"{op.name}@{op.bitrate}")
            return ops.codec(clip, op.name, op.bitrate)
        if isinstance(op, Gain):
            return ops.gain(clip, draw.value("gain_db", op.db, 2))
        raise TypeError(f"unhandled operation {op!r}")

    def fill(self, fill: str, frames: int, floor_dbfs: float, draw: _Draw) -> Samples:
        """Make `frames` of digital zero, a noise floor, or a random background stretch."""
        if fill == "zero":
            return ops.zeros(frames)
        if fill == "floor":
            return ops.noise_floor(frames, floor_dbfs, draw.rng)
        clips = self.pools[fill].clips
        background_id, samples = clips[int(draw.rng.integers(len(clips)))]
        draw.note("background_id", background_id)
        self.used[background_id] += 1
        return ops.excerpt(samples, frames, draw.rng) if frames else ops.zeros(0)


def _write_clip(root: Path, made: _Made) -> SampleRecord:
    """Write one clip as 16-bit wav and return its record, with the audio path relative."""
    phrase, meta = made.phrase, made.meta
    name = phrase.key if phrase is not None else f"{made.number:04d}"
    relative = Path("audio") / made.condition.name / f"{name}.wav"
    _write_wav(root / relative, made.clip)
    if phrase is not None:
        meta = {**meta, "source_id": phrase.key, "speech_truth": phrase.truth}
    return SampleRecord(
        id=f"{made.condition.name}/{name}",
        audio=relative,
        language=phrase.record.language if phrase else None,
        reference=phrase.record.reference if phrase else None,
        speech_segments=[(round(start, 4), round(end, 4)) for start, end in made.clip.speech],
        meta=meta,
    )


def _write_wav(path: Path, clip: Clip) -> None:
    """Write a clip as mono 16-bit PCM, the format every model and tool reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(clip.samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(clip.sample_rate)
        out.writeframes(pcm.tobytes())


def _write_manifest(path: Path, records: list[SampleRecord]) -> None:
    """Write records as JSONL, one per line, with audio paths relative to the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            row = record.model_dump(mode="json", exclude_none=True)
            row["audio"] = record.audio.as_posix()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sources(
    path: Path, recipe: Recipe, pools: Mapping[str, Pool], used: Counter[str]
) -> None:
    """Write every input with its full config (dataset, revision) and license."""
    inputs: list[dict[str, Any]] = [
        {"role": "speech", "source": speech.source, "take": speech.take} for speech in recipe.speech
    ]
    for name, background in recipe.backgrounds.items():
        ids = {clip_id for clip_id, _ in pools[name].clips}
        taken = sum(count for clip_id, count in used.items() if clip_id in ids)
        inputs.append(
            {
                "role": f"background:{name}",
                "source": background.source,
                "clips_in_pool_by_license": dict(sorted(pools[name].licenses.items())),
                "clips_used": taken,
            }
        )
    document = {"recipe": recipe.name, "seed": recipe.seed, "inputs": inputs}
    path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    """Command line: build one recipe."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("recipe", type=Path, help="recipe YAML file")
    parser.add_argument("--out", type=Path, required=True, help="folder to build the set in")
    parser.add_argument("--force", action="store_true", help="rebuild an existing set")
    args = parser.parse_args()
    manifest = build(load_recipe(args.recipe), args.out, force=args.force)
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
