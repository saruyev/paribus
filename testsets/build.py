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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from paribus.engine.audio import load_audio
from paribus.engine.data import build_source
from paribus.engine.types import MetaValue, SampleRecord
from testsets import ops
from testsets.ops import Clip, Samples
from testsets.recipe import Background, Concat, Condition, Range, Recipe, SpeechInput, load_recipe

# Rate of clips made from nothing when the recipe keeps every file at its own rate.
_DEFAULT_RATE = 16000


@dataclass(frozen=True)
class Phrase:
    """A source phrase, decoded once and corrupted by every speech condition.

    Attributes:
        key: `<source name>/<record id>`, unique across the recipe's inputs; a long clip
            made by `concat` is `<first part's key>+<number of parts>`.
        record_id: Id of the (first) source record, used for grouping and ordering.
        language: Expected language, from the source record.
        reference: Gold transcript, if the source has one.
        clip: Mono audio at the recipe's sample rate, with ground-truth speech spans.
        truth: `alignment` if the spans come from the source, `utterance_bounds` if
            they are just the whole phrase.
        meta: The source record's metadata, plus what `concat` adds; it goes into the
            built record, so slice search can use the source's own columns.
        origin: The source file, when `clip` is that whole file unchanged (its own rate,
            all of it); a condition without operations then copies it byte for byte.
    """

    key: str
    record_id: str
    language: str | None
    reference: str | None
    clip: Clip
    truth: str
    meta: dict[str, MetaValue] = field(default_factory=dict)
    origin: Path | None = None


@dataclass(frozen=True)
class Pool:
    """Background clips of one recipe background, decoded once.

    Attributes:
        clips: (clip id, clip) pairs, at the recipe's rate or, if it is `native`, at
            each file's own rate.
        licenses: Number of clips in the pool under each license, for `sources.json`.
        used: How many built clips took a stretch of this pool, for `sources.json`.
    """

    clips: list[tuple[str, Clip]]
    licenses: Counter[str] = field(default_factory=Counter)
    used: Counter[str] = field(default_factory=Counter)
    _resampled: dict[tuple[str, int], Samples] = field(default_factory=dict, init=False, repr=False)

    def pick(self, rng: np.random.Generator, sample_rate: int) -> tuple[str, Samples]:
        """Choose one clip at random, count it as used, return it at `sample_rate`.

        A whole background clip is resampled once per rate and kept, so a stretch of
        it is cut from the same samples whichever clip it goes under.
        """
        clip_id, clip = self.clips[int(rng.integers(len(self.clips)))]
        self.used[clip_id] += 1
        if clip.sample_rate == sample_rate:
            return clip_id, clip.samples
        key = (clip_id, sample_rate)
        if key not in self._resampled:
            self._resampled[key] = ops.resample(clip, sample_rate).samples
        return clip_id, self._resampled[key]


@dataclass
class _Context:
    """The `OpContext` of one clip: its generator, its `meta`, and the build's fills.

    Every drawn setting is written down, so slice search can later tell which SNR or
    pad length a clip got.

    Attributes:
        rng: The clip's own generator.
        meta: What the clip's operations drew and noted.
        sample_rate: Rate of clips made from nothing (`OpContext.sample_rate`).
        pools: Decoded backgrounds by name, for `fill`.
    """

    rng: np.random.Generator
    meta: dict[str, MetaValue]
    sample_rate: int
    pools: Mapping[str, Pool]

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

    def fill(self, fill: str, frames: int, floor_dbfs: float, sample_rate: int) -> Samples:
        """Make `frames` of digital zero, a noise floor, or a random background stretch."""
        if fill == "zero":
            return ops.zeros(frames)
        if fill == "floor":
            return ops.noise_floor(frames, floor_dbfs, self.rng)
        background_id, samples = self.pools[fill].pick(self.rng, sample_rate)
        self.note("background_id", background_id)
        return ops.excerpt(samples, frames, self.rng) if frames else ops.zeros(0)


@dataclass(frozen=True)
class _Made:
    """A finished clip and what is known about it, ready to be written."""

    condition: Condition
    number: int
    phrase: Phrase | None
    clip: Clip
    meta: dict[str, MetaValue]
    copy: Path | None = None


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
        name: _load_pool(background, _fixed_rate(recipe))
        for name, background in recipe.backgrounds.items()
    }
    run = _Build(recipe, pools)
    records = [
        _write_clip(partial, made)
        for condition in recipe.conditions
        for made in run.clips(condition, phrases)
    ]
    _write_manifest(partial / "manifest.jsonl", records)
    _write_sources(partial / "sources.json", run)
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
        candidates = _candidates(speech, _fixed_rate(recipe))
        if speech.concat is not None:
            _check_one_rate(candidates, number)
            packer = _Packer(speech.concat, np.random.default_rng([recipe.seed, number, 1]))
            candidates = packer.pack(candidates)
        if len(candidates) < speech.take:
            kind = "long clips" if speech.concat else "phrases within its duration filter"
            raise ValueError(
                f"speech input {number} has {len(candidates)} {kind}, "
                f"the recipe takes {speech.take}"
            )
        chosen = sorted(rng.choice(len(candidates), size=speech.take, replace=False))
        phrases.extend(candidates[index] for index in chosen)
    keys = [phrase.key for phrase in phrases]
    if len(set(keys)) != len(keys):
        raise ValueError("two speech inputs share a source name; give them distinct `name`s")
    return phrases


def _check_one_rate(phrases: list[Phrase], number: int) -> None:
    """Require one sample rate across the phrases that `concat` joins.

    With a fixed recipe rate this always holds. With `native` the files must already
    agree, because joining 8 kHz and 16 kHz audio would need a silent resample.

    Raises:
        ValueError: If the phrases have more than one rate, listing them.
    """
    rates = sorted({phrase.clip.sample_rate for phrase in phrases})
    if len(rates) > 1:
        listed = ", ".join(str(rate) for rate in rates)
        raise ValueError(
            f"speech input {number} joins phrases with concat, but they have different "
            f"sample rates ({listed} Hz); set the recipe's sample_rate to one rate"
        )


def _fixed_rate(recipe: Recipe) -> int | None:
    """The rate every input is resampled to on load, or None to keep each file's own."""
    return None if recipe.sample_rate == "native" else recipe.sample_rate


def _candidates(speech: SpeechInput, sample_rate: int | None) -> list[Phrase]:
    """Decode an input's records and keep those within its duration filter.

    `sample_rate` None keeps every file at its own rate.
    """
    source = build_source(speech.source)
    candidates = []
    for record in source.records():
        spans = record.speech_segments
        audio = load_audio(record, sample_rate)
        clip = Clip(audio.mono(), audio.sample_rate, list(spans or []))
        # Unchanged means: its own rate, every channel of the file (mono stays mono).
        unchanged = sample_rate is None and record.channel is None and audio.samples.shape[0] == 1
        if speech.duration is not None and not (
            speech.duration[0] <= clip.seconds <= speech.duration[1]
        ):
            continue
        if spans is None:  # no ground truth from the source: the whole phrase is speech
            clip = replace(clip, speech=[(0.0, clip.seconds)])
        candidates.append(
            Phrase(
                key=f"{source.name}/{record.id}",
                record_id=record.id,
                language=record.language,
                reference=record.reference,
                clip=clip,
                truth="alignment" if spans is not None else "utterance_bounds",
                meta=dict(record.meta),
                origin=record.audio if unchanged else None,
            )
        )
    return candidates


@dataclass(frozen=True)
class _Packer:
    """Joins consecutive phrases of each group into long clips of the drawn lengths.

    Attributes:
        concat: The input's `concat` settings.
        rng: Generator for target lengths and gaps, separate from phrase selection, so
            adding `concat` to one input does not change what other inputs pick.
    """

    concat: Concat
    rng: np.random.Generator

    def pack(self, phrases: list[Phrase]) -> list[Phrase]:
        """Group phrases and pack each group, in group-key order, into long clips.

        Within a group phrases follow their record ids, so the parts of a clip follow
        each other as in the source (a book chapter, read aloud).
        """
        groups: dict[str, list[Phrase]] = {}
        for phrase in phrases:
            groups.setdefault(self._group_of(phrase), []).append(phrase)
        packed: list[Phrase] = []
        for group, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda member: member.record_id)
            packed.extend(self._pack_group(group, ordered))
        return packed

    def _pack_group(self, group: str, phrases: list[Phrase]) -> list[Phrase]:
        """Pack one group's phrases, in order, into clips between the `seconds` bounds.

        A clip is closed when it reaches its drawn target, or early when the next phrase
        would push it past the high bound. A clip shorter than the low bound is dropped.
        """
        low, high = ops.bounds(self.concat.seconds)
        packed: list[Phrase] = []
        current: list[tuple[float, Phrase]] = []  # (silence before, phrase)
        target = ops.draw(self.concat.seconds, self.rng)

        def close() -> None:
            """Keep the current clip if it is long enough, and start the next one."""
            nonlocal current, target
            if current and _length(current) >= low:
                packed.append(self._join(group, current))
            current, target = [], ops.draw(self.concat.seconds, self.rng)

        for phrase in phrases:
            gap = ops.draw(self.concat.gap, self.rng) if current else 0.0
            if current and _length(current) + gap + phrase.clip.seconds > high:
                close()
                gap = 0.0
            current.append((gap, phrase))
            if _length(current) >= target:
                close()
        close()
        return packed

    def _group_of(self, phrase: Phrase) -> str:
        """The group a phrase belongs to: its id prefix, or a value from its metadata.

        Raises:
            ValueError: If grouping by a `meta` key the phrase does not have.
        """
        if self.concat.group_by == "id_prefix":
            return phrase.record_id.rsplit(self.concat.separator, 1)[0]
        if self.concat.group_by not in phrase.meta:
            raise ValueError(f"phrase {phrase.key} has no meta {self.concat.group_by!r}")
        return str(phrase.meta[self.concat.group_by])

    def _join(self, group: str, parts: list[tuple[float, Phrase]]) -> Phrase:
        """Join phrases into one with `ops.join`, each after its silence."""
        phrases = [phrase for _, phrase in parts]
        first = phrases[0]
        clip = first.clip
        for gap, phrase in parts[1:]:
            clip = ops.join(clip, ops.zeros(int(gap * clip.sample_rate)), phrase.clip)
        references = [phrase.reference for phrase in phrases if phrase.reference]
        aligned = all(phrase.truth == "alignment" for phrase in phrases)
        return Phrase(
            key=f"{first.key}+{len(phrases)}",
            record_id=first.record_id,
            language=first.language,
            reference=" ".join(references) if references else None,
            clip=clip,
            truth="alignment" if aligned else "utterance_bounds",
            meta={**first.meta, "concat_parts": len(phrases), "concat_group": group},
        )


def _length(parts: list[tuple[float, Phrase]]) -> float:
    """Length in seconds of phrases joined with their silences."""
    return sum(gap + phrase.clip.seconds for gap, phrase in parts)


def _load_pool(background: Background, sample_rate: int | None) -> Pool:
    """Decode a background's clips, keeping only those whose license is allowed.

    `sample_rate` None keeps every clip at its file's rate; `Pool.pick` converts it.

    Raises:
        ValueError: If no clip is left after the license filter.
    """
    source = build_source(background.source)
    clips: list[tuple[str, Clip]] = []
    licenses: Counter[str] = Counter()
    for record in source.records():
        license_name = (
            str(record.meta.get(background.license_column, ""))
            if background.license_column
            else source.license
        )
        if background.licenses is not None and license_name not in background.licenses:
            continue
        audio = load_audio(record, sample_rate)
        clips.append((f"{source.name}/{record.id}", Clip(audio.mono(), audio.sample_rate, [])))
        licenses[license_name] += 1
    if not clips:
        raise ValueError(f"background source {source.name!r} has no clip with an allowed license")
    return Pool(clips=clips, licenses=licenses)


@dataclass(frozen=True)
class _Build:
    """What stays the same for a whole build: the recipe and its decoded backgrounds.

    Attributes:
        recipe: The recipe being built.
        pools: Decoded background clips by background name; they count their own use.
    """

    recipe: Recipe
    pools: Mapping[str, Pool]

    def clips(self, condition: Condition, phrases: list[Phrase]) -> list[_Made]:
        """Make every clip of one condition: one per phrase, or `count` without speech."""
        starts: list[tuple[Phrase | None, Clip | None]]
        if condition.speech == "none":
            assert condition.count is not None  # guaranteed by Condition's validator
            starts = [(None, None)] * condition.count
        else:
            starts = [(phrase, phrase.clip) for phrase in phrases]
        salt = zlib.crc32(condition.name.encode())
        made_rate = _fixed_rate(self.recipe) or _DEFAULT_RATE
        made = []
        for number, (phrase, clip) in enumerate(starts):
            rng = np.random.default_rng([self.recipe.seed, salt, number])
            ctx = _Context(rng, {"condition": condition.name}, made_rate, self.pools)
            for op in condition.ops:
                clip = op.make(ctx) if clip is None else op.apply(clip, ctx)
            assert clip is not None  # a no-speech condition starts with a clip-making op
            copy = phrase.origin if phrase is not None and not condition.ops else None
            made.append(_Made(condition, number, phrase, clip, ctx.meta, copy))
        return made


def _write_clip(root: Path, made: _Made) -> SampleRecord:
    """Write one clip as 16-bit wav and return its record, with the audio path relative."""
    phrase, meta = made.phrase, made.meta
    name = phrase.key if phrase is not None else f"{made.number:04d}"
    folder = Path("audio") / made.condition.name
    if made.copy is not None:  # the original file, untouched
        relative = folder / f"{name}{made.copy.suffix.lower()}"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(made.copy, root / relative)
    else:
        relative = folder / f"{name}.wav"
        _write_wav(root / relative, made.clip)
    if phrase is not None:
        meta = {**phrase.meta, **meta, "source_id": phrase.key, "speech_truth": phrase.truth}
    return SampleRecord(
        id=f"{made.condition.name}/{name}",
        audio=relative,
        language=phrase.language if phrase else None,
        reference=phrase.reference if phrase else None,
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


def _write_sources(path: Path, run: _Build) -> None:
    """Write every input with its full config (dataset, revision) and license."""
    recipe = run.recipe
    inputs: list[dict[str, Any]] = [
        {"role": "speech", "source": speech.source, "take": speech.take} for speech in recipe.speech
    ]
    for name, background in recipe.backgrounds.items():
        pool = run.pools[name]
        inputs.append(
            {
                "role": f"background:{name}",
                "source": background.source,
                "clips_in_pool_by_license": dict(sorted(pool.licenses.items())),
                "clips_used": pool.used.total(),
            }
        )
    document = {
        "recipe": recipe.name,
        "seed": recipe.seed,
        "plugins": recipe.plugins,
        "inputs": inputs,
    }
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
