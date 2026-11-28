"""The YAML recipe of a test set: which speech, which backgrounds, which corruptions.

A recipe is data, not code: adding a condition or a dataset is a YAML change. Every
operation is written as a one-key mapping, `{mix: {background: music, snr_db: 0}}`, and an
operation with one main setting accepts it bare: `{resample: 8000}`, `{codec: opus}`.

Settings that vary from clip to clip are a `Range`: a number for a fixed value, or
`[low, high]` for a value drawn uniformly per clip from the recipe's seed.
"""

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self

import numpy as np
import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from paribus.engine.data import build_source
from paribus.engine.types import MetaValue
from testsets import ops
from testsets.ops import Clip, Samples


def _ordered(value: float | tuple[float, float]) -> float | tuple[float, float]:
    """Reject a `[low, high]` range whose low end is above its high end."""
    if isinstance(value, tuple) and value[0] > value[1]:
        raise ValueError(f"range low end {value[0]} is above high end {value[1]}")
    return value


type Range = Annotated[float | tuple[float, float], AfterValidator(_ordered)]

# A dotted Python module name, as `plugins` lists them.
_MODULE = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
# Fill names that `pad` and `generate` understand without a background of that name.
_BUILT_IN_FILLS = ("zero", "floor")


class _Model(BaseModel):
    """Base of every recipe model: immutable, and unknown keys are an error.

    A typo in a recipe (`snr: 0` for `snr_db: 0`) must fail, not build a different
    test set than the one written down.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def _check_source(value: dict[str, Any]) -> dict[str, Any]:
    """Validate a source config as `build_source` would, without reading any data."""
    build_source(value)
    return value


type SourceSettings = Annotated[dict[str, Any], AfterValidator(_check_source)]


class Concat(_Model):
    """Join consecutive phrases of one group (a book chapter) into long clips.

    Long recordings behave differently from short phrases (models cut them into
    windows, and hallucinate at the seams), so they get their own clips. Phrases are
    grouped, sorted by id within a group, and joined in that order until a clip reaches
    its drawn length.

    Attributes:
        seconds: Target length of each long clip, drawn per clip; a clip never gets
            longer than the high end unless a single phrase is, and a leftover shorter
            than the low end is dropped.
        gap: Digital silence between joined phrases, drawn per gap. Noise over the whole
            clip is a condition's job, so the clean long clip stays clean.
        group_by: `id_prefix` groups by the record id up to its last `separator`
            (`1089-134686-0000` -> `1089-134686`); any other value is a `meta` key.
        separator: Separator for `id_prefix`.
    """

    seconds: Range
    gap: Range = 0.5
    group_by: str = "id_prefix"
    separator: str = Field(default="-", min_length=1)


class SpeechInput(_Model):
    """Where source phrases come from and how many to take.

    Attributes:
        source: A source config, exactly as `paribus.engine.data.build_source` takes it.
        take: Number of phrases to keep from this source (long clips, with `concat`).
        duration: Keep only phrases whose length in seconds is within `[low, high]`;
            with `concat`, this filters the phrases before they are joined.
        concat: Join phrases into long clips before any condition is applied.
    """

    source: SourceSettings
    take: int = Field(ge=1)
    duration: tuple[float, float] | None = None
    concat: Concat | None = None

    @field_validator("duration")
    @classmethod
    def _duration(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Check that the duration filter is a valid, non-negative range."""
        if value is not None and not 0 <= value[0] <= value[1]:
            raise ValueError(f"duration must be [low, high] with 0 <= low <= high, got {value}")
        return value


class Background(_Model):
    """A pool of background audio (music, noise) mixed under speech or played alone.

    Attributes:
        source: A source config for the background clips.
        license_column: A `meta` key holding each clip's own license, for pools that mix
            licenses (MUSAN does). Needed when `licenses` is set.
        licenses: Allowed values of `license_column`; clips with any other license are
            left out of the pool.
    """

    source: SourceSettings
    license_column: str | None = None
    licenses: list[str] | None = None

    @model_validator(mode="after")
    def _licenses_need_a_column(self) -> Self:
        """Reject an allow-list without the column it applies to, and the reverse."""
        if (self.license_column is None) != (self.licenses is None):
            raise ValueError("license_column and licenses go together")
        return self


class OpContext(Protocol):
    """What an operation gets from the build for the one clip it is working on.

    Everything random goes through it, so a build stays reproducible from the seed, and
    every drawn setting lands in the clip's `meta`, so results can be sliced by it.
    """

    @property
    def rng(self) -> np.random.Generator:
        """The clip's random generator; draw from it, never from a global one."""
        ...

    @property
    def sample_rate(self) -> int:
        """Rate for a clip made from nothing: the recipe's, or 16000 if it is `native`."""
        ...

    def value(self, key: str, setting: Range, digits: int) -> float:
        """Draw a setting (fixed or a range), record it in `meta` under `key`, return it."""
        ...

    def note(self, key: str, value: MetaValue) -> None:
        """Record a value in `meta`; a repeated key gets `_2`, `_3`..."""
        ...

    def fill(self, fill: str, frames: int, floor_dbfs: float, sample_rate: int) -> Samples:
        """`frames` at `sample_rate` of `zero`, a noise `floor`, or a background's stretch."""
        ...


class Operation(_Model):
    """Base of every operation, built in or your own: its settings plus what it does.

    A subclass declares its settings as fields (validated when the recipe loads) and
    overrides `apply`. It becomes usable in recipes once registered with `register_op`.

    Attributes:
        shorthand: The setting a bare value stands for: `{gain: -20}` is `{gain: {db: -20}}`.
        starts_clip: True for an operation that makes a clip without speech from nothing
            (overrides `make`); it must be the first op of a `speech: none` condition.
    """

    shorthand: ClassVar[str | None] = None
    starts_clip: ClassVar[bool] = False

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Return the corrupted clip; keep `clip.speech` true to where speech now is."""
        raise NotImplementedError(f"{type(self).__name__} does not change a clip")

    def make(self, ctx: OpContext) -> Clip:
        """Return a new clip without speech (only for `starts_clip` operations)."""
        raise NotImplementedError(f"{type(self).__name__} does not make a clip")

    def backgrounds(self) -> list[str]:
        """Names of the recipe backgrounds this operation draws from, checked at load."""
        return []


_OPS: dict[str, type[Operation]] = {}


def register_op[O: type[Operation]](name: str) -> Callable[[O], O]:
    """Class decorator that makes an `Operation` usable in recipes as `name`.

    Raises:
        ValueError: If `name` is already taken, so two plugins cannot silently swap
            one operation for another.
    """

    def add(cls: O) -> O:
        """Register `cls` under `name` and return it unchanged."""
        if name in _OPS:
            raise ValueError(f"operation {name!r} is already registered by {_OPS[name]}")
        _OPS[name] = cls
        return cls

    return add


def _fill_backgrounds(fill: str) -> list[str]:
    """The background a fill names, or nothing for a built-in fill."""
    return [] if fill in _BUILT_IN_FILLS else [fill]


@register_op("mix")
class Mix(Operation):
    """Mix a background under the speech at `snr_db`, measured against the speech's RMS."""

    background: str
    snr_db: Range

    shorthand: ClassVar[str | None] = "background"

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Draw the SNR, take a background stretch as long as the clip, mix it in."""
        snr_db = ctx.value("snr_db", self.snr_db, 2)
        background = ctx.fill(self.background, clip.frames, 0.0, clip.sample_rate)
        return ops.mix(clip, background, snr_db)

    def backgrounds(self) -> list[str]:
        """The background mixed in."""
        return [self.background]


@register_op("pad")
class Pad(Operation):
    """Add `before` and `after` seconds of `fill` around the speech.

    `fill` is `zero` (digital silence), `floor` (white noise at `floor_dbfs`) or the name
    of a background.
    """

    before: Range = 0.0
    after: Range = 0.0
    fill: str = "zero"
    floor_dbfs: float = -60.0

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Draw both lengths in whole frames, record them, and pad with one fill."""
        rate = clip.sample_rate
        before = int(ops.draw(self.before, ctx.rng) * rate)
        after = int(ops.draw(self.after, ctx.rng) * rate)
        # Recorded from the frame counts, so the meta matches the audio exactly.
        ctx.note("pad_before", round(before / rate, 3))
        ctx.note("pad_after", round(after / rate, 3))
        filler = ctx.fill(self.fill, before + after, self.floor_dbfs, rate)
        return ops.pad(clip, filler[:before], filler[before:])

    def backgrounds(self) -> list[str]:
        """The fill's background, if it is one."""
        return _fill_backgrounds(self.fill)


@register_op("generate")
class Generate(Operation):
    """Make a clip without speech: `seconds` of `fill` (as in `Pad`).

    `sample_rate` is the new clip's rate; without it, the recipe's rate, or 16000 when
    the recipe keeps each file's own rate.
    """

    fill: str
    seconds: Range
    floor_dbfs: float = -60.0
    sample_rate: int | None = Field(default=None, gt=0)

    shorthand: ClassVar[str | None] = "fill"
    starts_clip: ClassVar[bool] = True

    def make(self, ctx: OpContext) -> Clip:
        """Draw the length and fill it, at the recipe's sample rate."""
        rate = self.sample_rate or ctx.sample_rate
        frames = int(ctx.value("seconds", self.seconds, 3) * rate)
        return ops.generate(ctx.fill(self.fill, frames, self.floor_dbfs, rate), rate)

    def backgrounds(self) -> list[str]:
        """The fill's background, if it is one."""
        return _fill_backgrounds(self.fill)


@register_op("resample")
class Resample(Operation):
    """Change the sample rate, e.g. to 8000 to imitate narrowband telephony."""

    rate: int = Field(gt=0)

    shorthand: ClassVar[str | None] = "rate"

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Resample, and record the rate the clip ends up at."""
        ctx.note("sample_rate_out", self.rate)
        return ops.resample(clip, self.rate)


@register_op("codec")
class Codec(Operation):
    """Encode with a lossy codec and decode back, keeping the codec's artifacts.

    `bitrate` (bits per second) applies to `opus` only.
    """

    name: Literal["pcm_mulaw", "pcm_alaw", "opus"]
    bitrate: int | None = Field(default=None, gt=0)

    shorthand: ClassVar[str | None] = "name"

    @model_validator(mode="after")
    def _bitrate_only_for_opus(self) -> Self:
        """Reject a bitrate for the PCM codecs, whose bitrate is fixed by the rate."""
        if self.bitrate is not None and self.name != "opus":
            raise ValueError(f"bitrate applies to opus only, not {self.name}")
        return self

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Round-trip through the codec, and record which one (with its bitrate)."""
        ctx.note("codec", self.name if self.bitrate is None else f"{self.name}@{self.bitrate}")
        return ops.codec(clip, self.name, self.bitrate)


@register_op("gain")
class Gain(Operation):
    """Change the level by `db` decibels (negative is quieter)."""

    db: Range

    shorthand: ClassVar[str | None] = "db"

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Draw the gain, record it, apply it."""
        return ops.gain(clip, ctx.value("gain_db", self.db, 2))


def _unwrap_op(value: Any) -> Operation:
    """Turn the YAML form `{mix: {...}}` or `{resample: 8000}` into a registered operation.

    Raises:
        ValueError: If the form is wrong, the operation is unknown, or its settings are
            invalid (the settings' own error is included).
    """
    if isinstance(value, Operation):
        return value
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"an operation is a mapping with one key, one of: {', '.join(_OPS)}")
    ((name, settings),) = value.items()
    if name not in _OPS:
        raise ValueError(f"unknown operation {name!r}, known: {', '.join(_OPS)}")
    cls = _OPS[name]
    if not isinstance(settings, dict):
        if cls.shorthand is None:
            raise ValueError(f"operation {name!r} needs a mapping of settings")
        settings = {cls.shorthand: settings}
    return cls.model_validate(settings)


class Condition(_Model):
    """One way of corrupting the phrases; every condition gets the same phrases.

    Attributes:
        name: Becomes `meta.condition` and the audio subfolder; letters, digits, `_`, `-`.
        speech: `all` corrupts every source phrase; `none` makes clips without speech,
            which must then start with `generate`.
        count: Number of clips for a `none` condition.
        ops: Operations applied in order.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    speech: Literal["all", "none"] = "all"
    count: int | None = Field(default=None, ge=1)
    ops: list[Operation] = Field(default_factory=list)

    @field_validator("ops", mode="before")
    @classmethod
    def _ops(cls, value: Any) -> Any:
        """Accept operations in their one-key YAML form."""
        return [_unwrap_op(item) for item in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def _generate_iff_no_speech(self) -> Self:
        """A clip is speech or made from nothing: a clip-making op comes first, and only then."""
        starts = [op.starts_clip for op in self.ops]
        if self.speech == "none":
            if not starts or not starts[0] or self.count is None:
                raise ValueError(
                    "speech: none needs count and must start with generate "
                    "(or another operation that makes a clip)"
                )
            if any(starts[1:]):
                raise ValueError("an operation that makes a clip can only be the first one")
        elif any(starts) or self.count is not None:
            raise ValueError("generate and count are only for speech: none")
        return self


class Recipe(_Model):
    """A whole test set: inputs, backgrounds and conditions, reproducible from `seed`.

    Attributes:
        name: Name of the test set; becomes the output folder.
        seed: Seed of every random choice, so a recipe always builds the same clips.
        sample_rate: `native` keeps every file at its own rate, so only a `resample`
            operation changes it; a number resamples every phrase and background to it
            when they are loaded.
        speech: Sources of the phrases that every speech condition corrupts.
        backgrounds: Background pools by name, referred to by `mix`, `pad` and `generate`.
        conditions: The corruptions; names must be unique.
        plugins: Python modules with your own operations, imported by `load_recipe`
            before the conditions are read. Kept here so `sources.json` says which
            code a set was built with.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    seed: int
    plugins: list[Annotated[str, Field(pattern=_MODULE)]] = Field(default_factory=list)
    sample_rate: Annotated[int, Field(gt=0)] | Literal["native"] = "native"
    speech: list[SpeechInput] = Field(min_length=1)
    backgrounds: dict[str, Background] = Field(default_factory=dict)
    conditions: list[Condition] = Field(min_length=1)

    @field_validator("backgrounds")
    @classmethod
    def _background_names(cls, value: dict[str, Background]) -> dict[str, Background]:
        """Keep background names apart from the built-in fills, so `fill: zero` is unambiguous."""
        for name in value:
            if name in _BUILT_IN_FILLS:
                raise ValueError(f"background name {name!r} is reserved for the built-in fill")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check that condition names are unique and every background an operation uses exists."""
        names = [condition.name for condition in self.conditions]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"condition names repeat: {', '.join(duplicates)}")
        for condition in self.conditions:
            for op in condition.ops:
                for background in op.backgrounds():
                    if background not in self.backgrounds:
                        raise ValueError(
                            f"condition {condition.name!r} uses unknown background {background!r}"
                        )
        return self


def load_recipe(path: Path) -> Recipe:
    """Read a recipe file, import its `plugins`, and validate it.

    Plugins are imported first because the conditions can name operations they
    register. A plugin is found on `sys.path` or next to the recipe file.

    Raises:
        ImportError: If a plugin cannot be imported (the cause is chained).
        ValidationError: If the recipe does not match the models above.
    """
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    plugins = raw.get("plugins") if isinstance(raw, dict) else None
    if isinstance(plugins, list):
        _import_plugins([str(name) for name in plugins], path.resolve().parent)
    return Recipe.model_validate(raw)


def _import_plugins(names: list[str], folder: Path) -> None:
    """Import each module, looking in `folder` too, so their `register_op` calls run.

    Raises:
        ImportError: If a module is not found, naming it and the recipe's folder; an
            import that fails inside the module is raised as it is.
    """
    sys.path.insert(0, str(folder))
    try:
        for name in names:
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as error:
                missing = error.name or ""
                if name != missing and not name.startswith(missing + "."):
                    raise  # the plugin exists, but something it imports does not
                raise ImportError(
                    f"recipe plugin {name!r} not found on sys.path or in {folder}"
                ) from error
    finally:
        sys.path.remove(str(folder))
