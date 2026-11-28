"""The YAML recipe of a test set: which speech, which backgrounds, which corruptions.

A recipe is data, not code: adding a condition or a dataset is a YAML change. Every
operation is written as a one-key mapping, `{mix: {background: music, snr_db: 0}}`, and an
operation with one main setting accepts it bare: `{resample: 8000}`, `{codec: opus}`.

Settings that vary from clip to clip are a `Range`: a number for a fixed value, or
`[low, high]` for a value drawn uniformly per clip from the recipe's seed.
"""

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

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


def _ordered(value: float | tuple[float, float]) -> float | tuple[float, float]:
    """Reject a `[low, high]` range whose low end is above its high end."""
    if isinstance(value, tuple) and value[0] > value[1]:
        raise ValueError(f"range low end {value[0]} is above high end {value[1]}")
    return value


type Range = Annotated[float | tuple[float, float], AfterValidator(_ordered)]

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


class _Op(_Model):
    """Base of operations; `shorthand` names the setting a bare value stands for."""

    shorthand: ClassVar[str | None] = None


class Mix(_Op):
    """Mix a background under the speech at `snr_db`, measured against the speech's RMS."""

    op: Literal["mix"] = "mix"
    background: str
    snr_db: Range

    shorthand: ClassVar[str | None] = "background"


class Pad(_Op):
    """Add `before` and `after` seconds of `fill` around the speech.

    `fill` is `zero` (digital silence), `floor` (white noise at `floor_dbfs`) or the name
    of a background.
    """

    op: Literal["pad"] = "pad"
    before: Range = 0.0
    after: Range = 0.0
    fill: str = "zero"
    floor_dbfs: float = -60.0


class Generate(_Op):
    """Make a clip without speech: `seconds` of `fill` (as in `Pad`)."""

    op: Literal["generate"] = "generate"
    fill: str
    seconds: Range
    floor_dbfs: float = -60.0

    shorthand: ClassVar[str | None] = "fill"


class Resample(_Op):
    """Change the sample rate, e.g. to 8000 to imitate narrowband telephony."""

    op: Literal["resample"] = "resample"
    rate: int = Field(gt=0)

    shorthand: ClassVar[str | None] = "rate"


class Codec(_Op):
    """Encode with a lossy codec and decode back, keeping the codec's artifacts.

    `bitrate` (bits per second) applies to `opus` only.
    """

    op: Literal["codec"] = "codec"
    name: Literal["pcm_mulaw", "pcm_alaw", "opus"]
    bitrate: int | None = Field(default=None, gt=0)

    shorthand: ClassVar[str | None] = "name"

    @model_validator(mode="after")
    def _bitrate_only_for_opus(self) -> Self:
        """Reject a bitrate for the PCM codecs, whose bitrate is fixed by the rate."""
        if self.bitrate is not None and self.name != "opus":
            raise ValueError(f"bitrate applies to opus only, not {self.name}")
        return self


class Gain(_Op):
    """Change the level by `db` decibels (negative is quieter)."""

    op: Literal["gain"] = "gain"
    db: Range

    shorthand: ClassVar[str | None] = "db"


_OPS: dict[str, type[_Op]] = {
    "mix": Mix,
    "pad": Pad,
    "generate": Generate,
    "resample": Resample,
    "codec": Codec,
    "gain": Gain,
}

type Op = Annotated[Mix | Pad | Generate | Resample | Codec | Gain, Field(discriminator="op")]


def _unwrap_op(value: Any) -> Any:
    """Turn the YAML form `{mix: {...}}` or `{resample: 8000}` into `{op: mix, ...}`."""
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"an operation is a mapping with one key, one of: {', '.join(_OPS)}")
    ((name, settings),) = value.items()
    if name not in _OPS:
        raise ValueError(f"unknown operation {name!r}, known: {', '.join(_OPS)}")
    if not isinstance(settings, dict):
        shorthand = _OPS[name].shorthand
        if shorthand is None:
            raise ValueError(f"operation {name!r} needs a mapping of settings")
        settings = {shorthand: settings}
    return {"op": name, **settings}


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
    ops: list[Op] = Field(default_factory=list)

    @field_validator("ops", mode="before")
    @classmethod
    def _ops(cls, value: Any) -> Any:
        """Accept operations in their one-key YAML form."""
        return [_unwrap_op(item) for item in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def _generate_iff_no_speech(self) -> Self:
        """A clip is either speech or generated: `generate` comes first, and only then."""
        generates = [isinstance(op, Generate) for op in self.ops]
        if self.speech == "none":
            if not generates or not generates[0] or self.count is None:
                raise ValueError("speech: none needs count and must start with generate")
            if any(generates[1:]):
                raise ValueError("generate can only be the first operation")
        elif any(generates) or self.count is not None:
            raise ValueError("generate and count are only for speech: none")
        return self


class Recipe(_Model):
    """A whole test set: inputs, backgrounds and conditions, reproducible from `seed`.

    Attributes:
        name: Name of the test set; becomes the output folder.
        seed: Seed of every random choice, so a recipe always builds the same clips.
        sample_rate: Rate of the speech before any `resample` operation.
        speech: Sources of the phrases that every speech condition corrupts.
        backgrounds: Background pools by name, referred to by `mix`, `pad` and `generate`.
        conditions: The corruptions; names must be unique.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    seed: int
    sample_rate: int = Field(default=16000, gt=0)
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
                background = _background_of(op)
                if background is not None and background not in self.backgrounds:
                    raise ValueError(
                        f"condition {condition.name!r} uses unknown background {background!r}"
                    )
        return self


def _background_of(op: _Op) -> str | None:
    """Return the background an operation draws from, if it draws from one."""
    if isinstance(op, Mix):
        return op.background
    if isinstance(op, Pad | Generate) and op.fill not in _BUILT_IN_FILLS:
        return op.fill
    return None


def load_recipe(path: Path) -> Recipe:
    """Read and validate a recipe file.

    Raises:
        ValidationError: If the recipe does not match the models above.
    """
    with path.open(encoding="utf-8") as handle:
        return Recipe.model_validate(yaml.safe_load(handle))
