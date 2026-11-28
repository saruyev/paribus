"""Corruption operations: small pure functions from a clip to a corrupted clip.

Every operation takes a `Clip` and its settings (with ranges already drawn) and returns
a new `Clip`; nothing is modified in place and nothing is random except through the
`rng` passed in, so a build is reproducible from the recipe's seed. Operations also keep
`speech` (where speech is, in seconds) true, since that is the ground truth the VAD and
hallucination benchmarks are scored against.
"""

import io
from dataclasses import dataclass, replace

import av
import numpy as np
import numpy.typing as npt
import soxr

type Samples = npt.NDArray[np.float32]
type Span = tuple[float, float]

# Peak level a mix is scaled down to when it would clip; a little under 1.0 so a later
# codec or resampler overshoot does not clip either.
_PEAK = 0.98
# Container and stream codec name for each recipe codec name.
_CODECS = {
    "pcm_mulaw": ("wav", "pcm_mulaw"),
    "pcm_alaw": ("wav", "pcm_alaw"),
    "opus": ("ogg", "libopus"),
}


@dataclass(frozen=True)
class Clip:
    """Mono audio being built, with where its speech is.

    Attributes:
        samples: float32 mono samples, shaped (frames,).
        sample_rate: Frames per second.
        speech: Spans of speech in seconds; empty for a clip without speech.
    """

    samples: Samples
    sample_rate: int
    speech: list[Span]

    @property
    def frames(self) -> int:
        """Number of frames in the clip."""
        return int(self.samples.shape[0])

    @property
    def seconds(self) -> float:
        """Length of the clip in seconds."""
        return self.frames / self.sample_rate


def draw(value: float | tuple[float, float], rng: np.random.Generator) -> float:
    """Return a fixed value as is, or a uniform draw from a `(low, high)` range."""
    if isinstance(value, tuple):
        return float(rng.uniform(value[0], value[1]))
    return float(value)


def bounds(value: float | tuple[float, float]) -> tuple[float, float]:
    """The `(low, high)` of a range; a fixed value is a range of one point."""
    return value if isinstance(value, tuple) else (value, value)


def zeros(frames: int) -> Samples:
    """Digital silence: exact zeros, which some models treat differently from quiet."""
    return np.zeros(frames, dtype=np.float32)


def noise_floor(frames: int, dbfs: float, rng: np.random.Generator) -> Samples:
    """White Gaussian noise with an RMS of `dbfs`, like a quiet room or a line's hiss."""
    return (rng.standard_normal(frames) * 10 ** (dbfs / 20)).astype(np.float32)


def excerpt(source: Samples, frames: int, rng: np.random.Generator) -> Samples:
    """A random stretch of `frames` from `source`, looping a source that is too short."""
    if source.shape[0] == 0:
        raise ValueError("cannot take an excerpt of empty audio")
    if source.shape[0] < frames:
        source = np.tile(source, frames // source.shape[0] + 1)
    start = int(rng.integers(0, source.shape[0] - frames + 1))
    return source[start : start + frames].astype(np.float32, copy=True)


def mix(clip: Clip, background: Samples, snr_db: float) -> Clip:
    """Add `background` under the speech so that speech is `snr_db` louder (by RMS).

    The speech level is measured over the speech spans only, so silence added by an
    earlier `pad` does not make the background louder than intended. `background` must
    be as long as the clip (see `excerpt`). If the mix would clip, the whole clip is
    scaled down, which keeps the SNR.

    Raises:
        ValueError: If the clip has no speech or the background is silent or too short.
    """
    if not clip.speech:
        raise ValueError("mix needs speech to set the level against")
    if background.shape[0] != clip.frames:
        raise ValueError(f"background has {background.shape[0]} frames, clip has {clip.frames}")
    background_rms = _rms(background)
    if background_rms == 0:
        raise ValueError("background is silent, so no gain gives the requested SNR")
    speech_rms = _rms(np.concatenate(_speech_parts(clip)))
    gain = speech_rms / (background_rms * 10 ** (snr_db / 20))
    return replace(clip, samples=_peak_safe(clip.samples + gain * background))


def pad(clip: Clip, before: Samples, after: Samples) -> Clip:
    """Put `before` in front of the clip and `after` behind it, shifting the speech."""
    shift = before.shape[0] / clip.sample_rate
    samples = np.concatenate([before, clip.samples, after]).astype(np.float32)
    speech = [(start + shift, end + shift) for start, end in clip.speech]
    return replace(clip, samples=samples, speech=speech)


def join(first: Clip, silence: Samples, second: Clip) -> Clip:
    """Play `first`, then `silence`, then `second`, keeping the speech spans of both.

    It is `pad` of the second clip by everything before it, so the shifting of speech
    spans lives in one place.

    Raises:
        ValueError: If the clips have different sample rates.
    """
    if first.sample_rate != second.sample_rate:
        raise ValueError(f"cannot join {first.sample_rate} Hz and {second.sample_rate} Hz")
    moved = pad(second, np.concatenate([first.samples, silence]), zeros(0))
    return replace(moved, speech=[*first.speech, *moved.speech])


def generate(fill: Samples, sample_rate: int) -> Clip:
    """A clip without speech made of `fill` (silence, a noise floor or a background)."""
    return Clip(samples=fill.astype(np.float32), sample_rate=sample_rate, speech=[])


def resample(clip: Clip, rate: int) -> Clip:
    """Change the sample rate with soxr; speech spans are in seconds and stay as they are."""
    if rate == clip.sample_rate:
        return clip
    samples = soxr.resample(clip.samples, clip.sample_rate, rate, quality="HQ")
    return replace(clip, samples=samples.astype(np.float32), sample_rate=rate)


def gain(clip: Clip, db: float) -> Clip:
    """Change the level by `db` decibels, clipping at full scale like an integer file.

    Too much gain is not scaled back as in `mix`: the clipping it causes is a corruption
    worth testing on its own.
    """
    samples = np.clip(clip.samples * 10 ** (db / 20), -1.0, 1.0).astype(np.float32)
    return replace(clip, samples=samples)


def codec(clip: Clip, name: str, bitrate: int | None = None) -> Clip:
    """Encode with a lossy codec and decode back, so the codec's artifacts stay.

    The decoder may return another rate (opus always decodes at 48 kHz) and a few frames
    more or less; the result is resampled back to the clip's rate and cut or padded to its
    length, so speech spans stay valid.

    Raises:
        ValueError: If `name` is not a supported codec.
    """
    if name not in _CODECS:
        raise ValueError(f"unknown codec {name!r}, known: {', '.join(_CODECS)}")
    container_format, codec_name = _CODECS[name]
    encoded = _encode(clip, container_format, codec_name, bitrate)
    decoded, rate = _decode_mono(encoded)
    decoded = resample(Clip(decoded, rate, []), clip.sample_rate).samples
    decoded = np.pad(decoded, (0, max(0, clip.frames - decoded.shape[0])))[: clip.frames]
    return replace(clip, samples=decoded.astype(np.float32))


def _encode(clip: Clip, container_format: str, codec_name: str, bitrate: int | None) -> bytes:
    """Encode a mono clip into an in-memory file of the given container and codec."""
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=container_format) as out:
        stream = out.add_stream(codec_name, rate=clip.sample_rate, layout="mono")
        if not isinstance(stream, av.AudioStream):  # narrows PyAV's union for mypy
            raise TypeError(f"{codec_name} is not an audio codec")
        if bitrate is not None:
            stream.bit_rate = bitrate
        frame = av.AudioFrame.from_ndarray(clip.samples[None, :], format="fltp", layout="mono")
        frame.sample_rate = clip.sample_rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buffer.getvalue()


def _decode_mono(data: bytes) -> tuple[Samples, int]:
    """Decode an in-memory mono file to float32 samples at the decoder's rate."""
    with av.open(io.BytesIO(data), "r") as container:
        stream = container.streams.audio[0]
        converter = av.AudioResampler(format="fltp", layout="mono", rate=stream.rate)
        chunks = [
            converted.to_ndarray()[0]
            for frame in container.decode(stream)
            for converted in converter.resample(frame)
        ]
        chunks += [converted.to_ndarray()[0] for converted in converter.resample(None)]
        return np.concatenate(chunks).astype(np.float32), stream.rate


def _speech_parts(clip: Clip) -> list[Samples]:
    """The clip's samples inside its speech spans."""
    rate = clip.sample_rate
    return [clip.samples[int(start * rate) : int(end * rate)] for start, end in clip.speech]


def _rms(samples: Samples) -> float:
    """Root mean square level, computed in float64 so long clips do not lose precision."""
    if samples.shape[0] == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _peak_safe(samples: Samples) -> Samples:
    """Scale the whole clip down if its peak is above `_PEAK`, otherwise leave it."""
    peak = float(np.abs(samples).max()) if samples.shape[0] else 0.0
    scale = _PEAK / peak if peak > _PEAK else 1.0
    return (samples * scale).astype(np.float32)
