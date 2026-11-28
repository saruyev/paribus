"""Tests for testsets.ops, on synthetic tones instead of speech."""

import numpy as np
import pytest

from testsets import ops
from testsets.ops import Clip

_RATE = 16000


def _tone(seconds: float = 1.0, hz: float = 440, level: float = 0.1) -> np.ndarray:
    """A sine tone standing in for speech."""
    t = np.arange(int(_RATE * seconds)) / _RATE
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _speech(seconds: float = 1.0) -> Clip:
    """A clip that is speech from start to end."""
    return Clip(_tone(seconds)[None, :], _RATE, [(0.0, seconds)])


def _rms(samples: np.ndarray) -> float:
    """Root mean square of samples."""
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _peak_hz(samples: np.ndarray, rate: int) -> float:
    """The frequency with the most energy."""
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(samples.size, 1 / rate)[spectrum.argmax()])


@pytest.mark.parametrize("snr_db", [0.0, 10.0])
def test_mix_sets_the_snr_against_speech(snr_db: float) -> None:
    """The added background is `snr_db` below the speech, by RMS."""
    clip = _speech()
    background = ops.noise_floor(clip.frames, -20, np.random.default_rng(0))

    mixed = ops.mix(clip, background, snr_db)

    added = mixed.samples - clip.samples
    assert 20 * np.log10(_rms(clip.samples) / _rms(added)) == pytest.approx(snr_db, abs=0.01)


def test_mix_measures_speech_only_inside_speech_spans() -> None:
    """Padding first must not make the background louder than the SNR says."""
    rng = np.random.default_rng(0)
    padded = ops.pad(_speech(), ops.zeros(3 * _RATE), ops.zeros(0))
    background = ops.noise_floor(padded.frames, -20, rng)

    mixed = ops.mix(padded, background, 0.0)

    speech = padded.samples[:, 3 * _RATE :]
    added = (mixed.samples - padded.samples)[:, 3 * _RATE :]
    assert _rms(added) == pytest.approx(_rms(speech), rel=0.05)


def test_mix_scales_down_instead_of_clipping() -> None:
    """A loud mix is scaled as a whole, so the SNR survives and nothing clips."""
    clip = Clip(_tone(level=0.95)[None, :], _RATE, [(0.0, 1.0)])
    background = _tone(hz=1000, level=0.95)

    mixed = ops.mix(clip, background, 0.0)

    assert np.abs(mixed.samples).max() == pytest.approx(0.98, abs=1e-4)


def test_mix_needs_speech_and_a_background_of_the_same_length() -> None:
    """Mistakes in the builder fail loudly instead of producing a wrong SNR."""
    with pytest.raises(ValueError, match="needs speech"):
        ops.mix(Clip(_tone()[None, :], _RATE, []), _tone(), 0.0)
    with pytest.raises(ValueError, match="frames"):
        ops.mix(_speech(), _tone(seconds=0.5), 0.0)
    with pytest.raises(ValueError, match="silent"):
        ops.mix(_speech(), ops.zeros(_RATE), 0.0)


def test_pad_shifts_speech_by_what_is_added_in_front() -> None:
    """Speech spans move by the length of `before`; `after` only lengthens the clip."""
    padded = ops.pad(_speech(), ops.zeros(_RATE // 2), ops.zeros(_RATE))

    assert padded.frames == int(2.5 * _RATE)
    assert padded.speech == [(0.5, 1.5)]


def test_generate_makes_a_clip_without_speech() -> None:
    """A generated clip has no speech spans: any transcript of it is a hallucination."""
    clip = ops.generate(ops.zeros(_RATE), _RATE)

    assert clip.speech == [] and clip.frames == _RATE


def test_excerpt_is_reproducible_and_loops_short_sources() -> None:
    """The same seed gives the same stretch; a short source is looped to fill."""
    source = _tone(seconds=0.25)

    first = ops.excerpt(source, _RATE, np.random.default_rng(3))
    second = ops.excerpt(source, _RATE, np.random.default_rng(3))

    assert first.shape == (_RATE,)
    assert np.array_equal(first, second)


def test_noise_floor_has_the_requested_level() -> None:
    """White noise at -60 dBFS has an RMS of 0.001."""
    noise = ops.noise_floor(10 * _RATE, -60, np.random.default_rng(0))

    assert _rms(noise) == pytest.approx(0.001, rel=0.01)


def test_draw_takes_fixed_values_and_ranges() -> None:
    """A number is used as is; a range is sampled within its bounds."""
    rng = np.random.default_rng(0)

    assert ops.draw(5.0, rng) == 5.0
    assert all(2 <= ops.draw((2.0, 3.0), rng) <= 3 for _ in range(100))


def test_resample_keeps_duration_and_speech() -> None:
    """Narrowband 8 kHz halves the frames; spans in seconds do not change."""
    clip = ops.resample(_speech(), 8000)

    assert clip.sample_rate == 8000 and clip.frames == 8000
    assert clip.speech == [(0.0, 1.0)]
    assert _peak_hz(clip.samples[0], 8000) == pytest.approx(440, abs=2)


def test_gain_changes_level_and_clips_at_full_scale() -> None:
    """-6 dB halves the amplitude; +20 dB clips instead of being scaled back."""
    clip = Clip(_tone(level=0.5)[None, :], _RATE, [(0.0, 1.0)])
    quieter = ops.gain(clip, -6.0206)
    louder = ops.gain(clip, 20.0)

    assert np.abs(quieter.samples).max() == pytest.approx(0.25, abs=1e-3)
    assert np.abs(louder.samples).max() == 1.0


@pytest.mark.parametrize(("name", "bitrate"), [("pcm_mulaw", None), ("opus", 12000)])
def test_codec_round_trip_keeps_length_rate_and_pitch(name: str, bitrate: int | None) -> None:
    """The codec changes the samples but not their length, rate or content."""
    clip = _speech()

    coded = ops.codec(clip, name, bitrate)

    assert coded.sample_rate == _RATE and coded.frames == clip.frames
    assert not np.array_equal(coded.samples, clip.samples)
    assert _peak_hz(coded.samples[0], _RATE) == pytest.approx(440, abs=2)


def test_unknown_codec_is_an_error() -> None:
    """Only the codecs the recipe allows are accepted."""
    with pytest.raises(ValueError, match="unknown codec 'mp3'"):
        ops.codec(_speech(), "mp3")


def test_join_plays_both_clips_and_shifts_the_second_speech() -> None:
    """The second clip's speech moves by the first clip plus the silence; the first stays."""
    first = Clip(_tone(seconds=1.0)[None, :], _RATE, [(0.25, 0.75)])
    second = Clip(_tone(seconds=0.5)[None, :], _RATE, [(0.0, 0.5)])

    joined = ops.join(first, ops.zeros(_RATE // 4), second)

    assert joined.seconds == 1.75
    assert joined.speech == [(0.25, 0.75), (1.25, 1.75)]


def test_join_refuses_clips_of_different_rates() -> None:
    """Joining 16 kHz and 8 kHz audio would play one of them at the wrong speed."""
    with pytest.raises(ValueError, match="16000 Hz and 8000 Hz"):
        ops.join(_speech(), ops.zeros(0), ops.resample(_speech(), 8000))


def test_bounds_of_fixed_values_and_ranges() -> None:
    """A fixed value is a range of one point."""
    assert ops.bounds(3.0) == (3.0, 3.0)
    assert ops.bounds((1.0, 2.0)) == (1.0, 2.0)


def _stereo(seconds: float = 1.0) -> Clip:
    """A stereo clip: 440 Hz on the left, 880 Hz on the right, speech throughout."""
    samples = np.stack([_tone(seconds, 440), _tone(seconds, 880)])
    return Clip(samples, _RATE, [(0.0, seconds)])


def test_a_clip_must_be_channels_by_frames() -> None:
    """Forgetting the channel axis fails at once instead of mixing up frames and channels."""
    with pytest.raises(ValueError, match=r"\(channels, frames\)"):
        Clip(_tone(), _RATE, [])


def test_stereo_keeps_its_channels_through_every_operation() -> None:
    """Mix, pad, resample, gain and both codec kinds keep the two channels apart."""
    clip = _stereo()
    background = ops.noise_floor(clip.frames, -30, np.random.default_rng(0))
    clip = ops.mix(clip, background, 20.0)
    clip = ops.pad(clip, ops.zeros(_RATE // 2), ops.zeros(0))

    for out, rate in [
        (ops.resample(clip, 8000), 8000),
        (ops.gain(clip, -6.0), _RATE),
        (ops.codec(clip, "pcm_mulaw"), _RATE),
        (ops.codec(clip, "opus", 32000), _RATE),
    ]:
        assert out.channels == 2 and out.speech == [(0.5, 1.5)]
        speech = out.samples[:, rate // 2 :]
        assert _peak_hz(speech[0], rate) == pytest.approx(440, abs=2)
        assert _peak_hz(speech[1], rate) == pytest.approx(880, abs=2)


def test_mix_puts_the_same_background_on_every_channel() -> None:
    """What is added is the same on both channels, at the SNR of both channels together."""
    clip = _stereo()
    background = ops.noise_floor(clip.frames, -20, np.random.default_rng(0))

    added = ops.mix(clip, background, 10.0).samples - clip.samples

    assert np.allclose(added[0], added[1])
    assert 20 * np.log10(_rms(clip.samples) / _rms(added)) == pytest.approx(10.0, abs=0.01)


def test_join_refuses_clips_of_different_channel_counts() -> None:
    """Joining mono to stereo would have to invent or drop a channel."""
    with pytest.raises(ValueError, match="1 and 2 channels"):
        ops.join(_speech(), ops.zeros(0), _stereo())


def test_generate_puts_the_fill_on_every_channel() -> None:
    """A generated stereo clip has the fill on both channels and no speech."""
    clip = ops.generate(ops.noise_floor(_RATE, -40, np.random.default_rng(0)), _RATE, 2)

    assert clip.channels == 2 and np.array_equal(clip.samples[0], clip.samples[1])


def test_split_cuts_windows_and_makes_speech_local() -> None:
    """2.5 s in 1 s windows: three pieces, the last shorter; spans cut to each window."""
    clip = Clip(_tone(2.5)[None, :], _RATE, [(0.5, 1.75)])

    pieces = ops.split(clip, 1.0)

    assert [piece.frames for piece in pieces] == [_RATE, _RATE, _RATE // 2]
    assert [piece.offset for piece in pieces] == [0.0, 1.0, 2.0]
    assert [piece.speech for piece in pieces] == [[(0.5, 1.0)], [(0.0, 0.75)], []]
    assert np.array_equal(np.concatenate([p.samples for p in pieces], axis=1), clip.samples)


def test_split_with_overlap_repeats_the_end_of_each_window() -> None:
    """With 0.5 s overlap, 1 s windows start every 0.5 s and stop once one reaches the end."""
    pieces = ops.split(_speech(2.0), 1.0, overlap=0.5)

    assert [piece.offset for piece in pieces] == [0.0, 0.5, 1.0]
    assert pieces[-1].frames == _RATE


def test_split_of_a_piece_adds_up_the_offsets() -> None:
    """Splitting a piece again keeps offsets on the timeline of the very first clip."""
    (_, second) = ops.split(_speech(2.0), 1.0)

    assert [piece.offset for piece in ops.split(second, 0.5)] == [1.0, 1.5]


def test_split_refuses_a_window_that_never_advances() -> None:
    """An overlap as long as the window would cut the same piece forever."""
    with pytest.raises(ValueError, match="overlap < seconds"):
        ops.split(_speech(), 1.0, overlap=1.0)
