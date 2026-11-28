# testsets: corrupted audio with a known right answer

`testsets` builds audio test sets out of speech. It takes clean phrases and damages them
on purpose with noise, music, silence, telephone codecs and low levels. It also makes
clips with no speech at all. The result is an ordinary paribus manifest, so the engine
reads it like any other source.

It is a tool that sits next to the package (`testsets/`, not `src/`). Nothing in
`paribus` depends on it.

## Why

To judge a VAD, a language detector, an audio tagger or an ASR model, you need the
truth: where the speech is, what was said, and how loud the noise was. Real pipeline data
rarely comes with that truth, and labeling it is slow and expensive. A built set has the
truth by construction, because the builder did the mixing itself.

The design is **paired**: every condition gets *the same* phrases. When WER goes up 30%
on `music_snr0`, the music caused it, not a different draw of phrases.

Some questions a set answers:

- Does the ASR hallucinate on silence, a noise floor, or music with no speech?
- At what SNR does the VAD start cutting words, or start firing on noise?
- Does language detection survive 8 kHz μ-law or low-bitrate opus?
- How do long recordings (30 s to 5 min) behave compared to short phrases?

## Quick start

```bash
uv sync --extra hf        # Hugging Face sources need the `hf` extra
python -m testsets.build testsets/recipes/t0-components.yaml --out ~/.cache/paribus/testsets
```

A recipe is a YAML file that says which speech to take, what to mix it with, and how to
damage it. The rest of this page explains how to write one. An existing set is never
overwritten silently: pass `--force` to rebuild it.

## How it works

```
recipe.yaml ──► load phrases (seeded) ──► for each condition: run its ops ──► wav + manifest
               (take, duration, concat)    (mix, pad, codec, ...)
```

| File | Role |
|---|---|
| `recipe.py` | The YAML schema (pydantic), the `Operation` base class and the built-in operations. A typo or an impossible setting fails at load time, not an hour into a build. |
| `ops.py` | The signal processing behind the built-in operations: small pure functions on a `Clip`. |
| `build.py` | Picks phrases, optionally joins them into long clips, runs every condition, and writes the set. |

### Reproducibility

- Each clip gets its own random generator, seeded from `(recipe seed, condition, clip number)`.
  A rebuild gives byte-identical files.
- Adding or removing a condition does not change any other condition's clips.
- `sources.json` records every input's pinned revision and license, and the plugins
  the set was built with. Revisions must be commit shas, not branch names, because a
  branch can move.
- The build writes to a hidden `.<name>.partial` folder and renames it at the end, so an
  interrupted build never leaves a half-written set behind.

## Making it yours

### 1. Use someone else's data

Every input is a paribus source config. The `hf` source streams a Hugging Face dataset at
a pinned revision. It downloads only the rows it reads, and caches each audio file once
under `~/.cache/paribus/hf/`.

```yaml
speech:
  - source:
      type: hf
      name: fleurs_ru                 # becomes part of every clip id
      dataset: google/fleurs
      revision: 0123abcd...           # a commit of refs/convert/parquet, never a branch
      data_dir: ru_ru                 # FLEURS keeps each language in a folder
      split: test
      text_column: raw_transcription
      transcripts: reference          # the column is gold, not a model's output
      meta_columns: [gender]          # copied into each clip's meta, for slicing
      language: ru
      license: CC-BY-4.0
    take: 30                          # how many phrases to use
    duration: [3, 20]                 # only phrases 3-20 s long
```

**Where speech is (`speech_segments`).** To score a VAD, or a model's timestamps, you
need to know where in each phrase the words are, not just that the phrase has speech.
Some datasets ship this as a column of forced alignments: one `{start, end}` item per word,
in seconds. Open the dataset's viewer on the Hub, find such a column, and name it in
`speech_segments_column`. For example, `gilkeyio/librispeech-alignments` has a `words`
column of `{word, start, end}` items:

```yaml
      dataset: gilkeyio/librispeech-alignments
      speech_segments_column: words   # extra keys such as `word` are ignored
```

Without such a column, the whole phrase counts as speech, and every clip says so with
`meta.speech_truth: utterance_bounds`. That is still exact about added silence and about
clips with no speech. It is only coarse inside the phrase.

All settings of every source type are listed under [Source settings](#source-settings).

Backgrounds work the same way. A pool that mixes licenses, like MUSAN noise, can be
filtered to the ones you are allowed to use:

```yaml
backgrounds:
  noise:
    source: {type: hf, dataset: ..., revision: ..., split: train, license: mixed}
    license_column: audio_license     # meta key holding each clip's own license
    licenses: [CC-BY-4.0, public-domain]
```

The shipped recipes in `recipes/` are ready-made examples of both.

### 2. Add your own data

**A folder.** Every audio file under `root` becomes a phrase. A `.txt` file next to it
with the same name is its transcript.

```
/data/calls/
├── agent-a/001.wav
├── agent-a/001.txt     # "thank you for calling"
└── agent-b/002.flac
```

```yaml
speech:
  - source:
      type: folder
      root: /data/calls
      transcripts: reference          # or `none` if the .txt files are not gold
      language: en
      license: private                # your data; it never leaves your machine
    take: 100
```

**A manifest**, when you have more than audio and text, such as known speech spans or
your own metadata. Use one JSON line per phrase:

```json
{"id": "call-17", "audio": "audio/call-17.wav", "language": "en", "reference": "hello there", "speech_segments": [[0.4, 2.1]], "meta": {"channel_type": "mobile"}}
```

```yaml
  - source: {type: manifest, path: /data/calls.jsonl, license: private}
    take: 100
```

Your own noise goes under `backgrounds` in exactly the same way, for example a folder
of call-center recordings:

```yaml
backgrounds:
  office:
    source: {type: folder, root: /data/office-noise, license: private}
```

### Source settings

Every source has these settings:

| Setting | Meaning |
|---|---|
| `type` | `hf`, `folder` or `manifest`, or a type a plugin adds (see below). |
| `license` | Required: the terms of the data, or `private` for your own. It goes into `sources.json`. |
| `name` | Optional. It becomes the middle part of clip ids and must differ between inputs. It defaults to the dataset, the folder, or the manifest file name. |

**`hf`**: a Hugging Face Hub dataset, streamed. It needs the `hf` extra.

| Setting | Default | Meaning |
|---|---|---|
| `dataset` | | The repository id, e.g. `google/fleurs`. |
| `revision` | | A commit sha or tag. It is required, so two builds never read different data. |
| `subset` | none | The dataset's configuration (`name` in `datasets.load_dataset`). |
| `data_dir` | none | A folder inside the dataset, for datasets laid out as folders rather than configurations. See below. |
| `split` | | e.g. `test`. |
| `audio_column` | `audio` | The column holding the audio. |
| `text_column` | none | The column holding a transcript. |
| `transcripts` | `none` | What `text_column` is: `reference` (gold) or `output` (a model's). |
| `id_column` | none | The column holding ids. Without it, ids are `<split>/<row>`. |
| `language` | none | The expected language of every row, e.g. `en`. |
| `meta_columns` | `[]` | Columns copied into each clip's `meta`. |
| `speech_segments_column` | none | The column of `{start, end}` spans or `[start, end]` pairs, in seconds. |
| `limit` | none | Read only the first N rows. |
| `cache_dir` | `~/.cache/paribus/hf` | Where audio files are cached. |

**`subset` or `data_dir`?** Look at the dataset's file tree on the Hub. If the dataset
lists configurations, pick one with `subset`. If it is a tree of folders such as
`ru_ru/test/0000.parquet`, name the folder with `data_dir`, and the split comes from the
folder below it. Script datasets such as `google/fleurs` can only be read through their
parquet conversion, the `refs/convert/parquet` branch. There, a pinned commit shows
folders, not configurations, so FLEURS needs `data_dir`. Each `data_dir` gets its own
cache folder.

**`folder`**: every audio file under a folder, recursively.

| Setting | Default | Meaning |
|---|---|---|
| `root` | | The folder. |
| `patterns` | wav, flac, mp3, ogg, opus, m4a, webm | Glob patterns of audio files. |
| `transcripts` | `none` | What the text file next to each audio file is: `reference` or `output`. |
| `transcript_suffix` | `.txt` | The text file's suffix: `a/001.wav` pairs with `a/001.txt`. |
| `language` | none | The expected language of every file. |

**`manifest`**: a `.jsonl` or `.csv` file of records in the paribus format (`id`,
`audio`, `language`, `reference`, `speech_segments`, `meta`...).

| Setting | Default | Meaning |
|---|---|---|
| `path` | | The file. Its suffix picks the format. Audio paths are relative to it. |
| `time_unit` | `s` | `s` or `ms`, for every time in the file. |
| `meta_types` | `{}` | Types of `meta` keys, e.g. `{snr: float}`. These are parsed from CSV text and checked in JSONL. |

**Another kind of source** (a database, an object store, your call system's export) is
a class with `name`, `license` and `records()`, plus a config class. You register it from
a plugin, the same way as an operation (see [4](#4-add-your-own-logic)):

```python
"""My call store as a paribus source."""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from paribus.engine.data import SourceConfig, register_source
from paribus.engine.types import SampleRecord


class CallStoreConfig(SourceConfig):
    """Settings of my call store's CSV export."""

    type: Literal["call_store"] = "call_store"
    export: Path


class CallStoreSource:
    """One record per row of the export: `call_id,file,transcript`."""

    def __init__(self, config: CallStoreConfig) -> None:
        """Keep the config; nothing is read until `records()`."""
        self.config = config
        self.name = config.name or "call_store"
        self.license = config.license

    def records(self) -> Iterator[SampleRecord]:
        """Yield one record per row."""
        with self.config.export.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield SampleRecord(
                    id=row["call_id"],
                    audio=self.config.export.parent / row["file"],
                    reference=row["transcript"] or None,
                    language="en",
                )


register_source("call_store", CallStoreConfig, CallStoreSource)
```

```yaml
plugins: [my_sources]
speech:
  - source: {type: call_store, export: /data/calls.csv, license: private}
    take: 100
```

### 3. Use the built-in logic

A **condition** is a name plus a list of operations applied in order. `clean`, with no
operations, is the baseline the others are compared with. Every condition gets the same
phrases.

| Op | Settings | What it does |
|---|---|---|
| `mix` | `background`, `snr_db` | Adds a background under the speech. The SNR is measured on the speech spans only, so padding does not skew it. |
| `pad` | `before`, `after`, `fill`, `floor_dbfs` | Adds seconds before and after. `fill` is `zero` (digital silence), `floor` (white noise at `floor_dbfs`, default -60) or a background name. |
| `generate` | `fill`, `seconds`, `floor_dbfs`, `sample_rate`, `channels` | Makes a clip with no speech. It must be the first op of a `speech: none` condition, which also needs `count`. The rate defaults to the recipe's, or 16000 under `native`; `channels` defaults to 1. |
| `resample` | `rate` | Changes the sample rate, for example 8000 for narrowband telephony. |
| `codec` | `name`, `bitrate` | Encodes and decodes with `pcm_mulaw`, `pcm_alaw` or `opus` (`bitrate` is for opus only), keeping the artifacts. |
| `gain` | `db` | Changes the level. It clips at full scale, because clipping is a corruption worth testing too. |
| `split` | `seconds`, `overlap` | Cuts the clip into windows, as a pipeline does before sending long audio to a model. The operations after it run on each piece (see below). |

Any number can be a fixed value or a `[low, high]` range drawn per clip. Whatever was
drawn is written into the clip's `meta`. An operation with one main setting can be written
bare: `{resample: 8000}`, `{codec: opus}`, `{gain: -20}`. Otherwise settings are a
mapping: `{mix: {background: music, snr_db: 5}}`.

**Sample rate and channels.** By default nothing is converted behind your back:

```yaml
sample_rate: native   # every file keeps its own rate; or a number, e.g. 16000, for all
channels: keep        # a stereo call stays stereo; or `mono` to average the channels
```

Under `native`, only a `resample` operation changes a clip's rate, so where you put it is
part of the experiment. Backgrounds are converted to the rate of the clip they go under.
They are always mono and go on every channel the same way, and `mix` measures the SNR
over all channels together. The speech truth is one list for the whole file, not one per
channel.

Common recipes:

```yaml
conditions:
  - name: clean

  # An SNR ladder: where does quality fall off?
  - {name: noise_snr0,  ops: [{mix: {background: noise, snr_db: 0}}]}
  - {name: noise_snr10, ops: [{mix: {background: noise, snr_db: 10}}]}

  # Silence around the speech: does the ASR invent words in it?
  - name: pad_floor
    ops: [{pad: {before: [1, 5], after: [1, 5], fill: floor, floor_dbfs: -50}}]

  # A phone line: narrowband, μ-law, a bit quiet.
  - name: telephone
    ops: [{resample: 8000}, {codec: pcm_mulaw}, {gain: [-12, -6]}]

  # No speech at all: anything the ASR outputs here is a hallucination.
  - name: only_music
    speech: none
    count: 20
    ops: [{generate: {fill: music, seconds: [5, 15]}}]
```

If an operation repeats in one condition, its meta keys get suffixes: `pad_before`,
`pad_before_2`.

**A dataset as it is.** With the defaults (`native`, `keep`), a condition with no
operations, like `clean`, copies each source file byte for byte, in its own format: an mp3
stays an mp3. You still get `take`, `duration` and the manifest, so a recipe with only
`clean` cuts a fixed, reproducible sample out of a large dataset:

```yaml
conditions:
  - name: clean
```

A file is copied only when nothing about it changed. With a fixed `sample_rate`,
`channels: mono` on a multichannel file, a source record that picks one `channel`, or
`concat`, the clip is written as 16-bit wav instead.

If you need neither a sample nor a manifest, you do not need `testsets` at all. paribus
reads `hf`, `folder` and `manifest` sources directly, with the same settings as above.

**Long clips.** `concat` on a speech input joins consecutive phrases of one group, such
as a book chapter, into long clips before any condition runs. Models split long audio
into windows and tend to hallucinate at the seams, so long clips are worth their own
input:

```yaml
  - source: {type: hf, name: librispeech_long, ...}
    take: 10
    concat:
      seconds: [30, 300]              # target length per clip
      gap: [0.3, 1.0]                 # silence between phrases
      group_by: id_prefix             # 1089-134686-0000 -> chapter 1089-134686
```

The phrases of a `concat` input must share one sample rate and one channel count, or
the build stops before joining anything. Joining them otherwise would need a silent
conversion. Set `sample_rate` to a number and `channels: mono` to join mixed files.

**Cutting into pieces.** `split` cuts a clip into windows, and every operation after it
runs on each piece separately. Each piece is its own record in the output, with
`group_id` (the whole clip), `offset` (where the piece starts in it) and speech spans
local to the piece. That is how paribus sees audio a pipeline has already chunked.

The order of operations is then the experiment. For example, does it matter whether you
resample the original file or each chunk?

```yaml
conditions:
  - {name: resample_whole,  ops: [{resample: 8000}, {split: 30}]}
  - {name: resample_pieces, ops: [{split: 30}, {resample: 8000}]}
```

Both give the same pieces with the same offsets and speech spans, so they line up piece
by piece. Only the samples differ, near the cuts. `split` takes `{seconds: [20, 40],
overlap: 1}` too: the window length is drawn once per clip, and windows overlap by
`overlap` seconds.

### 4. Add your own logic

When the built-in operations are not enough, you write an operation in your own file.
You import only what `testsets` exports, and you do not touch its modules.

Say your calls go over a network that drops packets. Put this in `my_corruptions.py`,
next to your recipe:

```python
"""My own corruptions for testsets recipes."""

from dataclasses import replace
from typing import ClassVar

import numpy as np

from testsets import Clip, OpContext, Operation, Range, register_op


def drop_chunks(clip: Clip, rate: float, chunk_ms: float, rng: np.random.Generator) -> Clip:
    """Silence random `chunk_ms` chunks, each with probability `rate`, like lost packets."""
    chunk = max(1, int(clip.sample_rate * chunk_ms / 1000))
    lost = rng.random(-(-clip.frames // chunk)) < rate  # one draw per chunk
    keep = np.repeat(~lost, chunk)[: clip.frames]
    return replace(clip, samples=(clip.samples * keep).astype(np.float32))


@register_op("dropout")
class Dropout(Operation):
    """Packet loss: silence random chunks of `chunk_ms`, each lost with probability `rate`."""

    rate: Range  # settings are fields, validated when the recipe loads
    chunk_ms: float = 20.0

    shorthand: ClassVar[str | None] = "rate"  # lets a recipe write `{dropout: 0.05}`

    def apply(self, clip: Clip, ctx: OpContext) -> Clip:
        """Draw the loss rate for this clip, record it in meta, drop the chunks."""
        rate = ctx.value("dropout_rate", self.rate, 3)
        return drop_chunks(clip, rate, self.chunk_ms, ctx.rng)
```

Then name the module in the recipe and use the operation like a built-in one:

```yaml
plugins: [my_corruptions]           # imported before the conditions are read

conditions:
  - name: voip_loss
    ops:
      - resample: 8000
      - codec: {name: opus, bitrate: 12000}
      - dropout: [0.02, 0.15]
```

Your part of the contract:

- **Settings are pydantic fields.** Use `Range` for anything that should vary per clip.
  Add a `@model_validator` if, for example, `rate` must be within [0, 1], and a bad recipe
  will fail at load time.
- **Randomness comes from `ctx.rng` only.** Then rebuilds stay byte-identical.
- **Draw settings with `ctx.value(key, setting, digits)`,** or record a value with
  `ctx.note(key, value)`. It shows up in the clip's `meta`, so results can be sliced by it.
- **`clip.samples` is shaped (channels, frames)**, float32 in [-1, 1]; mono is one
  channel. Index frames as `samples[:, start:end]`. Writing `samples[:n]` slices
  *channels*, and silently does the wrong thing. Use `clip.channels`, `clip.frames` and
  `clip.sample_rate` rather than the array's shape. A `Clip` with a one-dimensional array
  is refused.
- **Keep `clip.speech` true.** If you shift or cut audio, shift or cut the spans too, the
  way `pad` does. Dropping chunks leaves them as they are, because the speaker did talk there.
- **Backgrounds come from `ctx.fill(name, frames, floor_dbfs, sample_rate)`.** It returns
  mono `(frames,)` at the rate you ask for, usually `clip.sample_rate`. Put it on every
  channel with `samples + fill[None, :]`. Override `backgrounds()` so the recipe checks
  that the ones you use exist.
- **To make clips without speech,** set `starts_clip = True` and override `make(ctx)`
  instead of `apply`, the way `generate` does. `ctx.sample_rate` is the rate to use when
  nothing else says.
- **To turn one clip into several,** set `splits = True` and override `expand(clip, ctx)`
  to return the pieces, the way `split` does. Give each piece an `offset`.

Keep the signal processing in a plain function, like `drop_chunks` above, and test it
directly with a synthetic tone and a seeded `np.random.default_rng(0)`. The class only
wires settings to it.

A plugin is found on `sys.path` or in the recipe's folder. A name already taken (`mix`,
or another plugin's) is an error, so a plugin cannot silently replace an operation. A
plugin is ordinary Python and runs when the recipe loads, so only use recipes whose
plugins you trust.

### 5. Read what you get

```
<out>/<recipe name>/
├── audio/<condition>/<source>/<id>.wav   # 16-bit PCM, its channels, its final rate
│                                         # (a copied original keeps its own format;
│                                         #  a piece of `split` is .../<id>/<part>.wav)
├── manifest.jsonl                        # one record per clip
└── sources.json                          # every input with revision, license and use; plugins
```

An illustrative line of `manifest.jsonl`, formatted here for reading:

```json
{
  "id": "noise_snr10/librispeech_en/1089-134686-0003",
  "audio": "audio/noise_snr10/librispeech_en/1089-134686-0003.wav",
  "language": "en",
  "reference": "stuff it into you his belly counselled him",
  "speech_segments": [[0.52, 1.34], [1.41, 3.07]],
  "meta": {
    "condition": "noise_snr10",
    "source_id": "librispeech_en/1089-134686-0003",
    "speech_truth": "alignment",
    "snr_db": 10.0,
    "background_id": "musan_noise/noise-free-sound-0042",
    "sex": "M"
  }
}
```

What the fields mean:

- `speech_segments` is the truth for VAD: where speech is in *this* file, after padding
  and joining.
- `reference` is the truth for ASR. It does not change with the corruption, because the
  words said are the same.
- `meta.condition` says what was done to the clip, and the other `meta` keys say how much
  of it (the drawn `snr_db`, `pad_before`, `gain_db`, `codec`...).
- `meta.source_id` is the same across conditions for the same phrase. It is what makes the
  comparison paired: line up `clean` against `noise_snr10` phrase by phrase.
- `meta.speech_truth` says how exact `speech_segments` is: `alignment` means word timings,
  `utterance_bounds` means only the phrase's start and end.
- Columns you asked for with `meta_columns` (`sex` here) are carried over, so you can
  slice by speaker, channel, or whatever your data has.
- Clips without speech have no `reference` and an empty `speech_segments`.
- Pieces of a `split` also have `group_id` (the whole clip, `<condition>/<source>/<id>`),
  `offset` (seconds into it) and `meta.split_part`. Their `speech_segments` are local:
  add `offset` to get time in the whole clip. `reference` is still the whole phrase's,
  because which words fall into which piece is not known without word timings.

What to do with it:

- **Run your pipeline or model on it with paribus.** It is a `manifest` source:
  `{type: manifest, path: <out>/<name>/manifest.jsonl, license: ...}`. Then compare
  quality, hallucinations and VAD errors per `meta.condition`.
- **Or look at it yourself.** The files are plain wav, and the manifest loads into
  anything:

  ```python
  import pandas as pd

  rows = pd.read_json("manifest.jsonl", lines=True)
  meta = pd.json_normalize(rows["meta"])
  print(meta.groupby("condition").size())  # clips per condition
  print(meta.query("condition == 'noise_snr10'")["snr_db"].describe())
  ```

- **Share it with care.** The licenses come from the inputs and are listed in
  `sources.json`. Check them before you publish a built set. Only recipes go into git,
  never audio.

## Shipped recipes

- `recipes/t0-components.yaml`: the set used to choose the VAD, tagger, LID and local ASR.
  English and Russian speech, 13 conditions with speech and 4 without, about 1,500 clips.
- `recipes/t0-long.yaml`: long LibriSpeech clips (30 s to 5 min) under 5 conditions.

Every revision in them is pinned to a commit sha, so they build the same sets anywhere.
