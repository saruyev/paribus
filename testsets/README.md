# testsets: corrupted audio with a known right answer

`testsets` builds audio test sets from public speech. It takes clean phrases and damages
them on purpose with noise, music, silence, telephone codecs and low levels, and it also
makes clips with no speech at all. The result is an ordinary paribus manifest, so the
engine reads it like any other source.

It is a tool next to the package (`testsets/`, not `src/`). Nothing in `paribus` depends
on it.

## Why

Before you can judge a VAD, a language detector, an audio tagger or an ASR model, you
need the truth: where the speech is, what was said, and how loud the noise was. Real
pipeline data rarely comes with that truth, and labeling it is slow and expensive. A
built set has the truth by construction, because the builder did the mixing itself.

The design is **paired**: every condition gets *the same* phrases. When WER goes up 30%
on `music_snr0`, the music caused it, not a different draw of phrases.

Some questions this answers:

- Does the ASR hallucinate on silence, a noise floor or music with no speech?
- At what SNR does the VAD start cutting words, or start firing on noise?
- Does language detection survive 8 kHz μ-law, or low-bitrate opus?
- How do long recordings (30 s to 5 min) behave compared to short phrases?

## Quick start

```bash
uv sync --extra hf        # Hugging Face sources need the `hf` extra
python -m testsets.build testsets/recipes/t0-components.yaml --out ~/.cache/paribus/testsets
```

The output looks like this:

```
~/.cache/paribus/testsets/t0-components/
├── audio/<condition>/<source>/<id>.wav   # mono 16-bit PCM
├── manifest.jsonl                        # one SampleRecord per clip
└── sources.json                          # every input with its revision, license and use
```

Feed `manifest.jsonl` to paribus as a `manifest` source, then slice the results by
`meta.condition`.

Each record carries:

| Field | What it holds |
|---|---|
| `id` | `<condition>/<source>/<phrase id>` |
| `reference` | The transcript of the original phrase (empty for clips without speech) |
| `speech_segments` | Where speech is *after* padding and joining, in seconds |
| `meta.condition` | The condition the clip belongs to |
| `meta.source_id` | The original phrase, so you can compare one phrase across conditions |
| `meta.snr_db`, `pad_before`, `gain_db`, `codec`, `background_id`, ... | Each setting as it was drawn for this clip |
| `meta.speech_truth` | Where the speech segments come from: word alignments, or the phrase's bounds |

An existing set is never overwritten silently: pass `--force` to rebuild it.

## How it works

```
recipe.yaml ──► load phrases (seeded) ──► for each condition: run its ops ──► wav + manifest
               (take, duration, concat)    (mix, pad, codec, ...)
```

| File | Role |
|---|---|
| `recipe.py` | The YAML schema (pydantic). A typo or an impossible setting fails at load time, not an hour into a build. |
| `ops.py` | The corruptions: small pure functions `Clip -> Clip`. They keep `speech` spans correct, so the truth stays true. |
| `build.py` | Picks phrases, optionally joins them into long clips, runs every condition, and writes the set. |

### Reproducibility

- Each clip gets its own random generator seeded from `(recipe seed, condition, clip number)`.
  A rebuild gives byte-identical files.
- Adding or removing a condition does not change any other condition's clips.
- `sources.json` records the pinned revision and license of every input. Revisions must be
  commit shas, not branch names, because a branch can move.
- The build writes to a hidden `.<name>.partial` folder and renames it at the end, so an
  interrupted build never leaves a half-written set.

### Where the speech truth comes from

- **English (LibriSpeech):** forced word alignments (`gilkeyio/librispeech-alignments`).
  These are exact enough to score VAD and timestamps.
- **Other languages:** the phrase's start and end (`meta.speech_truth: utterance_bounds`).
  This is coarser, but still right about padding and clips without speech.

## Recipes

A recipe is data, not code. A minimal one:

```yaml
name: my-set
seed: 42
sample_rate: 16000

speech:
  - source: {type: folder, root: /data/calls, license: internal}
    take: 100
    duration: [3, 20]          # keep phrases 3-20 s long

backgrounds:
  office: {source: {type: folder, root: /data/office-noise, license: internal}}

conditions:
  - name: clean
  - name: office_snr5
    ops:
      - mix: {background: office, snr_db: [3, 7]}   # drawn per clip
  - name: phone
    ops:
      - pad: {before: [0.5, 2], after: 1, fill: floor}
      - resample: 8000
      - codec: pcm_mulaw
  - name: only_office
    speech: none
    count: 20
    ops:
      - generate: {fill: office, seconds: [5, 15]}
```

Any number can be a fixed value or a `[low, high]` range drawn per clip. An operation with
one main setting can be written bare: `{resample: 8000}`, `{codec: opus}`, `{gain: -20}`.

### Operations

| Op | Settings | What it does |
|---|---|---|
| `mix` | `background`, `snr_db` | Adds a background under the speech. The SNR is measured against the speech spans only, so padding does not skew it. |
| `pad` | `before`, `after`, `fill`, `floor_dbfs` | Adds seconds before and after the speech. `fill` is `zero` (digital silence), `floor` (white noise at `floor_dbfs`) or a background name. |
| `generate` | `fill`, `seconds`, `floor_dbfs` | Makes a clip with no speech. It must be the first op, and only in a `speech: none` condition with a `count`. |
| `resample` | `rate` | Changes the sample rate, for example 8000 for narrowband telephony. |
| `codec` | `name`, `bitrate` | Encodes and decodes with `pcm_mulaw`, `pcm_alaw` or `opus` (`bitrate` is for opus only), keeping the codec's artifacts. |
| `gain` | `db` | Changes the level. It clips at full scale, because clipping is a corruption worth testing too. |

### Speech inputs

`source` is any paribus source config (`folder`, `manifest`, `hf`), exactly as
`build_source` takes it. On top of that:

- `take`: how many phrases to use from this source.
- `duration`: keep only phrases within `[low, high]` seconds.
- `concat: {seconds: [30, 300], gap: [0.3, 1.0]}`: joins consecutive phrases of one group
  (a book chapter, by id prefix or a `meta` key) into long clips.

### Backgrounds

These are pools of audio, named and then used by `mix`, `pad` and `generate`. For a pool
with mixed licenses, `license_column` plus `licenses` keep only the allowed clips. The
names `zero` and `floor` are reserved for the built-in fills.

## Making it yours

| You want | You change |
|---|---|
| Your own speech or noise | The recipe: point a `folder` or `manifest` source at it. No code. |
| A new corruption mix (your codec + your call-center noise at 5 dB) | The recipe: a new condition. No code. |
| A new kind of corruption (reverb, dropouts, clipping) | Three places: a function in `ops.py`, a model in `recipe.py` (added to `_OPS` and `Op`), and a branch in `_Build.apply` in `build.py`. Add a test in `tests/unit/testsets/`. |

## Shipped recipes

- `recipes/t0-components.yaml`: the set used to choose the VAD, tagger, LID and local ASR.
  English and Russian speech, 13 conditions with speech and 4 without, about 1,500 clips.
- `recipes/t0-long.yaml`: long LibriSpeech clips (30 s to 5 min) under 5 conditions.

Revisions marked `PIN` must be replaced with commit shas before building.

## Licenses

Audio is never committed; only recipes are. The licenses of what you build come from its
sources and are listed in `sources.json`. Check them before you share a built set.
