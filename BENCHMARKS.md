# Benchmarks

Rerun this on **every engine change** — a model swap that improves accuracy and
halves throughput is a trade someone has to make deliberately.

> ## Scope: Apple Silicon only
>
> **Every measured figure in this file was taken on one machine: an Apple M4
> with `mlx-whisper`.** Transcriptor now also runs on Linux and Windows through
> `faster-whisper` (CTranslate2, CUDA or CPU), and **none of those paths has
> been benchmarked, or in fact executed at all.** No NVIDIA hardware and no
> Windows machine were available.
>
> Two consequences worth stating plainly:
>
> - The throughput and memory numbers below describe MLX. Do not quote them for
>   a CUDA or CPU build.
> - The **VRAM thresholds** shipped in `domain/pressure.py` are derived from
>   model weight sizes, not measured. They are marked as estimates in the code,
>   and in the README, and they should stay marked until
>   somebody with the hardware replaces them.
>
> Benchmarking the CUDA path is the single most useful contribution this file
> could receive. To do it without lying to yourself: clear the database first, measure through the MCP wire, keep
> fixtures out of the scratch directory, and never trust a single run.

## Method

```bash
uv sync --all-groups --extra mlx --extra vad --extra media
transcriptor-mcp setup
python scripts/benchmark.py            # drives the real stdio server over MCP
```

Every number below is measured **through the MCP wire protocol**, not by calling
Python functions — the same path Claude Desktop uses. Memory is read from
`transcript_health`, which reports the monitor's most recent sample, so peak RSS
is observed at the governor's poll interval rather than sampled inside the
process.

## Run of 2026-09-09

| | |
|---|---|
| Machine | Mac mini, Apple M4, 16GB unified memory |
| OS / Python | macOS (Darwin 25.6.0), Python 3.12.13 arm64 |
| Model | `mlx-community/whisper-large-v3-mlx-8bit` (1.59GB on disk) |
| mlx-whisper / silero-vad / yt-dlp | 0.4.3 / 6.2.1 / 2026.8.19 |
| Chunking | 120s chunks, 15s overlap |
| Machine state | **not quiet** — PyCharm (~2.3GB), Docker, and this session were running |
| Memory settings | self cap 4.0GB, system floor **2.0GB** for these runs (the shipped default is now 2.5GB) |

### Throughput

Measured before and after disabling `condition_on_previous_text` (see below).

Verified on the shipped defaults, no overrides, database cleared first so real
inference ran (peak occupancy 2.6-2.9GB confirms it):

| Workload | Audio | min/hour **before** | min/hour **after** |
|---|---|---|---|
| Local MP4, end to end | 77.2s | 10.9 | **8.3** |
| — transcribe stage alone | | ~9 | **7.4** |
| YouTube URL, end to end | 466.0s | 19.6 | **8.1** (9.3 with punctuation) |
| — transcribe stage alone | | 18.4 | **6.7** |
| — punctuation restoration | | — | ~8s (about 1 min/hr) |
| Layer-1 cache hit | — | 14 ms | 14 ms |

**The < 8 min/hour target is met for the transcribe stage on both**, at 7.4 and
6.7. The end-to-end figures include download, extraction and VAD.

Word counts are unchanged (210 -> 211, 932 -> 940) and repeated 5-grams are 0
and 1.

### The one change that mattered

`condition_on_previous_text` defaults to `True` in Whisper. With it on, the model
feeds its own decoded text back as context for each successive 30s window inside
a chunk: the context grows, every window costs more than the last, and when a
decode degrades the model chases its own tail.

Measured on the 466s file through the real chunk-and-stitch path, post-VAD:

| Config | Time | min/hour | Words | Repeated 5-grams | caps/100w | Full stops |
|---|---|---|---|---|---|---|
| 120s/15s `cond=True` | 211.3s | 27.2 | 1117 | **140** | 11.1 | 93 |
| **120s/15s `cond=False`** | **53.5s** | **6.9** | 940 | **1** | 8.7 | 63 |
| 60s/8s `cond=True` | 148.2s | 19.1 | 1162 | 178 | 6.8 | 70 |
| 30s/5s `cond=False` | 82.3s | 10.6 | 961 | 34 | 9.7 | 81 |
| 45s/6s `cond=False` | 62.3s | 8.0 | 1036 | 38 | 4.8 | 32 |

**~4x faster and 140x less repetition.** Those repeats are counted *after*
stitching, so they are genuine model looping rather than un-deduplicated
overlap — the extra ~180 words in the `cond=True` rows are duplicated text.

**The cost was fewer sentence boundaries — and it has since been paid back.**
Full stops dropped from 83 to 63 on the same 940 words, because only the first
30s window of each call is well segmented. A restoration stage now puts them
back, and overshoots the original:

| | `cond=True` (old slow path) | `cond=False` alone | `cond=False` + restoration |
|---|---|---|---|
| Full stops | 83 | 63 | **121** |
| Capitals | 115 | 82 | **142** |
| Repeated 5-grams | 140 | 1 | **1** |
| Words | 1117 | 940 | 940 |
| End to end | 19.6 min/hr | 8.1 min/hr | **9.3 min/hr** |

Restoration costs about 8 seconds on a 466s file — roughly 1 min/hour — and the
result is better punctuated than the path it replaced, at half the time.

### Two things that look like wins and are not

- **`temperature=0`** (pinning, rather than the default fallback ladder) appears
  to be a 1.78x speedup. It produced **503 words from 120 seconds** — about 4.2
  words per second, which is not physically plausible speech. Removing the
  temperature fallback removes the escape hatch that detects degenerate
  decoding, so it loops. A correctness hazard, rejected.
- **Smaller chunks.** The hypothesis was that at `chunk_s=30` every Whisper
  window would receive our seam prompt rather than only the first of each chunk,
  restoring punctuation. It buys back full stops (81 vs 63) but costs 54% more
  time and 34x the repetition — and the run-on passages are *identical* at 30s
  and 120s, so chunk size was never the cause.
- **A longer seam prompt.** Two or three sentences instead of one is slower
  (7.7-8.3 vs 6.7 min/hour), gains no punctuation, and re-introduces the prompt
  echo: 47-50 repeated 5-grams against 1.

### Stage breakdown

| Workload | resolve | download | extract | vad | transcribe |
|---|---|---|---|---|---|
| 77s local MP4 | 0.5s | — | <0.1s | 1.1s | 11.4s |
| 30s padded WAV | <0.1s | — | 0.5s | <0.1s | 3.2s |
| 466s YouTube | 2.6s | 4.8s | 1.0s | 1.1s | 142.5s |

### Responsiveness and memory

| Metric | Target | Measured |
|---|---|---|
| `transcript_create` | < 250 ms | **19–62 ms** |
| `transcript_status` during a job | < 300 ms every time | **max 157 ms**, mean 13–20 ms |
| Cache hit | < 50 ms | **14 ms** |
| Peak occupancy, 8-bit | < 3 GB, never above the cap | **3.04 GB** — over the 3 GB target, under the 4 GB cap |
| Swap growth during a job | < 200 MB | **0 MB** on all three |
| Model cold load | < 20 s | **1.1 s** warm, ~2 s cold |

## What these numbers say

> **Note on the memory figures.** The 1.19–1.32 GB reported by
> `transcript_health` during these runs was under-sampled twice over: the
> governor polled every 5s, and RSS does not fully account for MLX's Metal
> buffers. With the monitor asking the engine for its allocator reading and
> polling every 1s, the same 77s job peaks at **3.04 GB** — above the < 3 GB
> target, though still 24% under the 4 GB self cap. Two figures matter and they
> are not the same:
>
> | | Measured | Checked against |
> |---|---|---|
> | Process occupancy, peak | **3.04 GB** | `MEM_SELF_CAP_GB` (4.0) |
> | Available memory consumed | **~1.95 GB** | `MEM_SYSTEM_FLOOR_GB` (2.5) |
>
> Occupancy is higher because part of the model is shared with the page cache.
> Conflating them either refuses jobs that would fit, or lets the process grow
> past its cap unnoticed — which is what was happening before.

**Responsiveness and memory comfortably beat their targets.** Status never came
close to 300ms even while inference ran, peak RSS stayed under half the 3GB
budget, and swap never moved. The off-loop discipline and the memory guard are
doing their jobs.

**Throughput now meets its target** at 6.9 min/hour for the transcribe stage,
having been 18.4. The cause was never chunk overhead or model load — it was
`condition_on_previous_text`, and the fix costs some punctuation density.

**Things still worth trying, in order of expected value:**

- Measure accuracy. Everything above is throughput and readability; WER against
  reference transcripts is still unmeasured.
- Rerun on an idle machine. Every figure here was measured with PyCharm, Docker
  and an agent session running.
- Compare `large-v3-turbo` now that the decode path is no longer the bottleneck.

## Accuracy

Not yet measured against reference transcripts. WER needs a labelled fixture set
(project-details §17 names one: clean English, accented English, Hindi,
two-speaker, music bed, silence), and inventing reference text would produce a
number that means nothing. **Recording this gap rather than filling the table
with something unearned.**

What *is* verified, by test rather than by benchmark:

- Silence and a tremolo tone bed both yield **zero** speech regions — no
  hallucinated sentences over non-speech.
- A chunked transcript is 95.0% word-identical to the single-chunk transcript of
  the same audio, with **zero repeated 5-grams** across seams.
- Dropping every cached render and re-rendering is byte-identical.

## The engine field, as of this run

`large-v3` remains the pick, for the reason project-details §5 gives: nothing
else combines 99 languages, a permissive licence and mature Apple Silicon
tooling. Re-evaluate when any of these changes:

| Candidate | Languages | Why not, today |
|---|---|---|
| Parakeet TDT 0.6B v3 | 25 European | No Hindi, no CJK, no Arabic — disqualifying |
| Canary-Qwen 2.5B | English-focused | Needs an NVIDIA GPU |
| Qwen3-ASR 1.7B | 52 | Better on its 52, degrades on the long tail; thinner MLX tooling |
| `large-v3-turbo` | 99 | 6x faster, pruned decoder costs accuracy — **worth measuring given the throughput gap above** |

A swap is a one-file change (`adapters/whisper_mlx.py`) plus a rerun of this
document. That is what the `ASREngine` port is for.
