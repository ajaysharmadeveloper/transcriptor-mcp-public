# Transcriptor

**Local transcription MCP server. macOS, Linux and Windows.**

Point it at a file or a URL and it returns a job id; the transcript, its
segments and full-text search over them become available as the job completes.
Whisper large-v3, one SQLite file as the only durable store, stdio transport,
one native process.

**Nothing to set up but the code.** No database server, no Docker, no
connection string, no API key. Install it and transcribe something.

Nothing leaves the machine. No API keys, no per-minute billing, no upload of
client footage or unreleased video.

| | |
|---|---|
| **Status** | Phases 1–5 implemented. 465 tests, 9 architectural contracts. |
| **Platforms** | macOS (Apple Silicon), Linux, Windows. Python 3.12. |
| **Model** | Whisper large-v3 — 99 languages, MIT |
| **Database** | SQLite, inside the project. Created on first run |
| **Licence** | MIT, matching Whisper's |

### Which engine runs on your machine

The engine is chosen for you. Nothing above the adapter layer knows which one
it is talking to, so the tools, the pipeline and the database behave
identically either way.

| Your machine | Engine | Speed |
|---|---|---|
| **Apple Silicon** | `mlx-whisper` (Metal) | **Measured: ~6.9 min per hour of audio** |
| **NVIDIA GPU** (Linux, Windows) | `faster-whisper` (CTranslate2, CUDA) | Fast — see the note below |
| **Anything else** | `faster-whisper` (CTranslate2, CPU int8) | Roughly real-time |

> **What has actually been run.** Every performance figure in this repository
> was measured on Apple Silicon. The CUDA and Windows paths are unit-tested
> with the platform and GPU layers stubbed, and CI runs the full suite on
> Ubuntu, macOS and Windows — but **transcription itself has not been run on
> Windows or on an NVIDIA GPU by the author**, and this README will not pretend
> otherwise. If you run it there, an issue saying what happened is genuinely
> useful, working or not.

- **Measured performance:** [`BENCHMARKS.md`](BENCHMARKS.md)

---

## Install

You need **ffmpeg** and **uv**. That is all — the database comes with the
project. The only step that differs by platform is the first one and the
engine extra.

<details open>
<summary><b>macOS (Apple Silicon)</b></summary>

```bash
brew install ffmpeg uv
git clone https://github.com/ajaysharmadeveloper/transcriptor-mcp-public
cd transcriptor-mcp-public

uv sync --all-groups --extra mlx --extra vad --extra media --extra punctuate
```
</details>

<details>
<summary><b>Linux</b></summary>

```bash
sudo apt install ffmpeg          # or dnf / pacman / zypper
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ajaysharmadeveloper/transcriptor-mcp-public
cd transcriptor-mcp-public

# --extra cuda on a machine with an NVIDIA GPU, --extra cpu otherwise
uv sync --all-groups --extra cuda --extra vad --extra media --extra punctuate
```
</details>

<details>
<summary><b>Windows</b></summary>

```powershell
winget install Gyan.FFmpeg
winget install astral-sh.uv
git clone https://github.com/ajaysharmadeveloper/transcriptor-mcp-public
cd transcriptor-mcp-public

# --extra cuda on a machine with an NVIDIA GPU, --extra cpu otherwise
uv sync --all-groups --extra cuda --extra vad --extra media --extra punctuate
```

Restart your terminal after installing ffmpeg so it lands on `PATH`.
</details>

Then, on every platform:

```bash
transcriptor-mcp setup                     # ~1.6GB Whisper + ~2.1GB punctuation
transcriptor-mcp migrate                   # creates the database file
make doctor
```

That is the whole installation. There is no `.env` to write and no database to
create — `migrate` makes `<repo>/.transcriptor/transcriptor.db` and the schema
inside it. `.env.example` exists only for the settings you might want to
change later.

`make doctor` should print six green checks — it names your OS, architecture,
selected engine and device, and turns every "why isn't it working" into one
line of output.

### The extras are separable on purpose

| Extra | Brings | Without it |
|---|---|---|
| `mlx` | Apple Silicon inference (Metal) | No transcription on a Mac — everything else runs |
| `cuda` | CTranslate2 inference + VRAM reporting | — |
| `cpu` | CTranslate2 inference, CPU only | No transcription off a Mac — everything else runs |
| `vad` | Silero speech detection | Whisper sees silence and may hallucinate |
| `media` | yt-dlp | URLs are refused with a named error; local files fine |
| `punctuate` | Sentence restoration | Transcripts are correct, just less readable |
| `pgimport` | One-off import from an old Postgres install | Only needed once, by people upgrading |

Installing `mlx` on Linux or Windows is harmless — it carries an environment
marker and resolves to nothing there, so a copied command does not break.

The core install needs none of them. That is what lets the whole test suite run
with no model, no GPU and no network — and it means a missing extra **degrades
rather than fails**.

### Where things live

```
<repo>/.transcriptor/transcriptor.db   everything durable — the whole product
<repo>/.transcriptor/models/           model weights, ~3.7GB (HF_HOME points here)
<repo>/.transcriptor/tmp/              in-flight audio, wiped when a job ends
<repo>/.env                            optional settings, gitignored
```

Sources, requests, jobs, chunks, segments and renders all live in that one
file. Model weights are binaries that have to be files, and ffmpeg needs a real
path to read audio from; nothing else touches the disk.

### Upgrading from a Postgres install

Transcriptor used to need a Postgres server. If you have transcripts in one,
bring them across — it reads Postgres and never writes to it, so a failed run
costs nothing and can simply be repeated:

```bash
uv sync --extra pgimport
transcriptor-mcp migrate                                     # create the schema
transcriptor-mcp import-postgres --from "$OLD_DSN" --dry-run # see what would move
transcriptor-mcp import-postgres --from "$OLD_DSN"
```

Your old database is left exactly as it was. Drop it once you have checked the
transcripts read back.

**Back up `transcriptor.db`. It is the whole product**, and now that is a file
you can copy. Copy the `-wal` file alongside it, or run
`sqlite3 transcriptor.db ".backup out.db"` to get a consistent snapshot while
the server is running.

---

## Running it

It is a **stdio server**: the client launches it on demand and shuts it down
after. No port, no token, no daemon, no attack surface.

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "transcriptor": {
      "command": "/Users/ajaykumarsharma/Projects/transcriptor/.venv/bin/transcriptor-mcp"
    }
  }
}
```

Quit Claude Desktop completely (⌘Q) and reopen it.

### Claude Code

```bash
claude mcp add --scope user transcriptor \
  /Users/ajaykumarsharma/Projects/transcriptor/.venv/bin/transcriptor-mcp
```

User scope makes it available in every project, which is what you want — you
will usually transcribe something while working somewhere else. Verify with
`claude mcp get transcriptor`.

**There is no password to keep out of either config.** The database is a file
inside the project, reachable only by whoever can already read the directory.
Any settings you do set live in `.env`, which the server resolves from its own
install path rather than from a config that syncs and gets backed up.

### Both at once

They share one database and one model. `TRANSCRIPTOR_MAX_CONCURRENT` is `1`
deliberately — two simultaneous transcriptions on 16GB means swapping — so if
both start a job the second waits, and `transcript_status` reports
`queued_reason: max_concurrent`. That is the design working.

### Other transports and commands

```bash
transcriptor-mcp serve                      # stdio, the default
transcriptor-mcp serve --http --port 8765   # for an agent on another machine
transcriptor-mcp doctor                     # six-point preflight
transcriptor-mcp migrate                    # apply pending migrations
transcriptor-mcp setup                      # (re)download the models
transcriptor-mcp update-ytdlp               # sites break it; bump roughly monthly
```

HTTP binds to loopback unless told otherwise: Transcriptor has no
authentication of its own, so put it behind something that does before exposing
it.

Logs go to stderr — Claude Desktop captures them at
`~/Library/Logs/Claude/mcp-server-transcriptor.log`.

**After changing anything in the repo, restart the client.** Claude Desktop and
Claude Code each hold a running copy of the server, including its tool
descriptions — a stale one will refuse work it can actually do.

You do not need to configure `PATH`. A GUI-launched app cannot see Homebrew or
the venv, so the server puts both on `PATH` itself at startup.

### If a URL fails

| Error | What it means |
|---|---|
| `auth_required` | Login-walled, private, members-only or age-restricted. Set `TRANSCRIPTOR_COOKIES_FROM_BROWSER=chrome` in `.env` |
| `extractor_failed` mentioning a **JavaScript challenge** | YouTube's anti-bot check. Cookies will not help. Install Deno (`brew install deno`), or set `TRANSCRIPTOR_YTDLP_REMOTE_COMPONENTS=ejs:github` — which lets yt-dlp download and run a solver script from GitHub, so it is opt-in |
| `extractor_failed` otherwise | Sites break yt-dlp constantly: `transcriptor-mcp update-ytdlp` |
| `playlist_not_supported` | Pass one video URL, not a channel or playlist |

Downloading the audio yourself and passing the local path always works, whatever
the platform is doing.

---

## Configuration

Everything is a `TRANSCRIPTOR_*` environment variable, read from `<repo>/.env`
(mode `0600`, gitignored) or the client config. `.env.example` has the same list
with comments.

| Variable | Default | Purpose |
|---|---|---|
| **Engine** | | |
| `TRANSCRIPTOR_ENGINE` | `auto` | `auto` \| `mlx` \| `faster-whisper` \| `fake`. Auto picks MLX on Apple Silicon, faster-whisper elsewhere |
| `TRANSCRIPTOR_MODEL` | *(per engine)* | `mlx-community/whisper-large-v3-mlx-8bit` under MLX, `Systran/faster-whisper-large-v3` under faster-whisper. Neither engine can read the other's weights |
| `TRANSCRIPTOR_DEVICE` | `auto` | faster-whisper only. `auto` \| `cuda` \| `cpu` |
| `TRANSCRIPTOR_COMPUTE_TYPE` | *(per device)* | faster-whisper only. `float16` on CUDA, `int8` on CPU. Also `int8_float16`, `float32` |
| **Storage** | | |
| `TRANSCRIPTOR_DATA_DIR` | `<repo>/.transcriptor` | Model weights and scratch audio only |
| **Database** | | |
| `TRANSCRIPTOR_DATABASE_URL` | *(required)* | **Required.** Percent-encode `@ : / #` in the password |
| `TRANSCRIPTOR_DB_SCHEMA` | `public` | Set to share an existing database instead of a dedicated one |
| `TRANSCRIPTOR_DB_POOL_MIN` | `1` |  |
| `TRANSCRIPTOR_DB_POOL_MAX` | `4` |  |
| `TRANSCRIPTOR_DB_CONNECT_RETRY_S` | `120` | Startup wait — the client may launch before Docker is up |
| `TRANSCRIPTOR_SEGMENT_BUFFER_MAX` | `500` | Segments held in memory during a database outage |
| **Download viewer** | | |
| `TRANSCRIPTOR_VIEWER_ENABLED` | `true` | Serve finished transcripts as download links |
| `TRANSCRIPTOR_VIEWER_HOST` | `127.0.0.1` | Loopback. Change only if you understand what you are exposing |
| `TRANSCRIPTOR_VIEWER_PORT` | `8766` | Change if something else already has the port |
| **Memory** | | |
| `TRANSCRIPTOR_MEM_SELF_CAP_GB` | `4.0` | Max occupancy for Transcriptor. Measured peak is 3.04 GB (Apple Silicon) |
| `TRANSCRIPTOR_MEM_SYSTEM_FLOOR_GB` | `2.5` | Free memory left for everything else. Measured, not guessed |
| `TRANSCRIPTOR_MEM_POLL_S` | `5` |  |
| `TRANSCRIPTOR_MEM_PAUSE_MAX_S` | `60` | How long to wait under critical pressure before aborting |
| `TRANSCRIPTOR_IDLE_EVICT_S` | `600` | Unload the model after idle. Tightens under pressure |
| `TRANSCRIPTOR_MAX_CONCURRENT` | `1` | **Do not raise on 16 GB.** Two jobs means swapping |
| *VRAM floor* | `0.5` | Free GPU memory left for everything else. Applies only to a discrete GPU, and is **estimated rather than measured** — see `domain/pressure.py` |
| **Pipeline** | | |
| `TRANSCRIPTOR_CHUNK_S` | `120` | Chunk length; overlap is stitched away |
| `TRANSCRIPTOR_OVERLAP_S` | `15` |  |
| `TRANSCRIPTOR_PUNCTUATE` | `True` | Restore sentence boundaries after transcription |
| `TRANSCRIPTOR_PUNCTUATION_MODEL` | `kredor/punctuate-all` |  |
| `TRANSCRIPTOR_KEEP_SCRATCH` | `False` | Keep in-flight audio after a job, for debugging |
| `TRANSCRIPTOR_SCRATCH_TTL_DAYS` | `30.0` | Prune abandoned scratch. Active jobs are never touched |
| `TRANSCRIPTOR_PRUNE_INTERVAL_S` | `3600.0` |  |
| **Job lifecycle** | | |
| `TRANSCRIPTOR_HEARTBEAT_S` | `10` |  |
| `TRANSCRIPTOR_STALE_JOB_S` | `90` | Heartbeat age after which the reaper calls a job interrupted |
| `TRANSCRIPTOR_REAPER_S` | `60` |  |
| **Input** | | |
| `TRANSCRIPTOR_CAPTIONS_DEFAULT` | `manual_only` | `manual_only` | `never` | `prefer` |
| `TRANSCRIPTOR_MAX_DURATION_S` | `14400` | Refuse sources longer than this (4 hours) |
| `TRANSCRIPTOR_MAX_DOWNLOAD_MB` | `2048` |  |
| **yt-dlp** | | |
| `TRANSCRIPTOR_YTDLP_REMOTE_COMPONENTS` | *(unset)* | `ejs:github` to solve YouTube's JS challenge — downloads and runs remote code, so opt-in |
| `TRANSCRIPTOR_COOKIES_FROM_BROWSER` | *(unset)* | `chrome`, `safari`, `firefox`, `brave` — for login-walled sources |
| `TRANSCRIPTOR_COOKIES_FILE` | *(unset)* |  |

Defaults are chosen from measurement, not taste — see [`BENCHMARKS.md`](BENCHMARKS.md)
for how the memory and chunking numbers were arrived at. The one exception is
flagged where it appears: the **VRAM** figures are derived from model weight
sizes, because no NVIDIA hardware was available to measure them on. They are
deliberately generous, and replacing them with real numbers is a welcome first
contribution.

---

## Tools

| Tool | What it does | Destructive |
|---|---|---|
| `transcript_probe` | Inspect a source — duration, captions, whether you already have it. No download. | |
| `transcript_create` | Start a job, return a `job_id` immediately | |
| `transcript_batch` | Queue every media file in a folder | |
| `transcript_status` | State, stage, progress, ETA, why it is queued | |
| `transcript_get` | Transcript text in any format | |
| `transcript_export` | Write a format to a path you choose | |
| `transcript_search` | Find phrases with timestamps, fuzzy fallback | |
| `transcript_segments` | Pull a time window | |
| `transcript_list` | Recent jobs, filterable by status | |
| `transcript_cancel` | Stop a job | |
| `transcript_delete` | Remove a transcript | ⚠️ |
| `transcript_find_clips` | Self-contained high-density moments + ffmpeg commands | |
| `transcript_health` | Model state, real memory pressure, queue depth | |

### Talking to it

> *"Transcribe ~/Downloads/interview.mov and give me the SRT"*
> *"Probe https://youtu.be/… — how long is it, does it have captions?"*
> *"Search that transcript for where they talk about the algorithm"*
> *"Find me 5 clips from it, 20 to 45 seconds"*

`transcript_get` defaults to `mode="preview"` for a reason: a 2-hour transcript
is 40,000+ tokens and returning it whole makes the tool actively harmful to the
calling agent. The intended path into a long transcript is `transcript_search`
and `transcript_segments` — and, for the whole thing, the download link below.

### Getting the transcript out

A chat window is a poor place to receive 10,000 words. So a finished job also
reports links, served by the process that already holds the transcript:

```
"download_urls": {
    "txt":  "http://127.0.0.1:8766/t/<job_id>.txt?k=<token>",
    "srt":  "…/t/<job_id>.srt?k=<token>",
    "vtt":  "…/t/<job_id>.vtt?k=<token>",
    "json": "…/t/<job_id>.json?k=<token>"
},
"view_url": "http://127.0.0.1:8766/v/<job_id>?k=<token>"
```

Click a download link and the file lands in your Downloads folder, named after
the video rather than the job id. `view_url` opens a readable page: the
transcript, a **Copy all** button, and a download button for each of the four
formats — usually the only link worth handing over, since it leads to all the
others. Both
appear on `transcript_status` and `transcript_get` as soon as the job has
produced anything — including while it is still running.

Three things worth knowing:

- **Loopback only.** `127.0.0.1`, this machine, your browser. Nothing is
  uploaded and the links are not shareable. Change `TRANSCRIPTOR_VIEWER_HOST`
  only if you have thought about what you are exposing — the server has no
  authentication of its own beyond the link token.
- **The token is regenerated at every start**, so links do not outlive the
  process that made them. An old link returns 403; ask for a fresh one.
- **Nothing new is written to disk.** The file is rendered from the database when
  you click. `transcript_export` remains the only thing that writes a file, and
  only when you ask it to.

Turn it off with `TRANSCRIPTOR_VIEWER_ENABLED=false` if you would rather not
have a listening socket. If the port is taken the viewer disables itself, says
why in `transcript_health`, and the server carries on — a download link is a
convenience and losing it should never cost you a transcription.

**Claude Desktop starts two copies of the server**, and only one can hold the
port. The other checks who won: if it is another Transcriptor — same database,
same link token — it advertises the winner's links, so this is invisible in
normal use. You will see one line in the log saying so.

Only one number matters if you run two *different* Transcriptors against two
*different* databases: give the second one its own
`TRANSCRIPTOR_VIEWER_PORT`, or its links will point at the first one's
transcripts.

---

## Performance

Measured through the MCP wire — the same path a client uses — on a **busy**
machine (PyCharm, Docker and an agent session running). Full method and the
things that did *not* work are in [`BENCHMARKS.md`](BENCHMARKS.md).

**Every figure below is Apple Silicon** (M4, 16GB, `mlx-whisper`, large-v3
8-bit). None of it has been reproduced on CUDA or on CPU inference, so do not
read it as a cross-platform claim. The tool latencies are engine-independent
and should hold anywhere; the throughput and memory numbers are not.

| | Target | Measured |
|---|---|---|
| `transcript_create` | < 250 ms | **30–37 ms** |
| `transcript_status` during a job | < 300 ms | **21–99 ms** |
| Cache hit | < 50 ms | **14 ms** |
| Throughput, transcribe stage | < 8 min/hour | **6.7–7.4** |
| Throughput, end to end with punctuation | — | **9.3 min/hour** |
| Peak occupancy | < 3 GB, never above the cap | **3.04 GB** of a 4 GB cap |
| Swap growth | < 200 MB | **0** |

**Not yet measured: accuracy.** WER needs a labelled fixture set, and inventing
reference text would produce a number that means nothing. Recorded as a gap
rather than filled in.

### Memory, in practice

| State | Occupancy |
|---|---|
| Idle, model evicted | 0.12 GB |
| Transcribing | ~3.0 GB peak |
| Transcribing + punctuation restoration | 2.9 GB peak (ASR is evicted first) |

**To start a job you need about 5 GB free** — 2.5 GB for the model plus a 2.5 GB
floor left for everything else. Transcriptor is a guest on this machine, not its
owner: if the floor would be breached the job waits, and
`transcript_status` says `queued_reason: insufficient_memory`. Closing a browser
is usually enough.

**On a discrete GPU there are two pools, and both are checked.** Unified memory
lets one number govern everything, because the GPU spends the same memory
`psutil` reports. A separate card does not: its VRAM is invisible to the system
floor, which would happily read "healthy" while the GPU ran out. So a CUDA build
is admitted only if system memory *and* free VRAM both have room, and the
pressure ladder watches both. Those VRAM thresholds are estimates — see the note
under Configuration.

---

## What happens to your file

```
probe      metadata only — is it long, does it already have captions
fetch      yt-dlp pulls audio only; a 40-minute 4K video is ~30MB, not 2GB
extract    ffmpeg → 16kHz mono
detect     Silero finds the speech and drops the rest
transcribe 2-minute chunks, each committed as it finishes
punctuate  sentence boundaries restored
```

Three consequences worth knowing as a user:

- **You get a `job_id` in about 30ms** and can poll or walk away. Nothing waits.
- **A running job is already readable.** Because each chunk is committed as it
  completes, `transcript_get` and `transcript_search` work mid-job and report
  `partial: true`. If the machine dies, everything transcribed so far survives
  and the job resumes from its last committed chunk rather than starting over.
- **The same thing is never transcribed twice.** A shared YouTube link and a
  clean one are the same source; so is the same audio arriving from a different
  URL, or a file you already did. Asking again returns the existing result in
  milliseconds. Changing the model, the forced language or the caption policy
  correctly produces a fresh transcript, because that is a different artefact.

Nothing leaves your machine except the request to the site you asked it to
fetch.

## Layout

```
src/transcriptor/
  server.py     composition root — lifespan, supervision, CLI. Only importer of adapters.
  tools.py      MCP surface. Services only.
  services/     orchestration, transactions, policy.
  domain/       pure data + rules. stdlib and pydantic only.
  ports/        the contracts.
  adapters/     implementations — MLX or CTranslate2, Silero, yt-dlp, ffmpeg,
                SQLite, punctuation.
  migrations/   numbered .sql
```

`make lint` enforces the dependency rule with nine import-linter contracts, and
runs **before** the tests — a violated dependency rule is architectural, not a
bug. Verified to bite: injecting a `services → adapters` import breaks two
contracts.

---

## Documentation

| If you want to… | Read |
|---|---|
| Install, configure and use it | **this file** |
| Know how fast it is and how that was measured | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

---

## Development

```bash
make test      # import contracts + 465 tests, no engine or network needed
make lint      # the dependency rule alone
make doctor    # six-point preflight
make bench     # rerun the benchmarks (Apple Silicon)
make build     # wheel + sdist into dist/
```

The default suite runs with **no inference engine, no model weights, no GPU and
no database service** — that is what the `ASREngine` port and a file-backed
database buy between them. Tests needing real inference are marked `slow`.
Database tests get their own SQLite file in a temporary directory, so they
never touch your real one.

That last part is a recent and large improvement: 187 database tests — most of
the store, dedupe, resume and reaper coverage — used to skip for anyone without
a running Postgres. They now run everywhere.

CI runs the contracts and the whole suite on **Ubuntu, macOS and Windows** on
every push, plus a check that the migrations really are inside the built wheel.
Tests describing one platform's behaviour are marked to run on that platform, so
each runs somewhere on every push rather than being skipped everywhere.

Contributions are welcome — [`CONTRIBUTING.md`](CONTRIBUTING.md) covers the five
rules that will actually fail a PR, and what carries over if you add an engine.

### Releasing

```bash
make build
twine upload dist/*     # needs a PyPI token; deliberately not automated
```

The `.sql` migrations ship inside the wheel — the server refuses to start
without them, so they are not incidental data.

---

## Deliberately not built

- **Speaker diarization.** Evaluated and rejected: FluidAudio is a Swift package
  needing a helper binary and an IPC protocol; pyannote's models are gated on
  Hugging Face, which breaks both "no API keys" and offline install. Each fails
  a stated constraint. The port is written so this can be revisited without
  reshaping the pipeline.
- **Multi-user.** `requests` has no principal column. Single-machine,
  single-user is a constraint the design accepts in exchange for having no
  distributed-systems problems at all.
- **Real-time / microphone streaming**, translation, and a GUI.
- **A container image.** On macOS, Docker Desktop has no Metal and no MLX
  passthrough, so it would mean CPU inference at a tenth of the speed. On Linux
  with `--gpus all` the CTranslate2 engine would work, and a Dockerfile is now a
  reasonable contribution rather than a dead end — it simply has not been
  written or tested.
- **Accuracy numbers.** WER needs a labelled fixture set, and inventing
  reference text would produce a number that means nothing. Recorded as a gap
  in [`BENCHMARKS.md`](BENCHMARKS.md) rather than filled in.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) · [`SECURITY.md`](SECURITY.md) · [`CHANGELOG.md`](CHANGELOG.md)

The two contributions that would help most right now:

1. **Run it on Windows or on an NVIDIA GPU and say what happened.** Those paths
   are written and unit-tested but have never been executed by the author.
2. **Measure the VRAM figures.** They are currently derived from weight sizes
   and marked as estimates everywhere they appear.

## Licence

MIT, matching Whisper's. Whisper itself is MIT; `mlx-whisper`, `faster-whisper`
and Silero VAD are MIT; `yt-dlp` is Unlicense.
