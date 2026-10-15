# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — the database

- **Postgres is replaced by SQLite.** The database is now a file inside the
  project, `<data_dir>/transcriptor.db`, created on first run.
  `TRANSCRIPTOR_DATABASE_URL` is optional. There is no server to install, no
  DSN to write, and no Docker in the install path.
- The schema and every invariant survive unchanged: at most one active job per
  `result_key` is still a partial unique index, and segments still hang off
  chunks by composite foreign key. Four things had to be rebuilt rather than
  ported — `SKIP LOCKED`, `pg_trgm` similarity, the text-search configuration,
  and timestamp handling.
- Full-text search is FTS5, tokenised `unicode61 remove_diacritics 0`, plus a
  trigram index and a Python similarity pass. Misspelling tolerance is
  preserved; diacritics are no longer at risk of being stripped.
- Three unused columns removed after an audit: `sources.first_seen_at`,
  `sources.last_seen_at`, `sources.request_count`.
- **187 database tests no longer skip.** They needed a running Postgres and now
  need nothing, so they run on every CI runner and for every contributor. The
  separate Postgres CI job is gone.

**Upgrading from a Postgres install:**

```bash
uv sync --extra pgimport
transcriptor-mcp migrate
transcriptor-mcp import-postgres --from postgresql://user:pass@host:5432/transcriptor_db --dry-run
transcriptor-mcp import-postgres --from postgresql://user:pass@host:5432/transcriptor_db
```

It reads Postgres and never writes to it, so a failed run costs nothing and can
be repeated — every insert is `INSERT OR IGNORE`. Your old database is left
exactly as it was; drop it once you have checked the transcripts read back.

### Added

- **Windows and Linux support.** A second ASR engine, `FasterWhisperEngine`
  (CTranslate2), selected automatically on any machine MLX cannot run on. It
  uses CUDA where CTranslate2 reports a device and CPU otherwise. Apple Silicon
  is unchanged and still defaults to MLX.
- **VRAM governance.** `ResourceSample` now carries discrete-GPU memory, and
  the admission controller and pressure ladder respect a VRAM floor alongside
  the system floor. Inert on unified memory, where the GPU already spends the
  memory `psutil` reports.
- `TRANSCRIPTOR_ENGINE`, `TRANSCRIPTOR_DEVICE` and `TRANSCRIPTOR_COMPUTE_TYPE`
  settings. `TRANSCRIPTOR_MODEL` now defaults per engine rather than to a fixed
  MLX repo id.
- `cpu` and `cuda` install extras. The `mlx` extra is gated by an environment
  marker, so installing it off Apple Silicon resolves to nothing instead of
  failing.
- CI on Ubuntu, macOS and Windows, plus a Postgres job and a wheel-contents
  check.
- Open-source scaffolding: contributing guide, code of conduct, security
  policy, issue and pull request templates, Dependabot.

### Fixed

- **Absolute Windows paths were refused as relative.** The check used
  `PurePosixPath`, under which `C:\Users\me\talk.mp4` has no leading slash. Local
  files did not work on Windows at all.
- **Local-file dedupe on Windows.** Path keys now pass through
  `os.path.normcase`, so two spellings of one case-insensitive path are one
  identity. Without it a file could be transcribed twice with nothing reporting
  an error. No effect on POSIX, where existing keys are unchanged.
- **Cancellation leaked child processes on Windows.** `kill()` there stops only
  the named process, and yt-dlp starts ffmpeg to merge formats. Both subprocess
  adapters now share one tree-killing helper.
- `doctor` no longer hard-fails on any architecture that is not arm64. It asks
  whether the engine selected for this machine can run here, which is the
  question that was always meant.
- Child processes no longer flash a console window on Windows.

### Changed

- The `MLX stays behind the engine adapter` import contract is now
  `inference libraries stay behind the engine adapters`, and covers
  `faster_whisper` and `ctranslate2` as well.
- `transcriptor-mcp setup` warms whichever engine the machine selected, rather
  than assuming MLX.

### Not verified

The CUDA and Windows paths were written and unit-tested with the platform and
GPU layers stubbed, and CI runs the full suite on all three operating systems.
**Transcription itself has not been run on Windows or on an NVIDIA GPU.**
Reports from anyone who does are especially welcome.

## [0.1.0]

First working version. Apple Silicon only.

- MCP server over stdio and HTTP, thirteen tools.
- Whisper large-v3 8-bit via `mlx-whisper`.
- Postgres as the only durable store; the `jobs` table is the queue.
- Silero VAD with timestamp remapping, chunking with overlap stitching,
  per-chunk commits, and resume.
- Memory governance: self cap, system floor, and the escalation ladder.
- Three-layer dedupe, the captions shortcut, punctuation restoration,
  batch mode, clip finding, and a loopback download viewer.

[Unreleased]: https://github.com/ajaysharmadeveloper/transcriptor-mcp-public/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ajaysharmadeveloper/transcriptor-mcp-public/releases/tag/v0.1.0
