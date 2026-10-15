# Contributing

Thank you for looking. This document is short on ceremony and long on the
handful of rules that will actually block a pull request, because those are the
ones nobody can guess from reading the code.

## Before anything else

```bash
git clone https://github.com/ajaysharmadeveloper/transcriptor-mcp-public
cd transcriptor-mcp-public
make install          # uv sync --all-groups, into the repo-root .venv
make test             # import contracts, then 465 tests
```

`make test` needs **no inference engine, no model weights, no GPU and no
database service**. If it
does not pass on a clean clone, that is a bug in the project rather than in your
setup, and an issue about it is welcome.

`make doctor` reports what your machine can actually run: OS, architecture,
selected engine, device, ffmpeg, memory pressure, database and migrations.

## The five rules that will fail your PR

Everything else is negotiable. These are not, and each is enforced by something
other than a reviewer's attention.

**1. The dependency rule.** `domain` imports nothing from the project.
`services` imports `domain` and `ports`, never `adapters`. `adapters` imports
`domain` and `ports`, never `services`. Only `server.py` imports adapters.
Nine `import-linter` contracts check this and they run *before* the tests,
because a dependency violation is architectural — it should fail the build
before a behavioural test gets a chance to pass.

**2. Nothing over ~5ms runs on the event loop.** Inference and VAD go to
`ThreadPoolExecutor(max_workers=1)`. ffmpeg and yt-dlp go to
`asyncio.create_subprocess_exec`, never `subprocess.run`. Hashing goes to the
cpu executor.

**3. `ASREngine.transcribe` is synchronous, deliberately.** Do not make it
async. The sync signature is what stops it being called on the loop. Its only
legal caller is the inference executor.

**4. Tool handlers touch the database and nothing else.** No engine, no
filesystem, no subprocess. They must answer in under 300ms while a job is
running.

**5. The queue is the `jobs` table.** No in-memory queue, no Redis, no Celery.
`status = 'queued'` *is* the queue.

## Ask before deciding

Open an issue first for: a tool name or signature change, a schema change, a new
dependency, or anything that bends a rule above. These are cheap to discuss and
expensive to undo.

## Tests

Write the test that **fails without your change**. This project has been bitten
twice by tests that asserted a consequence instead of causing it — a viewer
bind-failure test that set the error flag by hand and passed green while the
real path killed the server, and a fake engine that could not express the bug it
was meant to catch. A test that cannot fail is worse than no test, because it
looks like coverage.

Three specific things worth knowing:

- **Fixtures must build the same object graph `server.py` does.** A suite that
  wraps the plain store while production wraps the decorator cannot catch a
  decorator bug. That exact mismatch killed a live job.
- **If two configurations produce identical output, suspect the harness.** An
  early benchmark never passed the differing argument, so both runs were the
  baseline and came back byte-identical.
- **Single runs prove little.** Whisper's temperature fallback is stochastic;
  the same configuration gave 81 and 46 full stops on two runs. Variance can
  exceed the effect you are chasing.

Platform-specific tests are marked to run on the OS they describe. CI runs the
suite on Ubuntu, macOS and Windows, so each of them runs somewhere on every
push.

## Adding an engine

This is the most likely substantial contribution, and it is designed to be a
one-file job: implement the `ASREngine` protocol in `adapters/`, add a branch to
`_build_engine` in `server.py`, and add the library to the "inference libraries
stay behind the engine adapters" contract in `pyproject.toml`.

Two things carry over and are not optional:

- `condition_on_previous_text=False`, or the equivalent. Measured at ~4x
  throughput and 140x less repetition. See BENCHMARKS.md.
- Never pin `temperature=0`. It looks like a 1.78x speedup and removes the
  fallback ladder that detects degenerate decoding — it produced 503 words from
  120 seconds of audio, which is not physically plausible.

`engine.name` must be the real model id, because it lands in `result_key`. A
transcript produced by a different engine is a different artefact and must never
be served as a cache hit for another.

## Documentation is part of the change

Tool descriptions are behaviour, not comments — the model reads them to decide
what to do. A stale one is worse than a stale comment: `transcript_create` once
claimed URLs were unsupported long after they worked, and the model duly refused
valid work and told users something false. Change the description in the same
commit as the capability.

The same applies to a capability a response never mentions. A tool that returns
a download link users cannot see does not have that feature as far as any model
is concerned.

## Commit messages

Say what changed and why, in prose. If a measurement drove the change, give the
number. The commit log is the closest thing this project has to a design
history, and it is used that way.

## Licence

By contributing you agree that your contributions are licensed under the MIT
Licence, the same as the rest of the project.
