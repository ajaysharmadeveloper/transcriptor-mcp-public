## What this changes

<!-- One or two sentences. What was wrong, or what is now possible. -->

## Why

<!-- The reasoning, not the diff. If a measurement drove it, give the number. -->

## Checklist

- [ ] `make test` passes (import contracts run first, then the suite)
- [ ] New behaviour has a test that **fails without the change** — a test that
      asserts a consequence rather than causing it tests nothing
- [ ] No layer above `adapters` learned which library is underneath it
- [ ] Tool descriptions match what the tool now does (the model reads those;
      a stale one makes it refuse valid work)
- [ ] Any performance claim is measured, or is labelled as unmeasured

## If this touches the engine, chunking, or memory limits

BENCHMARKS.md records measurements that several obvious improvements
contradict. Say which number you are moving and what you measured.
