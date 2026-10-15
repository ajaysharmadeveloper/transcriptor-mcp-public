"""Pipeline stages.

Each has ``satisfied(ctx)`` (an idempotency probe) and ``run(ctx)``.  Execution
is: skip if satisfied, else run.  Resume is not a special code path — it is the
same loop over a job whose early stages already report satisfied.

Phase 1 uses resolve / extract / transcribe / finalize.  fetch, vad and the
chunked form of transcribe land in phase 2.
"""
