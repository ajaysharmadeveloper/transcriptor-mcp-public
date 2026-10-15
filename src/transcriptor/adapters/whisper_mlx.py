"""WhisperMLXEngine — Whisper large-v3 8-bit via ``mlx-whisper``.

The only module in the project that may import ``mlx``; an import-linter
contract enforces that, which is what makes swapping the engine (driver D7) a
one-file change rather than a rewrite.

``name`` is the model id, because ``name`` is what lands in ``result_key`` — a
transcript produced by a different build is a different artefact.

Verified against the installed MLX on 2026-09-09, because project-details §18
warns this API moved namespace between releases:

* ``set_memory_limit``, ``set_cache_limit``, ``set_wired_limit``,
  ``clear_cache``, ``get_active_memory``, ``get_peak_memory``,
  ``get_cache_memory``, ``reset_peak_memory`` are all **top level** on
  ``mlx.core`` — *not* under ``mlx.core.metal``, where older guides put them.
* ``mlx_whisper.transcribe(audio, path_or_hf_repo=..., word_timestamps=...,
  **decode_options)`` — ``language`` is passed through ``decode_options``.
* ``mlx_whisper.transcribe.ModelHolder`` is a module-level singleton cache, so
  warming it in :meth:`load` is exactly what :meth:`transcribe` reuses.

Installed: mlx-whisper 0.4.3, Metal available. Check these again on any MLX bump
rather than trusting this comment.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from pathlib import Path

from transcriptor.adapters._hf_cache import resolve_snapshot
from transcriptor.domain.errors import EngineError
from transcriptor.domain.transcript import AudioSlice, Segment, Word

log = logging.getLogger(__name__)

# How much *available system memory* a job with this build consumes — which is
# what AdmissionController compares against, not what the process occupies.
#
# The two differ and it matters. Measured for large-v3 8-bit on an M4/16GB:
#   process occupancy, peak      3.04 GB   (bounded by MEM_SELF_CAP_GB)
#   available memory consumed    ~1.95 GB   (checked against MEM_SYSTEM_FLOOR_GB)
# Occupancy is higher because part of the model is shared with the page cache.
# 2.5 sits above the observed consumption with margin, without demanding the
# full occupancy figure before a job may start.
MEMORY_ESTIMATE_GB = {
    "mlx-community/whisper-large-v3-mlx-8bit": 2.5,
    "mlx-community/whisper-large-v3-mlx": 6.5,
}
DEFAULT_MEMORY_ESTIMATE_GB = 2.5

_GB = 1024**3


class WhisperMLXEngine:
    """Implements :class:`transcriptor.ports.engine.ASREngine`."""

    #: Which engine this is, for `transcript_health`. `name` is the model id
    #: and answers a different question.
    kind = "mlx"
    #: MLX is Metal or nothing.
    device = "metal"

    def __init__(
        self,
        *,
        model: str,
        models_dir: Path,
        executor: Executor | None = None,
        self_cap_gb: float = 4.0,
        word_timestamps: bool = False,
    ) -> None:
        self.name = model
        self.memory_estimate_gb = MEMORY_ESTIMATE_GB.get(
            model, DEFAULT_MEMORY_ESTIMATE_GB
        )
        self._models_dir = models_dir
        # The single-worker inference executor. Loading happens on the same
        # thread that transcribes, because MLX is not safe to drive from several
        # threads for one model.
        self._executor = executor
        self._self_cap_gb = self_cap_gb
        # DTW-derived word timings roughly triple segment row size and cost real
        # time. Phase 1 does not expose them, so they stay off.
        self._word_timestamps = word_timestamps
        self._detected_language: str | None = None
        self._resolved: Path | None = None

    # ------------------------------------------------------------ lifecycle

    @property
    def loaded(self) -> bool:
        from mlx_whisper.transcribe import ModelHolder

        return (
            ModelHolder.model is not None
            and self._resolved is not None
            and ModelHolder.model_path == str(self._resolved)
        )

    @property
    def detected_language(self) -> str | None:
        return self._detected_language

    async def load(self) -> None:
        """Bring the weights in, off the event loop.

        A cold load takes seconds; doing it on the loop would freeze every tool
        call for the duration, which is the failure this whole design exists to
        avoid.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_sync)

    async def unload(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._unload_sync)

    def _load_sync(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        path = self._resolve()

        # Bound MLX's own allocator rather than relying on noticing afterwards.
        # The cache limit is the buffer pool MLX reuses between calls; capping it
        # is what stops resident memory drifting upwards across chunks.
        try:
            mx.set_memory_limit(int(self._self_cap_gb * _GB))
            mx.set_cache_limit(int(min(self._self_cap_gb * 0.25, 1.0) * _GB))
        except Exception as exc:  # noqa: BLE001
            # Documented fallback: if the API moved again, fall back to
            # monitor-and-abort rather than refusing to run.
            log.warning("MLX memory limits unavailable", extra={"error": str(exc)})

        ModelHolder.get_model(str(path), mx.float16)
        log.info("model loaded", extra={"model": self.name, "path": str(path)})

    def _unload_sync(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        mx.clear_cache()

    def allocator_memory_gb(self) -> float:
        """What MLX has allocated, in GB.

        RSS under-reports Metal buffers, so the resource monitor asks the engine
        rather than trusting the process table. Zero when nothing is loaded, and
        zero if MLX is not importable at all — this must never be the reason a
        sample fails.
        """
        try:
            import mlx.core as mx

            return (mx.get_active_memory() + mx.get_cache_memory()) / _GB
        except Exception:  # noqa: BLE001
            return 0.0

    def clear_cache(self) -> None:
        """Drop MLX's buffer cache between chunks under pressure."""
        import mlx.core as mx

        mx.clear_cache()

    # ------------------------------------------------------------ inference

    def transcribe(
        self,
        audio: AudioSlice,
        language: str | None,
        *,
        prompt: str | None = None,
    ) -> list[Segment]:
        """Synchronous by design.  Its only legal caller is the inference
        executor (rule N3, invariant I5)."""
        import mlx_whisper
        from mlx_whisper.audio import SAMPLE_RATE, load_audio

        path = self._resolve()
        samples = load_audio(str(audio.path), sr=SAMPLE_RATE)

        # Slice to the requested window. Phase 1 has one chunk spanning the
        # whole file; phase 2 will pass real chunk boundaries here.
        start = int(max(audio.start_s, 0.0) * SAMPLE_RATE)
        stop = int(audio.end_s * SAMPLE_RATE) if audio.end_s is not None else len(samples)
        window = samples[start:min(stop, len(samples))]

        if window.size == 0:
            return []

        options: dict = {
            "fp16": True,
            # The single biggest throughput lever, measured at 3.3x on real
            # speech (BENCHMARKS.md).
            #
            # With it on, Whisper feeds its own decoded text back as context for
            # each successive 30s window inside a chunk. The context grows, every
            # window costs more than the last, and when the decode degrades the
            # model chases its own tail — on a 120s chunk that produced 36
            # segments and 297 words in 38.6s, against 18 segments and 254 words
            # in 11.7s with it off.
            #
            # Continuity across *chunk* seams does not depend on this: that is
            # what `initial_prompt` below is for, and it is measured separately.
            "condition_on_previous_text": False,
        }
        if language:
            options["language"] = language
        if prompt:
            # What the previous chunk ended with. Whisper conditions on it, so
            # the new chunk continues the sentence rather than restarting cold.
            options["initial_prompt"] = prompt

        try:
            result = mlx_whisper.transcribe(
                window,
                path_or_hf_repo=str(path),
                word_timestamps=self._word_timestamps,
                verbose=None,
                **options,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"{self.name} failed to transcribe: {exc}",
                model=self.name,
                start_s=audio.start_s,
            ) from exc

        self._detected_language = result.get("language") or language

        # Timings come back relative to the slice, so they are remapped onto the
        # original timeline here — the one place that offset is applied.
        offset = audio.start_s
        segments: list[Segment] = []
        for idx, raw in enumerate(result.get("segments", [])):
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            words = None
            if self._word_timestamps and raw.get("words"):
                words = [
                    Word(
                        start=w["start"] + offset,
                        end=w["end"] + offset,
                        text=(w.get("word") or "").strip(),
                        probability=w.get("probability"),
                    )
                    for w in raw["words"]
                ]
            segments.append(
                Segment(
                    idx=idx,
                    start=raw["start"] + offset,
                    end=raw["end"] + offset,
                    text=text,
                    words=words,
                )
            )
        return segments

    # ------------------------------------------------------------- internal

    def _resolve(self) -> Path:
        """The local snapshot directory for the model.

        Resolved from the on-disk cache rather than passing the repo id, so a
        run needs no network and a missing model is a named error instead of a
        surprise download mid-job.  Shared with the CTranslate2 engine, which
        needs precisely the same lookup — see ``adapters/_hf_cache``.
        """
        if self._resolved is None:
            self._resolved = resolve_snapshot(self._models_dir, self.name)
        return self._resolved
