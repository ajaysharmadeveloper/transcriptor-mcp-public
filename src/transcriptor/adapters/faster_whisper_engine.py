"""FasterWhisperEngine — Whisper large-v3 via ``faster-whisper`` (CTranslate2).

The engine for every machine MLX cannot run on: Linux, Windows, and Intel Macs,
on a CUDA GPU where there is one and on the CPU where there is not.

This and ``whisper_mlx`` are the only modules allowed to import an inference
library; the import contract in ``pyproject.toml`` enforces it, which is what
keeps adding a *third* engine a one-file job rather than a rewrite.

``name`` is the model id, because ``name`` is what lands in ``result_key`` — so
a transcript decoded by CTranslate2 is never handed back as though MLX had
produced it, and vice versa.  That falls out of §7 rather than needing a rule
of its own.

Three things about this library are traps, and all three are load-bearing:

* **``transcribe`` returns a lazy generator.**  Nothing is decoded until it is
  iterated, so materialising it *here* — inside the synchronous method that
  only ever runs on the inference executor — is what keeps the work off the
  event loop.  Returning the generator would satisfy the type checker, pass
  every test, and violate N2 the first time a caller iterated it.
* **``vad_filter`` must be passed ``False`` explicitly.**  Silero has already
  cut this audio (§13) and ``Timeline.to_original`` will remap against what
  Silero produced.  A second VAD inside the decoder would shift every
  timestamp, and the transcript would look plausible and be wrong.
* **``temperature`` is left unset** so the fallback ladder survives.  §14
  measured what pinning it to zero costs: 503 words out of 120 seconds of
  audio, which is not a speed-up but a model looping.

Verified against the faster-whisper API on 2026-09-10.  The decode options are
the openai-whisper surface, so §14 and §15 transfer unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from concurrent.futures import Executor
from pathlib import Path

from transcriptor.adapters._hf_cache import resolve_snapshot
from transcriptor.domain.errors import EngineError, InfraError
from transcriptor.domain.transcript import AudioSlice, Segment, Word

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_GB = 1024**3

# What a job costs, per compute type.
#
# UNMEASURED, and the difference from the MLX table above matters: those
# numbers came from BENCHMARKS.md on real hardware, these are derived from
# weight sizes plus a margin for activations. They are deliberately generous —
# refusing a job that would have fitted is recoverable, an OOM mid-decode is
# not — and anyone with the hardware should replace them with measurements.
#
#   system_gb  available system memory a job consumes (checked against the
#              system floor, exactly as on macOS)
#   vram_gb    discrete GPU memory it wants (checked against the VRAM floor;
#              zero on CPU, where there is no second pool)
_ESTIMATES = {
    # large-v3 CTranslate2 weights are ~3.1GB in float16, ~1.6GB in int8.
    ("float16", True): (1.5, 4.5),
    ("int8_float16", True): (1.5, 3.0),
    ("int8", True): (1.5, 2.5),
    ("float32", True): (2.0, 7.0),
    # On CPU the weights are the process, so the whole cost lands on RAM.
    ("float16", False): (4.0, 0.0),
    ("int8", False): (2.5, 0.0),
    ("float32", False): (6.0, 0.0),
}
_DEFAULT_ESTIMATE = (2.5, 3.0)


def cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can actually use.

    CTranslate2's own probe, not torch's: CTranslate2 is what will run the
    model, and a torch that reports CUDA proves only that torch was built for
    it.  Zero when the library is missing entirely, so this is safe to call
    before anything is installed.
    """
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - a probe must never be the thing that fails
        return 0


def _vram_reading() -> tuple[float, float, float] | None:
    """``(used_gb, free_gb, total_gb)`` for the current CUDA device.

    Device-wide rather than per-process, deliberately: the VRAM floor protects
    room for *everything else* on the GPU — the compositor, another model, a
    browser decoding video — so what this process alone has allocated is the
    wrong question.

    Torch first because ``silero-vad`` already brings it in, NVML second for a
    machine with no torch, and ``None`` when neither can answer.  Never raises:
    an unreadable GPU means "govern on system memory alone", not "fail".
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (total - free) / _GB, free / _GB, total / _GB
    except Exception:  # noqa: BLE001
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / _GB, info.free / _GB, info.total / _GB
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return None

    return None


class FasterWhisperEngine:
    """Implements :class:`transcriptor.ports.engine.ASREngine`."""

    #: Which engine this is, for `transcript_health`. `device` is set per
    #: instance below, because unlike MLX this one has a choice.
    kind = "faster-whisper"

    def __init__(
        self,
        *,
        model: str,
        models_dir: Path,
        executor: Executor | None = None,
        device: str = "auto",
        compute_type: str | None = None,
        word_timestamps: bool = False,
        cpu_threads: int = 0,
    ) -> None:
        self.name = model
        self._models_dir = models_dir
        # The single-worker inference pool. CTranslate2 is thread-safe in ways
        # MLX is not, but residency still needs exactly one owner (I5) and the
        # executor is what provides it structurally rather than by convention.
        self._executor = executor
        self._cpu_threads = cpu_threads
        self._word_timestamps = word_timestamps

        self.device = self._choose_device(device)
        self.compute_type = compute_type or self._default_compute_type(self.device)

        system_gb, vram_gb = _ESTIMATES.get(
            (self.compute_type, self.device == "cuda"), _DEFAULT_ESTIMATE
        )
        self.memory_estimate_gb = system_gb
        # Read by the admission controller through the same duck-typed lookup
        # as `clear_cache`, so an engine without a GPU needs no opinion on it.
        self.vram_estimate_gb = vram_gb

        self._model = None
        self._detected_language: str | None = None
        self._resolved: Path | None = None

    # ------------------------------------------------------------ selection

    @staticmethod
    def _choose_device(requested: str) -> str:
        if requested != "auto":
            return requested
        return "cuda" if cuda_device_count() > 0 else "cpu"

    @staticmethod
    def _default_compute_type(device: str) -> str:
        # float16 on the GPU, int8 on the CPU. int8 on a CPU is not a
        # compromise so much as the only sensible choice — float16 there is
        # emulated and slower than the quantised path it was meant to beat.
        return "float16" if device == "cuda" else "int8"

    # ------------------------------------------------------------ lifecycle

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def detected_language(self) -> str | None:
        return self._detected_language

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_sync)

    async def unload(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._unload_sync)

    def _load_sync(self) -> None:
        from faster_whisper import WhisperModel

        path = self._resolve()
        try:
            self._model = WhisperModel(
                str(path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self._cpu_threads,
                # The weights are already on disk; this forbids reaching for
                # the network mid-job if they somehow are not.
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"{self.name} failed to load on {self.device}/"
                f"{self.compute_type}: {exc}",
                model=self.name,
            ) from exc

        log.info(
            "model loaded",
            extra={
                "model": self.name,
                "device": self.device,
                "compute_type": self.compute_type,
                "path": str(path),
            },
        )

    def _unload_sync(self) -> None:
        self._model = None
        self._release_gpu_memory()

    def vram_gb(self) -> tuple[float, float, float] | None:
        """Injected into the resource monitor so CUDA stays behind this
        adapter (N1).  ``None`` where there is no discrete GPU."""
        if self.device != "cuda":
            return None
        return _vram_reading()

    def clear_cache(self) -> None:
        """Drop what can be dropped between chunks under pressure.

        Weaker than the MLX equivalent, and worth being honest about:
        CTranslate2 manages its own allocator and exposes no cache-clearing
        call, so this releases torch's CUDA caching allocator and nothing
        more.  It is not a no-op — on a shared GPU that block can be
        substantial — but it will not shrink the model.
        """
        self._release_gpu_memory()

    def _release_gpu_memory(self) -> None:
        if self.device != "cuda":
            return
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort, never fatal
            pass

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
        if self._model is None:
            self._load_sync()

        window = _read_wav_window(audio.path, audio.start_s, audio.end_s)
        if window.size == 0:
            return []

        options: dict = {
            # §14: the single biggest throughput lever, ~4x. With it on,
            # Whisper feeds its own decoded text back as context for each
            # successive 30s window; the context grows, each window costs more
            # than the last, and a degraded decode makes the model chase its
            # own tail.
            "condition_on_previous_text": False,
            # §13 owns voice activity detection. The audio reaching this point
            # is already speech-only and the timings will be remapped through
            # Timeline.to_original, so a second VAD here would silently move
            # every cue.
            "vad_filter": False,
            "word_timestamps": self._word_timestamps,
        }
        if language:
            options["language"] = language
        if prompt:
            # §15: one sentence of what the previous chunk ended with. More
            # than that and the model echoes the prompt back into the output,
            # past the seam where stitching looks for it.
            options["initial_prompt"] = prompt

        try:
            raw_segments, info = self._model.transcribe(window, **options)
            # Materialised here on purpose — see the module docstring. This is
            # the line that keeps decoding inside the inference executor.
            decoded = list(raw_segments)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"{self.name} failed to transcribe: {exc}",
                model=self.name,
                start_s=audio.start_s,
            ) from exc

        self._detected_language = getattr(info, "language", None) or language

        # Timings come back relative to the slice, so they are remapped onto
        # the original timeline here — the one place that offset is applied.
        offset = audio.start_s
        segments: list[Segment] = []
        for idx, raw in enumerate(decoded):
            text = (raw.text or "").strip()
            if not text:
                continue
            words = None
            if self._word_timestamps and getattr(raw, "words", None):
                words = [
                    Word(
                        start=w.start + offset,
                        end=w.end + offset,
                        text=(w.word or "").strip(),
                        probability=getattr(w, "probability", None),
                    )
                    for w in raw.words
                ]
            segments.append(
                Segment(
                    idx=idx,
                    start=raw.start + offset,
                    end=raw.end + offset,
                    text=text,
                    words=words,
                )
            )
        return segments

    # ------------------------------------------------------------- internal

    def _resolve(self) -> Path:
        if self._resolved is None:
            self._resolved = resolve_snapshot(self._models_dir, self.name)
        return self._resolved


def _read_wav_window(path: Path, start_s: float, end_s: float | None):
    """The requested window of a 16kHz mono PCM file, as float32 in [-1, 1].

    Read directly rather than through ``faster_whisper.decode_audio``, for the
    reason §13 gives for not using ``silero_vad.read_audio``: the extract stage
    always produces 16kHz mono signed-16 PCM, so this is a few lines against a
    whole media-decoding dependency in the hot path.

    Seeking rather than reading the file and slicing it means a 4-hour source
    costs the same here as a 4-minute one.
    """
    import numpy as np

    with wave.open(str(path), "rb") as fh:
        if fh.getnchannels() != 1 or fh.getsampwidth() != 2:
            raise InfraError(
                f"expected 16-bit mono PCM, got {fh.getnchannels()} channel(s) "
                f"at {fh.getsampwidth() * 8}-bit",
                path=str(path),
            )
        if fh.getframerate() != SAMPLE_RATE:
            raise InfraError(
                f"expected {SAMPLE_RATE}Hz, got {fh.getframerate()}Hz",
                path=str(path),
            )
        total = fh.getnframes()
        start = min(max(int(start_s * SAMPLE_RATE), 0), total)
        stop = total if end_s is None else min(int(end_s * SAMPLE_RATE), total)
        if stop <= start:
            return np.empty(0, dtype=np.float32)
        fh.setpos(start)
        frames = fh.readframes(stop - start)

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
