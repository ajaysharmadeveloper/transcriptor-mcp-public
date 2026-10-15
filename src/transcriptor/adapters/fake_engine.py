"""FakeEngine — a deterministic ASREngine for tests.

This is what the port is *for*: the entire application layer exercises the real
pipeline, runner, stages and tools with no MLX, no 1.8GB model, and no Apple
Silicon, so the suite runs anywhere.

It is a real adapter rather than a test helper because it implements a port, and
because ``make doctor`` and the smoke path can use it to prove the wiring works
before any weights exist.
"""

from __future__ import annotations

import threading
import time

from transcriptor.domain.errors import EngineError
from transcriptor.domain.transcript import AudioSlice, Segment, Word


class FakeEngine:
    """Implements :class:`transcriptor.ports.engine.ASREngine`.

    Emits one segment per ``segment_s`` of the slice, with text drawn from
    ``phrases``, so tests can assert exact timings and exact output.
    """

    kind = "fake"
    device = "none"

    def __init__(
        self,
        *,
        name: str = "fake-engine",
        memory_estimate_gb: float = 0.0,
        segment_s: float = 2.0,
        # Multi-word, mutually distinct, punctuated, and enough of them that a
        # test-length run never wraps.
        #
        # Distinctness matters twice over: an overlap has to carry at least
        # `min_match_tokens` shared words for stitch_overlap to anchor the seam,
        # and a phrase list that repeats would make a genuine seam duplicate
        # indistinguishable from the list simply cycling.
        #
        # Punctuation matters because a real engine emits it, and anything that
        # reasons about sentences — clip finding, the seam prompt — would be
        # silently untestable against a fake that never ends one.
        phrases: tuple[str, ...] = (
            "hello world from the fake engine.",
            "the quick brown fox jumps over.",
            "pack my box with five dozen jugs.",
            "sphinx of black quartz judge my vow.",
            "how vexingly quick daft zebras jump.",
            "the five boxing wizards jump lazily.",
            "jackdaws love my big sphinx of quartz.",
            "bright vixens jump dozy fowl quack.",
            "quick zephyrs blow vexing daft jim.",
            "waltz bad nymph for quick jigs vex.",
            "glib jocks quiz nymph to vex dwarf.",
            "crazy fredrick bought many very exquisite opals.",
            "we promptly judged antique ivory buckles.",
            "a mad boxer shot a quick gloved jab.",
            "jaded zombies acted quaintly but kept driving.",
            "my girl wove six dozen plaid jackets.",
        ),
        language: str = "en",
        delay_s: float = 0.0,
        fail_with: Exception | None = None,
        emit_words: bool = False,
    ) -> None:
        self.name = name
        self.memory_estimate_gb = memory_estimate_gb
        self._segment_s = segment_s
        self._phrases = phrases
        self._language = language
        self._delay_s = delay_s
        self._fail_with = fail_with
        self._emit_words = emit_words

        self._loaded = False
        self._detected_language: str | None = None

        # Recorded so a test can assert invariant I5 — exactly one thread ever
        # calls transcribe — without reaching into the executor.
        self.calling_threads: set[int] = set()
        self.transcribe_calls: list[AudioSlice] = []
        # What the stage carried across each seam, so a test can assert it.
        self.prompts: list[str | None] = []
        self.load_calls = 0
        self.unload_calls = 0

    # ------------------------------------------------------------ lifecycle

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    async def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def allocator_memory_gb(self) -> float:
        """No allocator of its own — the fake holds nothing."""
        return 0.0

    # ------------------------------------------------------------ inference

    @property
    def detected_language(self) -> str | None:
        return self._detected_language

    def transcribe(
        self,
        audio: AudioSlice,
        language: str | None,
        *,
        prompt: str | None = None,
    ) -> list[Segment]:
        """Synchronous, exactly like the real engine."""
        self.calling_threads.add(threading.get_ident())
        self.transcribe_calls.append(audio)
        self.prompts.append(prompt)

        if not self._loaded:
            raise EngineError("transcribe called before load", engine=self.name)
        if self._fail_with is not None:
            raise self._fail_with
        if self._delay_s:
            time.sleep(self._delay_s)

        self._detected_language = language or self._language

        start = audio.start_s
        end = audio.end_s if audio.end_s is not None else start + self._segment_s
        segments: list[Segment] = []
        idx = 0
        while start < end - 1e-9:
            stop = min(start + self._segment_s, end)
            # Keyed on absolute time, not on position within the slice, so the
            # same audio window always yields the same words. That is what makes
            # overlapping chunks produce real duplicates for stitch_overlap to
            # find — a slice-relative index would fabricate different text for
            # the same seconds and quietly make the stitching tests meaningless.
            phrase = int(start / self._segment_s)  # floor: round() is banker's
            text = self._phrases[phrase % len(self._phrases)]
            words = None
            if self._emit_words:
                pieces = text.split()
                step = (stop - start) / max(len(pieces), 1)
                words = [
                    Word(start + i * step, start + (i + 1) * step, w)
                    for i, w in enumerate(pieces)
                ]
            segments.append(Segment(idx=idx, start=start, end=stop, text=text, words=words))
            start = stop
            idx += 1
        return segments


# The name is what lands in `result_key`, so a transcript made by this engine can
# never be reused as though the real model had produced it.
