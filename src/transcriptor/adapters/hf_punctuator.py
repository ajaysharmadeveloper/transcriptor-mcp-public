"""HuggingFacePunctuator — token classification over the transcript text.

The model predicts a punctuation mark for each token and reports character
offsets into the input, so marks are inserted at exact positions rather than by
re-tokenising and rejoining — which mangles subwords ("easiest" came back as
"e, asi, est" on the first attempt).

Casing is not predicted by this model; the first letter after each terminator is
upper-cased here.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor

from transcriptor.domain.errors import InfraError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "kredor/punctuate-all"

# What `kredor/punctuate-all` was trained on. Transcriptor targets 99 languages
# and this covers twelve, so everything else is left untouched rather than
# guessed at.
SUPPORTED_LANGUAGES = frozenset(
    {"en", "de", "fr", "es", "bg", "it", "pl", "nl", "cs", "pt", "sk", "sl"}
)

# Labels the model emits. '0' means "no punctuation here".
_MARKS = frozenset(".,?:-")
# Already-punctuated positions are left alone, which is what makes this
# idempotent.
_EXISTING = frozenset(".,?!:;-")
_TERMINATORS = frozenset(".?!")

# Roughly 400 tokens; the model's window is 512, and a segment is far shorter.
_MAX_CHARS = 1200


class HuggingFacePunctuator:
    """Implements :class:`transcriptor.ports.punctuator.Punctuator`."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        executor: Executor | None = None,
    ) -> None:
        self.name = model
        self.memory_estimate_gb = 1.1
        self._executor = executor
        self._pipeline = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def supports(self, language: str | None) -> bool:
        if not language:
            # Language unknown: the model would still run, but on an unsupported
            # language it inserts marks confidently and wrongly.
            return False
        return language.split("-")[0].lower() in SUPPORTED_LANGUAGES

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_sync)

    async def unload(self) -> None:
        self._pipeline = None

    def _load_sync(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise InfraError(
                "transformers is not installed — run: uv sync --extra punctuate",
                model=self.name,
            ) from exc

        # "simple" merges subword tokens into groups and reports character
        # offsets, which is what makes exact insertion possible.
        self._pipeline = pipeline(
            "token-classification", model=self.name, aggregation_strategy="simple"
        )
        log.info("punctuator loaded", extra={"model": self.name})

    # Sync — runs in the inference executor.
    def restore(self, text: str) -> str:
        if not text.strip() or len(text) > _MAX_CHARS:
            return text
        self._load_sync()

        try:
            groups = self._pipeline(text)
        except Exception as exc:  # noqa: BLE001
            # A transcript that cannot be punctuated is still a good transcript.
            log.warning("punctuation failed, leaving text as is", extra={"error": str(exc)})
            return text

        return _capitalise(_insert(text, groups))


def _insert(text: str, groups) -> str:
    """Insert predicted marks at the character offsets the model reported."""
    out: list[str] = []
    last = 0
    for group in groups:
        mark = group.get("entity_group") or group.get("entity") or ""
        end = int(group.get("end", -1))
        if mark not in _MARKS or end <= last or end > len(text):
            continue
        if text[end - 1] in _EXISTING:
            # Already punctuated here — leave it, so this stays idempotent.
            continue
        out.append(text[last:end])
        out.append(mark)
        last = end
    out.append(text[last:])
    return "".join(out)


def _capitalise(text: str) -> str:
    """Upper-case the first letter of each sentence.

    Any non-space character closes the opportunity, not just a letter —
    otherwise "89% can be taken over" becomes "89% Can be taken over".
    """
    result: list[str] = []
    starting = True
    for ch in text:
        if starting and ch.isalpha():
            result.append(ch.upper())
            starting = False
        else:
            result.append(ch)
            if not ch.isspace():
                starting = False
        if ch in _TERMINATORS:
            starting = True
    return "".join(result)
