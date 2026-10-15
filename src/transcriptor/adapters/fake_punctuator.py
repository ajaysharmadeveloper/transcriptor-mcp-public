"""FakePunctuator — a Punctuator for tests that are not about the model.

Deterministic: it terminates any text that does not already end in a terminator
and upper-cases the first letter. Enough to prove the stage rewrites segments,
drops renders and respects language support, with no 1GB download.
"""

from __future__ import annotations

from transcriptor.adapters.hf_punctuator import SUPPORTED_LANGUAGES, _capitalise


class FakePunctuator:
    """Implements :class:`transcriptor.ports.punctuator.Punctuator`."""

    def __init__(self, *, supported: frozenset[str] | None = None) -> None:
        self.name = "fake-punctuator"
        self.memory_estimate_gb = 0.0
        self._supported = supported if supported is not None else SUPPORTED_LANGUAGES
        self.calls: list[str] = []
        self.load_calls = 0
        self.unload_calls = 0

    def supports(self, language: str | None) -> bool:
        return bool(language) and language.split("-")[0].lower() in self._supported

    async def load(self) -> None:
        self.load_calls += 1

    async def unload(self) -> None:
        self.unload_calls += 1

    def restore(self, text: str) -> str:
        self.calls.append(text)
        stripped = text.strip()
        if not stripped:
            return text
        if stripped[-1] not in ".?!":
            stripped += "."
        return _capitalise(stripped)
