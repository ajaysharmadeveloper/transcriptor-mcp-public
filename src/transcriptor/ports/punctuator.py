"""Punctuator — put the sentence boundaries back.

Disabling `condition_on_previous_text` made transcription roughly four times
faster and all but eliminated repetition, at the cost of punctuation: only the
first 30s window of each Whisper call is well segmented, so later windows arrive
as run-ons.  Every word is correct; what is missing is the full stops.

That cannot be recovered from timing — the degraded segment boundaries cut
mid-sentence ("...so how does" | "one sentence do all that") — so a model is the
only honest route.

Sync, like the other model-backed ports: it runs in the inference executor, and
only after the ASR model has been evicted, because the two do not fit together
under the self cap.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Punctuator(Protocol):
    name: str
    memory_estimate_gb: float

    def supports(self, language: str | None) -> bool:
        """Whether this model covers the language.

        Transcriptor targets 99 languages and no punctuation model comes close,
        so the honest behaviour for the rest is to leave the transcript alone
        rather than corrupt it.
        """
        ...

    async def load(self) -> None: ...

    async def unload(self) -> None: ...

    # Sync — the inference executor is the only legal caller.
    def restore(self, text: str) -> str:
        """Return ``text`` with punctuation and sentence casing restored.

        Must be idempotent: text that is already punctuated comes back
        unchanged, so re-running after a crash cannot compound.
        """
        ...
