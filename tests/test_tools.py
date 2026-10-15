"""The four phase 1 tools, driven through a real FastMCP client.

Not by calling the handler functions directly — through ``Client``, over the
in-memory transport, so schema generation, argument coercion and serialisation
are all exercised.  This is as close to Claude Desktop as a test gets.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastmcp import Client, FastMCP

from transcriptor import tools

pytestmark = pytest.mark.db

# The full set from project-details §4.
REGISTERED_TOOLS = {
    "transcript_probe",
    "transcript_create",
    "transcript_status",
    "transcript_get",
    "transcript_export",
    "transcript_search",
    "transcript_segments",
    "transcript_list",
    "transcript_cancel",
    "transcript_delete",
    "transcript_health",
    # Phase 4 addition, beyond the eleven in §4.
    "transcript_batch",
    "transcript_find_clips",
}


@pytest.fixture
def mcp(app):
    server = FastMCP(name="transcriptor-test")
    tools.register(
        server,
        tools.Deps(
            transcription=app.transcription,
            reads=app.reads,
            health=app.health,
            lifecycle=app.lifecycle,
            clips=app.clips,
            requests=None,
        ),
    )
    return server


def payload(result) -> dict:
    """FastMCP returns structured content; unwrap it the way a client would."""
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


# ------------------------------------------------------------------ registry


async def test_exactly_the_implemented_tools_are_registered(mcp) -> None:
    """Registering a tool early would advertise capability that does not
    exist."""
    async with Client(mcp) as client:
        listed = {t.name for t in await client.list_tools()}
    assert listed == REGISTERED_TOOLS


async def test_every_tool_is_described_for_a_model_to_choose_from(mcp) -> None:
    async with Client(mcp) as client:
        for tool in await client.list_tools():
            assert tool.description and len(tool.description) > 40, tool.name


# -------------------------------------------------------------------- create


async def test_create_returns_a_job_id_without_doing_the_work(mcp, wav) -> None:
    async with Client(mcp) as client:
        started = time.monotonic()
        out = payload(await client.call_tool("transcript_create", {"source": str(wav)}))
        elapsed_ms = (time.monotonic() - started) * 1000

    assert out["ok"] is True
    assert out["status"] == "queued", "no work has started"
    assert out["reused"] is False
    assert out["job_id"]
    # Rule N5. Generous versus the 250ms target so the assertion is about the
    # design, not about how loaded the machine happens to be.
    assert elapsed_ms < 250


async def test_create_accepts_a_url_and_normalises_it(mcp) -> None:
    async with Client(mcp) as client:
        out = payload(
            await client.call_tool(
                "transcript_create",
                {"source": "https://youtu.be/7ksp9N7vguE?si=jDRyqzNlBauV804x"},
            )
        )
    assert out["ok"] is True
    assert out["status"] == "queued"


async def test_create_refuses_a_playlist_with_a_named_error(mcp) -> None:
    async with Client(mcp) as client:
        out = payload(
            await client.call_tool(
                "transcript_create",
                {"source": "https://www.youtube.com/playlist?list=PL123"},
            )
        )
    assert out["ok"] is False
    assert out["error_code"] == "playlist_not_supported"


async def test_create_refuses_a_relative_path_and_explains_why(mcp) -> None:
    async with Client(mcp) as client:
        out = payload(
            await client.call_tool("transcript_create", {"source": "videos/ep12.mp4"})
        )
    assert out["error_code"] == "relative_path_not_allowed"
    assert "absolute" in out["error"]


async def test_create_reports_a_missing_file_with_the_exact_path(mcp, tmp_path) -> None:
    missing = tmp_path / "nope.mp4"
    async with Client(mcp) as client:
        out = payload(
            await client.call_tool("transcript_create", {"source": str(missing)})
        )
    assert out["error_code"] == "source_not_found"
    assert str(missing) in out["error"]


async def test_an_invalid_captions_value_is_rejected_not_guessed(mcp, wav) -> None:
    async with Client(mcp) as client:
        out = payload(
            await client.call_tool(
                "transcript_create", {"source": str(wav), "captions": "sometimes"}
            )
        )
    assert out["ok"] is False
    assert out["error_code"] == "invalid_parameter"


async def test_a_second_create_attaches_to_the_running_job(mcp, wav) -> None:
    """Layer 3 — two requests must not start two transcriptions."""
    async with Client(mcp) as client:
        first = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
        second = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
    assert second["job_id"] == first["job_id"]
    assert second["reused"] is True
    assert second["reuse_reason"] == "already_running"


# -------------------------------------------------------------------- status


async def test_status_reports_a_queued_job(mcp, wav) -> None:
    async with Client(mcp) as client:
        created = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
        out = payload(
            await client.call_tool(
                "transcript_status", {"job_id": created["job_id"]}
            )
        )
    assert out["status"] == "queued"
    assert out["progress_pct"] == 0.0
    assert out["error_code"] is None


async def test_status_on_an_unknown_job_says_so(mcp) -> None:
    async with Client(mcp) as client:
        out = payload(
            await client.call_tool("transcript_status", {"job_id": "not-a-uuid"})
        )
    assert out["ok"] is False
    assert out["error_code"] == "job_not_found"


# ----------------------------------------------------------------------- get


async def test_get_returns_srt_after_the_job_completes(mcp, app, wav) -> None:
    async with Client(mcp) as client:
        created = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
        await app.runner.tick()
        out = payload(
            await client.call_tool(
                "transcript_get",
                {"job_id": created["job_id"], "format": "srt", "mode": "full"},
            )
        )

    assert out["ok"] is True
    assert out["status"] == "done"
    assert out["partial"] is False
    assert "-->" in out["content"]
    assert out["word_count"] > 0


async def test_get_defaults_to_a_truncated_preview(mcp, app, wav) -> None:
    """A long transcript returned whole makes the tool actively harmful to the
    calling agent, so preview is the default."""
    async with Client(mcp) as client:
        created = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
        await app.runner.tick()
        out = payload(
            await client.call_tool(
                "transcript_get", {"job_id": created["job_id"], "max_chars": 4}
            )
        )
    assert out["truncated"] is True
    assert len(out["content"]) == 4
    assert out["total_chars"] > 4


async def test_get_rejects_an_unknown_format(mcp, app, wav) -> None:
    async with Client(mcp) as client:
        created = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
        out = payload(
            await client.call_tool(
                "transcript_get", {"job_id": created["job_id"], "format": "docx"}
            )
        )
    assert out["error_code"] == "unknown_format"


# -------------------------------------------------------------------- health


async def test_health_reports_real_memory_numbers(mcp) -> None:
    async with Client(mcp) as client:
        out = payload(await client.call_tool("transcript_health", {}))

    assert out["ok"] is True
    assert out["pressure"] in {"normal", "elevated", "critical"}
    assert out["system_available_gb"] > 0
    assert out["self_cap_gb"] == 4.0
    assert out["system_floor_gb"] == 2.5
    assert out["queue_depth"] == 0
    assert out["database_reachable"] is True
    assert out["degraded_tasks"] == []
    assert out["model_loaded"] is False


async def test_health_names_the_engine_and_device(mcp) -> None:
    """"Why is this so slow" is unanswerable without them: the same model on
    Metal, CUDA and a CPU differs by an order of magnitude, and a client that
    cannot see which one is running will promise the wrong thing."""
    async with Client(mcp) as client:
        out = payload(await client.call_tool("transcript_health", {}))

    assert out["engine"] == "fake"
    assert out["device"] == "none"


async def test_health_omits_vram_where_there_is_no_discrete_gpu(mcp) -> None:
    """Absent rather than zero.

    On unified memory the GPU spends the memory `system_available_gb` already
    accounts for, so a "vram_free_gb: 0" line would read as a machine out of
    graphics memory rather than one with no separate pool to report.
    """
    async with Client(mcp) as client:
        out = payload(await client.call_tool("transcript_health", {}))

    assert "vram_free_gb" not in out


async def test_health_counts_the_queue(mcp, wav) -> None:
    async with Client(mcp) as client:
        await client.call_tool("transcript_create", {"source": str(wav)})
        out = payload(await client.call_tool("transcript_health", {}))
    assert out["queue_depth"] == 1


async def test_health_surfaces_a_degraded_task(mcp, app) -> None:
    """A silently dead reaper is exactly the kind of failure that looks fine for
    a week and then does not."""
    app.tasks.mark_degraded("reaper", "RuntimeError: boom")
    async with Client(mcp) as client:
        out = payload(await client.call_tool("transcript_health", {}))
    assert out["degraded_tasks"] == ["reaper"]
    assert out["task_restarts"]["reaper"] == 1


# ------------------------------------------------------------ responsiveness


async def test_tools_stay_responsive_during_a_long_transcription(
    mcp, app, wav
) -> None:
    """Invariant I4, and the single most likely bug in this project.

    A slow engine runs in the executor while status is polled on the loop. If
    inference ever blocked the loop, these polls would queue behind it.
    """
    app.engine._delay_s = 1.0

    async with Client(mcp) as client:
        created = payload(
            await client.call_tool("transcript_create", {"source": str(wav)})
        )
        job = asyncio.create_task(app.runner.tick())
        await asyncio.sleep(0.15)  # let inference get going

        latencies = []
        while not job.done():
            started = time.monotonic()
            out = payload(
                await client.call_tool(
                    "transcript_status", {"job_id": created["job_id"]}
                )
            )
            latencies.append((time.monotonic() - started) * 1000)
            assert out["ok"] is True
            await asyncio.sleep(0.1)

        await job

    assert len(latencies) >= 3, "the job must actually have been slow"
    assert max(latencies) < 300, f"slowest status call was {max(latencies):.0f}ms"
