"""Benchmark run for BENCHMARKS.md.

Real MLX, real audio, driven over the MCP wire — the same path Claude Desktop
uses, so the numbers describe the product rather than a Python function call.

    python scripts/benchmark.py

Rerun on every engine change and paste the results into BENCHMARKS.md.
"""
import asyncio, json, os, statistics, time
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
REPO = Path("/Users/ajaykumarsharma/Projects/transcriptor")
def out(*a): print(*a, flush=True)

async def bench(client, label, source, extra=None):
    t0 = time.monotonic()
    c = (await client.call_tool("transcript_create",
         {"source": source, "force": True, **(extra or {})})).structured_content
    create_ms = (time.monotonic()-t0)*1000
    if not c.get("ok"):
        out(f"{label}: FAILED {c}"); return None
    wall = time.monotonic(); lat = []; peak = 0.0; swap = 0.0
    stages = {}
    last_stage, last_t = None, wall
    while True:
        t0 = time.monotonic()
        s = (await client.call_tool("transcript_status", {"job_id": c["job_id"]})).structured_content
        lat.append((time.monotonic()-t0)*1000)
        h = (await client.call_tool("transcript_health", {})).structured_content
        peak = max(peak, h["self_rss_gb"]); swap = max(swap, h["swap_delta_gb"])
        if s["stage"] != last_stage:
            if last_stage: stages[last_stage] = stages.get(last_stage,0)+time.monotonic()-last_t
            last_stage, last_t = s["stage"], time.monotonic()
        if s["status"] in ("done","failed"): break
        if time.monotonic()-wall > 1800: out(f"{label}: TIMEOUT"); return None
        await asyncio.sleep(0.5)
    total = time.monotonic()-wall
    if last_stage: stages[last_stage] = stages.get(last_stage,0)+time.monotonic()-last_t
    return {
        "label": label, "status": s["status"], "duration_s": s["duration_s"],
        "wall_s": round(total,1), "create_ms": round(create_ms),
        "status_ms_max": round(max(lat)), "status_ms_mean": round(statistics.mean(lat)),
        "words": s["word_count"], "language": s["language"],
        "peak_rss_gb": round(peak,2), "peak_swap_delta_gb": round(swap,2),
        "min_per_hour": round(total/s["duration_s"]*60,1) if s["duration_s"] else None,
        "stages": {k: round(v,1) for k,v in stages.items()},
    }

async def main():
    env = {**os.environ, "TRANSCRIPTOR_MEM_SYSTEM_FLOOR_GB": "2.0"}
    t = StdioTransport(command=str(REPO/".venv/bin/transcriptor-mcp"), args=[], env=env, cwd=str(REPO))
    results = []
    async with Client(t) as client:
        for label, src, extra in [
            ("77s local MP4 (HEVC/AAC)", str(REPO/"local_video_test/test-video.mp4"), None),
            ("30s padded WAV (10s silence)", str(REPO/"local_video_test/padded.wav"), None),
            ("466s YouTube URL", "https://youtu.be/7ksp9N7vguE?si=jDRyqzNlBauV804x",
             {"captions": "never"}),
        ]:
            r = await bench(client, label, src, extra)
            if r: results.append(r); out(json.dumps(r))
        # cache hit
        t0=time.monotonic()
        c=(await client.call_tool("transcript_create",
           {"source": str(REPO/"local_video_test/test-video.mp4")})).structured_content
        out(json.dumps({"label":"cache hit (layer 1)","ms":round((time.monotonic()-t0)*1000),
                        "reused":c.get("reused"),"reason":c.get("reuse_reason")}))
        # search latency
        done=[r for r in results if r["status"]=="done"]
        if done:
            jid=(await client.call_tool("transcript_list",{"limit":1})).structured_content
    (REPO/"bench.json").write_text(json.dumps(results, indent=2))
asyncio.run(main())
