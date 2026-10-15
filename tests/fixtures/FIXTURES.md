# Test fixtures

Media is not committed — these are references to local paths and URLs used for acceptance runs.

## Phase 1 — local files only

| Fixture | Path | Properties |
|---|---|---|
| Real speech | `tests/fixtures/speech-sample.wav` | 20s, 16kHz mono, ~625KB |

**Not committed** — it is someone's media, and the repository should not carry
it. Six VAD tests need audio a real detector will recognise as speech, which
synthetic tones cannot provide, so without this file **those six skip rather
than fail**.

Make one from any video or audio with speech in it, and the tests light up:

```bash
ffmpeg -nostdin -y -i <any-recording-with-speech> \
  -vn -t 20 -ac 1 -ar 16000 -acodec pcm_s16le \
  tests/fixtures/speech-sample.wav
```

Twenty seconds at 16kHz mono is about 625KB — 0.2% of the size of the video it
was originally cut from, which is the whole reason a sample is used rather than
the source.

Phase 1 accepts **absolute local paths only**, no URLs, no chunking, 10-minute cap
(`TRANSCRIPTOR_PHASE1_MAX_DURATION_S=600`).

The phase 1 acceptance criterion names a *5-minute* MP4. The owner chose on 2026-09-09 to
accept the 77s fixture instead: it exercises every stage and is well under the 10-minute cap,
it just does not stress duration.

Synthetic audio is generated per test with ffmpeg (`sine=frequency=440`), so the fast suite
needs no committed media at all.

### VAD fixtures

| Fixture | How it is made | Must produce |
|---|---|---|
| `silence_wav` | `anullsrc`, 8s | zero speech regions |
| `music_bed_wav` | `sine=220` + tremolo, 8s | zero speech regions |
| `speech_wav` | first 20s of `test-video.mp4` | >10s of speech detected |
| `padded.wav` (built on demand) | 5s silence + 20s speech + 5s silence | regions `[[5.0, 25.2]]` |

`padded.wav` is the one that proves remapping — it is not kept, because it is
five seconds of work to rebuild:

```bash
ffmpeg -nostdin -y \
  -f lavfi -t 5 -i anullsrc=r=16000:cl=mono \
  -i tests/fixtures/speech-sample.wav \
  -f lavfi -t 5 -i anullsrc=r=16000:cl=mono \
  -filter_complex "[1:a]aresample=16000,aformat=channel_layouts=mono[sp];[0:a][sp][2:a]concat=n=3:v=0:a=1[out]" \
  -map "[out]" -ac 1 -ar 16000 -acodec pcm_s16le /tmp/padded.wav
```

Measured on it, 30s chunks / 5s overlap, real Silero and real Whisper:

| | first cue |
|---|---|
| VAD on | `[5.00-8.06] Same tool, same arguments, 5 turns in a row.` |
| VAD off (`TRANSCRIPTOR_VAD=off`) | `[0.00-11.94] same tool same arguments five turns in a row that's a react` |

Without VAD the timestamps are five seconds wrong, casing and punctuation are gone,
and segments smear to twelve seconds.

## Phase 2 — URL ingestion (yt-dlp), NOT phase 1

| Fixture | Source |
|---|---|
| YouTube | `https://youtu.be/7ksp9N7vguE?si=jDRyqzNlBauV804x` |

Verified end to end on 2026-09-09, real yt-dlp and real Whisper:

```
PROBE   title="89% of AI Agents Can Be Hijacked by One Document ..." uploader="AJAY SHARMA AI"
        duration_s=466.0  platform=youtube  has_auto_captions=True  auth_required=False
        source_key=https://www.youtube.com/watch?v=7ksp9N7vguE      <- ?si= stripped
        estimated_transcribe_s=79   (no download performed)
CREATE  resolving 1.0s -> downloading 2.1s -> vad 9.0s -> transcribing 11.0s -> done 145s
        932 words, 106 cues, lang=en
DEDUPE  the clean URL hit the same job: reused=True reason=exact_source
```

The suite itself never touches the network — `test_url_ingestion.py` drives a fake gateway,
because yt-dlp against a live site tests YouTube, not Transcriptor.

### Owner's reference video — keep this one

| | |
|---|---|
| URL | `https://youtu.be/rHLjxrbXmmY?si=2nMnkezKnIXppmSw` |
| Content | A content creator discussing the YouTube algorithm |
| Added | 2026-09-09, at the owner's request |

**Requested by the owner specifically and wanted for future work — do not drop it from this
file.** Long-form conversational speech about a single subject, which is the shape most of
the intended workload takes, and therefore a better quality reference than the short
fixtures above. Useful for: WER once a reference transcript exists, `transcript_find_clips`
against real long-form content, and seam quality over many chunks.

Its `?si=` parameter is a tracking token and must normalise away, exactly as the fixture
above does.
