-- The whole schema, in one migration.
--
-- This replaces five Postgres migrations. Collapsing them is safe exactly once
-- and this is the moment: no SQLite database of this project exists anywhere
-- yet, so there is no history to preserve. The Postgres files recorded real
-- decisions -- splitting requested_language from language, adding forced,
-- adding speech_timeline -- and those decisions survive here as columns. What
-- is lost is the order they arrived in, which no longer helps anyone.
--
-- Three columns from the Postgres schema are deliberately absent:
--   sources.first_seen_at    was never read by anything
--   sources.last_seen_at     was written on every repeat request, never read
--   sources.request_count    likewise
-- Audited before the port rather than carried across out of habit.
--
-- SQLite specifics that are load-bearing, not stylistic:
--   * Timestamps are TEXT holding ISO-8601 UTC with an offset. SQLite has no
--     timestamp type, and the alternative -- naive local strings -- turns the
--     stale-heartbeat reaper into a clock-dependent guess twice a year.
--     `UtcDateTime` in orm.py is the only thing that reads or writes them.
--   * UUIDs are TEXT. Storing them as BLOB would save 20 bytes a row and make
--     every hand-written query unreadable.
--   * Foreign keys need `PRAGMA foreign_keys = ON` per connection. SQLite
--     parses the constraints either way and silently ignores them without it,
--     which is the most expensive default in the whole engine. db.py sets it
--     on every connection.
--   * Every PRIMARY KEY column says NOT NULL explicitly. SQLite allows NULLs
--     in a primary key -- a documented, deliberately preserved bug from before
--     version 3.0 -- for every type except INTEGER PRIMARY KEY. Postgres does
--     not, so the models were written assuming it could not happen. The
--     schema-drift test caught the mismatch; without the explicit NOT NULL the
--     database would have been quietly laxer than the code believed.

-- ---------------------------------------------------------------- sources
-- One row per distinct video or file ever seen. Lets "have I transcribed this
-- before" be answered without scanning jobs, and accumulates probe metadata.
CREATE TABLE sources (
    id                  TEXT PRIMARY KEY NOT NULL,
    kind                TEXT NOT NULL CHECK (kind IN ('file', 'url')),
    raw                 TEXT NOT NULL,        -- exactly what was passed in
    source_key          TEXT NOT NULL UNIQUE, -- normalised URL, or normcased absolute path
    file_fingerprint    TEXT,                 -- local files: size + mtime, detects edits in place
    audio_hash          TEXT,                 -- blake2b over decoded PCM, filled after extraction
    title               TEXT,
    uploader            TEXT,
    platform            TEXT,                 -- youtube | instagram | tiktok | generic | local
    duration_s          REAL,
    has_manual_captions INTEGER,              -- SQLite booleans are 0/1
    has_auto_captions   INTEGER
);

CREATE INDEX sources_audio_hash_idx ON sources (audio_hash);
CREATE INDEX sources_platform_idx   ON sources (platform);

-- --------------------------------------------------------------- requests
-- The audit trail: what was asked for, when, by which tool, and what came back.
CREATE TABLE requests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    tool       TEXT NOT NULL,
    source_id  TEXT REFERENCES sources (id) ON DELETE SET NULL,
    job_id     TEXT REFERENCES jobs (id) ON DELETE SET NULL,
    params     TEXT NOT NULL,                 -- JSON, as received, minus nothing
    outcome    TEXT,                          -- ok | error code
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX requests_created_idx ON requests (created_at DESC);
CREATE INDEX requests_source_idx  ON requests (source_id, created_at DESC);

-- ------------------------------------------------------------------- jobs
CREATE TABLE jobs (
    id                 TEXT PRIMARY KEY NOT NULL,
    source_id          TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    result_key         TEXT NOT NULL,         -- what makes two results equivalent
    origin             TEXT,                  -- asr | captions_manual | captions_auto
    status             TEXT NOT NULL
        CHECK (status IN ('queued', 'running', 'paused', 'done',
                          'failed', 'cancelled', 'interrupted')),
    stage              TEXT,                  -- meaningful only while running
    resumable          INTEGER NOT NULL DEFAULT 0,
    forced             INTEGER NOT NULL DEFAULT 0,
    heartbeat_at       TEXT,                  -- refreshed every 10s while running
    progress_pct       REAL NOT NULL DEFAULT 0,
    partial            INTEGER NOT NULL DEFAULT 0,
    covered_until_s    REAL NOT NULL DEFAULT 0,
    last_chunk_idx     INTEGER NOT NULL DEFAULT -1,
    error              TEXT,
    error_code         TEXT,
    language           TEXT,                  -- detected
    requested_language TEXT,                  -- what the caller asked for, if anything
    model              TEXT,                  -- which engine build produced this
    duration_s         REAL,
    word_count         INTEGER,
    transcribe_ms      INTEGER,               -- feeds ETA estimates for later jobs
    speech_timeline    TEXT,                  -- JSON [[start, end], ...]; the Detect checkpoint
    created_at         TEXT NOT NULL,
    started_at         TEXT,
    finished_at        TEXT
);

CREATE INDEX jobs_source_idx ON jobs (source_id, created_at DESC);
CREATE INDEX jobs_status_idx ON jobs (status, created_at DESC);

-- At most ONE active job per result_key, enforced by the database rather than
-- by application logic. This is what makes a duplicate structurally impossible
-- instead of merely unlikely: check-then-act in application code races.
--
-- SQLite supports partial indexes, so invariant I1 keeps exactly the guarantee
-- it had on Postgres. Nothing about it is weaker here.
CREATE UNIQUE INDEX jobs_one_active_idx ON jobs (result_key)
    WHERE status IN ('queued', 'running', 'paused');

-- Newest completed result for a key, for the reuse lookup.
CREATE INDEX jobs_reuse_idx ON jobs (result_key, finished_at DESC)
    WHERE status = 'done';

-- ------------------------------------------------------------- job_events
-- Every status transition. Makes "how did this job get here" a query rather
-- than an investigation.
CREATE TABLE job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    job_id      TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    stage       TEXT,
    note        TEXT,
    at          TEXT NOT NULL
);

CREATE INDEX job_events_job_idx ON job_events (job_id, at);

-- ----------------------------------------------------------------- chunks
-- One row per unit of work within a job: precise resume, per-chunk timing,
-- retry counts.
CREATE TABLE chunks (
    job_id       TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    status       TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts     INTEGER NOT NULL DEFAULT 0,
    engine_ms    INTEGER,
    committed_at TEXT,
    PRIMARY KEY (job_id, idx)
);

-- ---------------------------------------------------------------- renders
-- A lazy cache. Always regenerable from segments, which is invariant I7 --
-- correcting a segment is one DELETE away from consistent.
CREATE TABLE renders (
    job_id       TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    format       TEXT NOT NULL CHECK (format IN ('txt', 'srt', 'vtt', 'json')),
    content      TEXT NOT NULL,
    bytes        INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, format)
);

-- --------------------------------------------------------------- segments
-- The transcript itself. Source of truth for every output format.
--
-- The composite FK into chunks is what enforces invariant I3: a segment cannot
-- exist without its chunk row, so `segments` can never contain rows whose chunk
-- is not `done` -- given that both are written in one transaction. It only
-- enforces anything with `PRAGMA foreign_keys = ON`, which db.py guarantees.
CREATE TABLE segments (
    job_id    TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    idx       INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    start_s   REAL NOT NULL,
    end_s     REAL NOT NULL,
    text      TEXT NOT NULL,
    words     TEXT,                           -- JSON; nullable, roughly triples row size
    PRIMARY KEY (job_id, idx),
    FOREIGN KEY (job_id, chunk_idx) REFERENCES chunks (job_id, idx) ON DELETE CASCADE
);

CREATE INDEX segments_time_idx ON segments (job_id, start_s);

-- ------------------------------------------------------------ full text
-- Two FTS5 indexes over the same rows, because they answer different questions
-- and Postgres needed two mechanisms for the same reason (tsvector, pg_trgm).
--
--   segments_fts   whole words. `unicode61 remove_diacritics 0` is the direct
--                  equivalent of the 'simple' text search configuration this
--                  used on Postgres: no stemming, because across 99 languages
--                  no-stemming is the only configuration that behaves
--                  consistently -- and crucially it does NOT strip diacritics,
--                  which would mangle every Hindi, Arabic and Vietnamese
--                  transcript in the index.
--
--   segments_trgm  substrings and near-misses, standing in for pg_trgm. This
--                  is where a half-remembered phrase gets found.
--
-- Both are external-content tables: the text lives once, in `segments`, and
-- these hold only the index. Triggers keep them in step, which is the
-- documented FTS5 pattern -- an external-content table does not update itself.
CREATE VIRTUAL TABLE segments_fts USING fts5(
    text,
    content = 'segments',
    content_rowid = 'rowid',
    tokenize = "unicode61 remove_diacritics 0"
);

CREATE VIRTUAL TABLE segments_trgm USING fts5(
    text,
    content = 'segments',
    content_rowid = 'rowid',
    tokenize = 'trigram'
);

CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts  (rowid, text) VALUES (new.rowid, new.text);
    INSERT INTO segments_trgm (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts  (segments_fts,  rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_trgm (segments_trgm, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts  (segments_fts,  rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_trgm (segments_trgm, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_fts  (rowid, text) VALUES (new.rowid, new.text);
    INSERT INTO segments_trgm (rowid, text) VALUES (new.rowid, new.text);
END;
