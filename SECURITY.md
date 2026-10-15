# Security

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/ajaysharmadeveloper/transcriptor-mcp-public/security/advisories/new),
not as a public issue.

Please include what an attacker could achieve, the steps to reproduce it, and
the affected version or commit. You will get an acknowledgement within a week.
This is a single-maintainer project, so a fix may take longer than that; you
will be told where it stands rather than left waiting.

## What this project's threat model actually is

Transcriptor runs on one machine, for one person, and keeps its data in a file
beside the code. It
has **no authentication of its own**. That is a deliberate constraint rather
than an omission, and it shapes what counts as a vulnerability here.

Things that are **in scope**:

- Anything that causes data to leave the machine that was not meant to. The
  premise of the project is that audio and transcripts stay local.
- Command injection through a filename, a URL, or a media title. Every child
  process is started with `create_subprocess_exec` and an argument list,
  never through a shell — a way around that is a real finding.
- The transcript viewer serving content to someone who should not have it. It
  binds loopback by default and its link token is derived per process.
- Credentials appearing in a log, an error message, or a tool response. The
  default database is a file with no password at all, but `masked_dsn` still
  guards any URL that carries one, and anything bypassing it is a bug.
- SQL injection anywhere a user-supplied string reaches a query.

Things that are **out of scope**, because they are the documented design:

- No authentication on the MCP server itself. It speaks stdio to a client
  already running as you.
- Setting `TRANSCRIPTOR_VIEWER_HOST` to something other than `127.0.0.1`
  exposes transcripts to your network. The README says so. That is a choice the
  operator makes.
- `TRANSCRIPTOR_YTDLP_REMOTE_COMPONENTS` downloads and runs a solver script
  from GitHub. It is off by default and the trade is stated where it is
  configured. Enabling it is opting into remote code.
- Anyone with an account on the machine can read the transcripts. Single-user
  is a stated constraint of the design.

## Handling secrets

**There is usually no secret here.** The database is a file inside the project,
protected by the permissions of the directory holding it, and nothing needs a
password to reach it. What is worth protecting is a cookies file, and any
database URL you set that does carry credentials — both belong in `<repo>/.env`,
which is gitignored, and on POSIX should be mode `0600`.

**On Windows, file mode is not a meaningful protection** — `chmod` there does
not do what it does on Unix, and the file is protected by NTFS ACLs inherited
from its directory instead. If the repository lives somewhere other users can
read, so can they. This is stated because a document claiming `0600` protects
you on Windows would be worse than saying nothing.

Never hardcode a credential, never commit one, never print one.

## Dependencies

Dependabot opens grouped monthly PRs. The inference engines are deliberately
excluded from the group: an MLX or CTranslate2 bump can move a memory API or a
decode default, and those changes need reading rather than merging.
