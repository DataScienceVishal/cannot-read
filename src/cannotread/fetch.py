"""Download a benchmark file at a pinned revision and refuse it if its SHA-256 is wrong.

The same rule as trail-scorer-audit's `upstream.py`, cut down to single files.
That module fetches one git commit and then checks two digests, one for the
scorer and one rolled up over the dataset JSONs. Here each file gets its own
pin: a URL with the revision in it, that revision, and the SHA-256 of the bytes.
docs/licences.md names the commits for LegalBench, BFCL and HumanEval and the
CodaLab bundle for SQuAD's scorer. Which host serves each file is not settled.
The assumption here is raw.githubusercontent.com at those commits, huggingface.co
at a revision for the LegalBench task files, and CodaLab by bundle id. BFCL's
scorer is the `bfcl_eval` package and HumanEval's is in `human_eval`, so each of
those needs one pin per file it imports. No adapter pins anything yet, so none
of this has been tried against the real hosts.

A fetched file is stored under its own digest, not its upstream name. Anyone
can then check the name with `shasum -a 256`, and LegalBench's `CUAD_` and
`cuad_` folders cannot collide on a case-insensitive disk, which they would if
the upstream paths were kept (docs/licences.md).

Nothing in the test suite fetches over the network. The tests use `file://`
URLs, which urllib opens without a socket.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_HEX64 = re.compile(r"[0-9a-f]{64}")

# A git commit, full or abbreviated to at least 7 characters, or a CodaLab
# bundle id. A branch or tag name matches neither, and huggingface.co
# revisions are git commits.
_REVISION = re.compile(r"[0-9a-f]{7,40}|0x[0-9a-f]{32}")

TIMEOUT_SECONDS = 60


class NotFetched(FileNotFoundError):
    """Nothing on disk for this pin yet. `fetch` writes it."""


class PinMismatch(ValueError):
    """Bytes that do not hash to the pinned digest, from the URL or on disk."""


@dataclass(frozen=True)
class Pin:
    url: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        # `main` in a URL could pass the digest today and fail the day the branch
        # moves, with the pin claiming a revision it was never fetched at. The
        # shape check refuses a branch name. The path segment check stops a short
        # revision from matching some other part of the URL by accident.
        if not _REVISION.fullmatch(self.revision):
            raise ValueError(f"{self.revision!r} is not a commit hash or a CodaLab bundle id")
        if f"/{self.revision}/" not in self.url:
            raise ValueError(f"{self.url} does not have {self.revision} as a path segment")
        if not _HEX64.fullmatch(self.sha256):
            raise ValueError(f"{self.sha256!r} is not a lowercase hex SHA-256")


def stored_at(pin: Pin, into: Path) -> Path:
    return into / pin.sha256


def verified_bytes(pin: Pin, into: Path) -> bytes:
    """The stored file's bytes, hashed in the same read that hands them back.

    One read, so the caller gets the bytes that were hashed, not whatever is on
    disk a moment after a check. A scorer has to be compiled from what this
    returns, never opened again by its path.
    """
    path = stored_at(pin, into)
    if not path.is_file():
        raise NotFetched(f"nothing at {path} for {pin.url}. Fetch it first")
    content = path.read_bytes()
    found = hashlib.sha256(content).hexdigest()
    if found != pin.sha256:
        raise PinMismatch(
            f"{path} hashes to {found}. It was stored as {pin.url} at {pin.revision} and has "
            f"changed on disk since. Delete it and fetch again."
        )
    return content


def fetch(pin: Pin, into: Path) -> bytes:
    """The pinned file's verified bytes, downloaded unless a verified copy is stored.

    It hands back bytes and not a path, because a path invites a second read
    that nothing hashes. A stored copy that fails its digest is refused, not
    overwritten, because a file edited on disk is something to find out about.
    Downloaded bytes that fail the digest are never written at all.
    """
    path = stored_at(pin, into)
    if path.exists():
        return verified_bytes(pin, into)
    with urllib.request.urlopen(pin.url, timeout=TIMEOUT_SECONDS) as response:
        content = response.read()
    found = hashlib.sha256(content).hexdigest()
    if found != pin.sha256:
        raise PinMismatch(
            f"{pin.url} served {len(content)} bytes hashing to {found}, not the pinned "
            f"{pin.sha256}. Nothing was written. If upstream moved on purpose, move the pin."
        )
    into.mkdir(parents=True, exist_ok=True)
    # Written beside the target and renamed, so a run killed mid-write leaves a
    # .partial file and never a file under the digest's name that fails it.
    partial = path.with_name(f"{path.name}.partial")
    partial.write_bytes(content)
    os.replace(partial, path)
    return content
