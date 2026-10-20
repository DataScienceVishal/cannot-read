from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cannotread.fetch import NotFetched, Pin, PinMismatch, fetch, stored_at, verified_bytes

REVISION = "0123abc"
CONTENT = b'{"version": "v2.0", "data": []}\n'


def _upstream(tmp_path: Path, content: bytes = CONTENT) -> Path:
    # Its file:// URL has the revision in the path, standing in for a raw GitHub
    # URL at a commit. urllib opens it without a socket.
    source = tmp_path / "upstream" / REVISION / "dev.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    return source


def _pin(url: str, content: bytes = CONTENT) -> Pin:
    return Pin(url, REVISION, hashlib.sha256(content).hexdigest())


def test_fetch_stores_the_bytes_under_their_digest(tmp_path: Path) -> None:
    pin = _pin(_upstream(tmp_path).as_uri())
    into = tmp_path / "fetched"
    assert fetch(pin, into) == CONTENT
    assert (into / pin.sha256).read_bytes() == CONTENT
    assert verified_bytes(pin, into) == CONTENT


def test_a_wrong_digest_writes_nothing(tmp_path: Path) -> None:
    pin = _pin(_upstream(tmp_path, b"what upstream serves today\n").as_uri())
    into = tmp_path / "fetched"
    with pytest.raises(PinMismatch, match="Nothing was written"):
        fetch(pin, into)
    # Not even the .partial file, and not the directory.
    assert not into.exists()


def test_one_changed_byte_on_disk_is_refused(tmp_path: Path) -> None:
    pin = _pin(_upstream(tmp_path).as_uri())
    into = tmp_path / "fetched"
    fetch(pin, into)
    path = stored_at(pin, into)
    edited = bytearray(CONTENT)
    edited[0] ^= 1
    path.write_bytes(bytes(edited))
    with pytest.raises(PinMismatch, match="changed on disk"):
        verified_bytes(pin, into)
    # fetch refuses it too, and leaves the edited file for someone to look at.
    with pytest.raises(PinMismatch):
        fetch(pin, into)
    assert path.read_bytes() == bytes(edited)


def test_a_verified_copy_is_not_downloaded_again(tmp_path: Path) -> None:
    source = _upstream(tmp_path)
    pin = _pin(source.as_uri())
    into = tmp_path / "fetched"
    fetch(pin, into)
    source.unlink()
    assert fetch(pin, into) == CONTENT


def test_a_write_killed_halfway_leaves_nothing_under_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = _pin(_upstream(tmp_path).as_uri())
    into = tmp_path / "fetched"
    real_write = Path.write_bytes

    def half_then_die(self: Path, content: bytes) -> int:
        real_write(self, content[: len(content) // 2])
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_bytes", half_then_die)
    with pytest.raises(KeyboardInterrupt):
        fetch(pin, into)
    assert not stored_at(pin, into).exists()
    # The leftover .partial does not stop the next run.
    monkeypatch.setattr(Path, "write_bytes", real_write)
    assert fetch(pin, into) == CONTENT


def test_reading_before_fetching_says_so(tmp_path: Path) -> None:
    pin = _pin(_upstream(tmp_path).as_uri())
    with pytest.raises(NotFetched):
        verified_bytes(pin, tmp_path / "fetched")


def test_a_branch_name_is_not_a_revision() -> None:
    digest = hashlib.sha256(CONTENT).hexdigest()
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    # "main0123abc" holds a valid short hash, so it fails only if the whole
    # string has to match.
    for revision in ("master", "", "a", "main0123abc"):
        with pytest.raises(ValueError, match="not a commit hash"):
            Pin(url, revision, digest)


def test_the_revision_has_to_be_a_path_segment_of_the_url() -> None:
    digest = hashlib.sha256(CONTENT).hexdigest()
    with pytest.raises(ValueError, match="as a path segment"):
        Pin("https://raw.githubusercontent.com/openai/human-eval/master/x", "6d43fb9", digest)
    # Inside a longer segment is not enough either.
    with pytest.raises(ValueError, match="as a path segment"):
        Pin(f"https://example.org/{REVISION}ff/x", REVISION, digest)
    Pin(f"https://example.org/bundles/0x{'ab' * 16}/contents/blob/", f"0x{'ab' * 16}", digest)


@pytest.mark.parametrize("digest", ["ABC" * 21 + "D", "a" * 63, "a" * 65, "g" * 64])
def test_a_digest_must_be_64_lowercase_hex(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase hex"):
        Pin(f"https://example.org/{REVISION}/f", REVISION, digest)
