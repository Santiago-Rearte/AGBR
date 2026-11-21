"""Unified-diff parsing and application (git-style patches, one or more files)."""

import re

_HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch:
    def __init__(self):
        self.old_path = None  # None for an added file
        self.new_path = None  # None for a deleted file
        self.hunks = []  # (old_start, old_len, new_start, new_len, [(tag, bytes)])
        self.binary = False
        self.mode_only = False

    @property
    def path(self):
        return self.new_path or self.old_path

    @property
    def added(self):
        return self.old_path is None

    @property
    def deleted(self):
        return self.new_path is None

    def __repr__(self):
        return f"FilePatch({self.old_path!r} -> {self.new_path!r}, {len(self.hunks)} hunks)"


def _strip(path):
    path = path.strip()
    if path.startswith(b'"') and path.endswith(b'"'):
        path = path[1:-1]
    if b"\t" in path:
        path = path.split(b"\t", 1)[0]
    if path == b"/dev/null":
        return None
    if path[:2] in (b"a/", b"b/"):
        path = path[2:]
    return path.decode("utf-8", "surrogateescape")


def parse(data: bytes):
    files = []
    cur = None
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(b"diff --git "):
            cur = FilePatch()
            files.append(cur)
            m = re.match(rb"diff --git (\S+) (\S+)", line)
            if m:
                cur.old_path = _strip(m.group(1))
                cur.new_path = _strip(m.group(2))
            i += 1
            continue
        if line.startswith(b"--- ") and i + 1 < n and lines[i + 1].startswith(b"+++ "):
            if cur is None or cur.hunks:
                cur = FilePatch()
                files.append(cur)
            cur.old_path = _strip(line[4:])
            cur.new_path = _strip(lines[i + 1][4:])
            i += 2
            continue
        if cur is not None and line.startswith(b"new file mode"):
            cur.old_path = None
        elif cur is not None and line.startswith(b"deleted file mode"):
            cur.new_path = None
        elif cur is not None and line.startswith(b"rename from "):
            cur.old_path = line[len(b"rename from ") :].decode("utf-8", "surrogateescape")
        elif cur is not None and line.startswith(b"rename to "):
            cur.new_path = line[len(b"rename to ") :].decode("utf-8", "surrogateescape")
        elif cur is not None and (line.startswith(b"Binary files") or line.startswith(b"GIT binary patch")):
            cur.binary = True
        elif cur is not None and line.startswith(b"@@"):
            m = _HUNK.match(line)
            if not m:
                i += 1
                continue
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            body = []
            i += 1
            o = nw = 0
            while i < n and (o < old_len or nw < new_len or (i < n and lines[i].startswith(b"\\"))):
                hl = lines[i]
                if hl.startswith(b"\\"):
                    if body:
                        body[-1] = (body[-1][0], body[-1][1], True)
                    i += 1
                    continue
                tag = hl[:1] or b" "
                text = hl[1:]
                if tag == b" ":
                    o += 1
                    nw += 1
                elif tag == b"-":
                    o += 1
                elif tag == b"+":
                    nw += 1
                else:
                    break
                body.append((tag, text, False))
                i += 1
            cur.hunks.append((old_start, old_len, new_start, new_len, body))
            continue
        i += 1
    for fp in files:
        if not fp.hunks and not fp.binary and fp.old_path and fp.new_path and fp.old_path == fp.new_path:
            fp.mode_only = True
    return files


class ApplyError(Exception):
    pass


def apply(old: bytes, fp: FilePatch) -> bytes:
    """Returns the new content. Raises ApplyError when the context does not match."""
    if fp.deleted:
        return b""
    old_lines = old.split(b"\n") if old else []
    # Track whether the old file ended with a newline.
    old_trailing_nl = old.endswith(b"\n")
    if old_trailing_nl:
        old_lines.pop()
    out = []
    pos = 0  # index into old_lines
    new_ends_without_nl = False
    for old_start, old_len, _new_start, _new_len, body in fp.hunks:
        start = old_start - 1 if old_len > 0 else old_start
        if start < pos or start > len(old_lines):
            raise ApplyError(f"hunk at {old_start} out of range")
        out.extend(old_lines[pos:start])
        pos = start
        for tag, text, no_nl in body:
            if tag == b" ":
                if pos >= len(old_lines) or old_lines[pos].rstrip(b"\r") != text.rstrip(b"\r"):
                    raise ApplyError(f"context mismatch at old line {pos + 1}")
                out.append(old_lines[pos])
                pos += 1
                if no_nl:
                    new_ends_without_nl = True
            elif tag == b"-":
                if pos >= len(old_lines) or old_lines[pos].rstrip(b"\r") != text.rstrip(b"\r"):
                    raise ApplyError(f"removed line mismatch at old line {pos + 1}")
                pos += 1
            else:
                out.append(text)
                new_ends_without_nl = no_nl
    out.extend(old_lines[pos:])
    if pos < len(old_lines):
        new_ends_without_nl = not old_trailing_nl
    elif not out:
        return b""
    text = b"\n".join(out)
    if not new_ends_without_nl:
        text += b"\n"
    return text


def changed_old_lines(fp: FilePatch):
    """Old-file line numbers that were removed, and old positions where lines were inserted.

    An insertion at position k means new lines appear between old lines k and k+1.
    """
    removed, inserted = set(), set()
    for old_start, old_len, _ns, _nl, body in fp.hunks:
        pos = old_start if old_len > 0 else old_start + 1
        for tag, _text, _ in body:
            if tag == b" ":
                pos += 1
            elif tag == b"-":
                removed.add(pos)
                pos += 1
            else:
                inserted.add(pos - 1)
    return removed, inserted
