"""
artifacts.py — Trích xuất và lưu trữ file code từ output của SDLC agents.

Sau khi mỗi bước workflow hoàn thành, output markdown được quét để tìm:
  1. Explicit ### FILE: path/to/file.ext  (ưu tiên — định dạng có cấu trúc)
  2. Code block có dòng comment đầu chứa tên file (// filename: ...)
  3. Bold/heading text ngay trước code block (**src/Login.tsx**)
  4. Fallback: tự đặt tên theo ngôn ngữ + index (lang-01.ts)

Full raw output luôn được lưu dưới tên _output.md.

Layout thư mục:
  /data/artifacts/{workflow_id}/{role}/
    _output.md          ← luôn có: toàn bộ markdown output của agent
    src/Login.tsx       ← file code được trích xuất
    schema.sql          ← v.v.
"""
from __future__ import annotations

import logging
import os
import posixpath
import re
from pathlib import Path

logger = logging.getLogger("artifacts")

ARTIFACT_BASE: str = os.environ.get("ARTIFACT_DIR", "/data/artifacts")

# Chỉ trích xuất artifact cho các role này
ARTIFACT_ROLES: frozenset[str] = frozenset(
    {"fe", "mobile", "be", "dba", "da", "tech_lead", "devsecops"}
)

# language identifier → file extension (không có dấu chấm)
_LANG_EXT: dict[str, str] = {
    "typescript": "ts",   "ts": "ts",
    "tsx": "tsx",
    "javascript": "js",   "js": "js",
    "jsx": "jsx",
    "python": "py",       "py": "py",
    "sql": "sql",
    "yaml": "yml",        "yml": "yml",
    "json": "json",
    "dockerfile": "__dockerfile__",
    "bash": "sh",         "shell": "sh",   "sh": "sh",   "zsh": "sh",
    "markdown": "md",     "md": "md",
    "text": "txt",        "plaintext": "txt",
    "html": "html",       "css": "css",    "scss": "scss",
    "dart": "dart",       "kotlin": "kt",  "swift": "swift",
    "java": "java",       "go": "go",      "rust": "rs",  "rs": "rs",
    "toml": "toml",       "ini": "ini",    "xml": "xml",
    "hcl": "tf",          "terraform": "tf", "tf": "tf",
    "groovy": "groovy",   "properties": "properties",
    "nginx": "conf",      "conf": "conf",
    "env": "env",
}

# Khớp:  ### FILE: some/path/file.ext  rồi đến ```lang\ncontent\n```
_FILE_HDR_RE = re.compile(
    r"###\s+FILE:\s*([^\n]+?)\s*\n```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Khớp bất kỳ code block nào
_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

# Tên file từ dòng comment đầu tiên của code block
_CMT_FNAME_RE = re.compile(
    r"^[/#*\-]+\s*(?:file(?:name)?:\s*)?([a-zA-Z0-9_.][^\s]*\.[a-zA-Z0-9]{1,10})\s*$",
    re.IGNORECASE,
)

# Bold / heading ngay trước code block — ví dụ: **src/Login.tsx** hoặc ### src/Login.tsx
_PRE_FNAME_RE = re.compile(
    r"(?:\*{1,2}|`|#{1,4}\s+)([a-zA-Z0-9_.][^\s*`\n]*\.[a-zA-Z0-9]{1,10})`?\*{0,2}\s*:?\s*\n?\s*$"
)


def _sanitize_relpath(raw: str) -> str:
    """Làm sạch đường dẫn do model sinh ra: chuẩn hoá dấu phân cách, loại bỏ chuỗi traversal."""
    # Normalise Windows backslashes then collapse /../ via posixpath.normpath.
    normalized = posixpath.normpath(raw.replace("\\", "/"))
    # Remove any remaining leading slashes or .. components after normpath.
    parts = [p for p in normalized.split("/") if p and p != ".."]
    p = "/".join(parts)
    # Allow only safe characters (word chars, dot, dash, slash).
    p = re.sub(r"[^\w.\-/]", "_", p)
    return p or "file.txt"


def _ext_for_lang(lang: str) -> str:
    v = _LANG_EXT.get(lang.lower(), lang.lower() or "txt")
    return "Dockerfile" if v == "__dockerfile__" else (v or "txt")


def _default_name(lang: str, idx: int) -> str:
    ext = _ext_for_lang(lang)
    if ext == "Dockerfile":
        return "Dockerfile"
    prefix = lang[:6] if lang else "file"
    return f"{prefix}-{idx:02d}.{ext}"


# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save(role: str, output: str, workflow_id: str) -> list[dict]:
    """
    Trích xuất code block có tên từ *output*, ghi file, trả về danh sách metadata.
    Non-fatal: lỗi được log, trả về list rỗng (hoặc một phần) nếu thất bại.
    """
    if role not in ARTIFACT_ROLES:
        return []
    if not output or not output.strip():
        return []
    try:
        return _do_extract(role, output, workflow_id)
    except Exception as exc:
        logger.warning("Artifact extraction failed for %s: %s", role, exc)
        return []


def _do_extract(role: str, output: str, workflow_id: str) -> list[dict]:
    base = Path(ARTIFACT_BASE) / workflow_id / role
    base.mkdir(parents=True, exist_ok=True)
    base_resolved = base.resolve()  # compute once; passed to every _save call

    artifacts: list[dict] = []
    seen: set[str] = set()

    # ── Pass 1: explicit ### FILE: headers ──────────────────────────────────
    structured_spans: list[tuple[int, int]] = []
    for m in _FILE_HDR_RE.finditer(output):
        raw_path = m.group(1)
        lang     = m.group(2).lower() or "text"
        content  = m.group(3)
        if not content.strip():
            continue
        rel = _sanitize_relpath(raw_path)
        _save(base, base_resolved, role, rel, content, lang, artifacts, seen)
        structured_spans.append((m.start(), m.end()))

    def _in_structured(start: int) -> bool:
        return any(s <= start <= e for s, e in structured_spans)

    # ── Pass 2: free code blocks (fallback) ─────────────────────────
    counter = [0]
    for m in _BLOCK_RE.finditer(output):
        if _in_structured(m.start()):
            continue
        lang         = m.group(1) or "text"
        content      = m.group(2)
        if not content.strip():
            continue
        lines        = content.split("\n")
        filename     = None
        save_content = content

        # Thử tên file từ dòng comment đầu
        if lines:
            m2 = _CMT_FNAME_RE.match(lines[0].strip())
            if m2:
                filename     = m2.group(1).strip()
                save_content = "\n".join(lines[1:])

        # Thử tên file từ text ngay trước code block
        if not filename:
            pre = output[: m.start()].rstrip()
            m3  = _PRE_FNAME_RE.search(pre)
            if m3:
                filename = m3.group(1).strip()

        if not filename:
            counter[0] += 1
            filename = _default_name(lang, counter[0])

        rel = _sanitize_relpath(filename)
        _save(base, base_resolved, role, rel, save_content, lang, artifacts, seen)

    # ── Luôn lưu toàn bộ output raw dưới dạng _output.md ───────────
    md_path = base / "_output.md"
    md_path.write_text(output, encoding="utf-8")
    artifacts.insert(0, {
        "path": f"{role}/_output.md",
        "filename": "_output.md",
        "language": "markdown",
        "size": len(output.encode("utf-8")),
    })

    logger.info(
        "Artifacts — workflow=%s role=%s: %d files", workflow_id, role, len(artifacts)
    )
    return artifacts


def _save(
    base: Path,
    base_resolved: Path,
    role: str,
    rel: str,
    content: str,
    lang: str,
    artifacts: list,
    seen: set,
) -> None:
    if rel in seen:
        return
    seen.add(rel)
    dest = base / rel
    # Defense-in-depth: confirm the resolved path stays within base (guards against
    # any edge case that bypasses _sanitize_relpath).
    try:
        dest.resolve().relative_to(base_resolved)
    except ValueError:
        logger.warning("Path confinement blocked write outside base: %s", dest)
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        artifacts.append({
            "path": f"{role}/{rel}",
            "filename": rel,
            "language": lang,
            "size": len(content.encode("utf-8")),
        })
    except Exception as exc:
        logger.warning("Failed to write artifact %s: %s", dest, exc)


# ─────────────────────────────────────────────────────────────────────────────

def list_artifacts(workflow_id: str) -> dict[str, list[dict]]:
    """Quét thư mục artifact đã lưu, trả về {role: [file_meta]}."""
    wf_dir = Path(ARTIFACT_BASE) / workflow_id
    if not wf_dir.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for role_dir in sorted(wf_dir.iterdir()):
        if not role_dir.is_dir():
            continue
        files: list[dict] = []
        for f in sorted(role_dir.rglob("*")):
            if not f.is_file():
                continue
            rel  = f.relative_to(role_dir).as_posix()
            ext  = f.suffix.lstrip(".")
            files.append({
                "path": f"{role_dir.name}/{rel}",
                "filename": rel,
                "language": _LANG_EXT.get(ext, ext or "text"),
                "size": f.stat().st_size,
            })
        if files:
            result[role_dir.name] = files
    return result


def read_artifact(workflow_id: str, role: str, rel_path: str) -> tuple[str, str] | None:
    """
    Trả về (content, language) hoặc None nếu không tìm thấy.
    *rel_path* là đường dẫn tương đối từ thư mục role.
    """
    safe = _sanitize_relpath(rel_path)
    base = Path(ARTIFACT_BASE)
    path = base / workflow_id / role / safe
    # Ensure the resolved path stays within ARTIFACT_BASE to block traversal via
    # crafted workflow_id or role parameters.
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        logger.warning("Path confinement blocked read outside ARTIFACT_BASE: %s", path)
        return None
    if not path.exists() or not path.is_file():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    ext  = path.suffix.lstrip(".")
    lang = _LANG_EXT.get(ext, ext or "text")
    return content, lang
