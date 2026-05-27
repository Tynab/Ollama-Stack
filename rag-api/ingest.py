"""
ingest.py — Pipeline nạp tài liệu vào Qdrant và Neo4j cho YAN RAG stack.

Quy trình xử lý
-----------------
  load_documents()   →  split_documents()  →  embed (OllamaEmbeddings)
  →  upsert (Qdrant)  →  save_graph (Neo4j, tùy chọn)

Idempotency
-----------
  make_point_id() sinh UUID5 tất định từ path + content hash.
  Chạy /ingest nhiều lần trên cùng file sẽ upsert đúng point_id — không tạo bản sao.

Metadata chunk
--------------
  Mỗi chunk Qdrant có payload: source_file, relative_path, file_type,
  chunk_index, content_hash, embedding_model, module, doc_type, chunk_type,
  language, status, created_at, page_content, project.

Graph enrichment (tùy chọn)
----------------------------
  Nếu GRAPH_ENABLED=true, mỗi chunk được ánh xạ vào Neo4j.
  Nếu GRAPH_ENTITY_EXTRACTION=true, LLM sẽ trích xuất thực thể và lưu quan hệ MENTIONS.
"""

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set")
    return value


LOG_LEVEL = _require_env("LOG_LEVEL")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rag-ingest")

RAW_DATA_DIR = _require_env("RAW_DATA_DIR")
OLLAMA_BASE_URL = _require_env("OLLAMA_BASE_URL")
QDRANT_URL = _require_env("QDRANT_URL")
COLLECTION_NAME = _require_env("COLLECTION_NAME")
EMBEDDING_MODEL = _require_env("EMBEDDING_MODEL")

CHUNK_SIZE = int(_require_env("CHUNK_SIZE"))
CHUNK_OVERLAP = int(_require_env("CHUNK_OVERLAP"))
UPSERT_BATCH_SIZE = int(_require_env("UPSERT_BATCH_SIZE"))

GRAPH_ENABLED: bool = os.environ.get("GRAPH_ENABLED", "true").lower() == "true"
GRAPH_ENTITY_EXTRACTION: bool = os.environ.get(
    "GRAPH_ENTITY_EXTRACTION", "false").lower() == "true"
CHAT_MODEL: str | None = os.environ.get("CHAT_MODEL")

_graph_module = None
if GRAPH_ENABLED:
    try:
        import graph as _graph_module  # type: ignore[import-not-found]
    except Exception as _graph_import_err:
        logger.warning(
            "Neo4j graph module unavailable, running Qdrant-only: %s", _graph_import_err)
        GRAPH_ENABLED = False


def get_collection_name(project: str) -> str:
    """Tạo tên Qdrant collection: {COLLECTION_NAME}__{project}."""
    return f"{COLLECTION_NAME}__{project}"


def get_projects(raw_data_dir: str = RAW_DATA_DIR) -> list[str]:
    """Trả về danh sách tên project đã sắp xếp (thư mục con cấp một trong raw_data_dir)."""
    root = Path(raw_data_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".log",
    ".json",
    ".jsonl",
    ".yml",
    ".yaml",
    ".xml",
    ".html",
    ".htm",
    ".sql",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".java",
    ".go",
    ".rs",
    ".sh",
}


def _safe_load_text(path: Path) -> list[Document]:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return TextLoader(str(path), encoding=encoding).load()
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Cannot load text file {path}: {last_error}")


def load_documents(raw_data_dir: str = RAW_DATA_DIR) -> list[Document]:
    """
    Tải tất cả file được hỗ trợ từ *raw_data_dir*.

    Phương pháp tải theo đuôi file:
      .pdf   → PyPDFLoader (mỗi trang = 1 Document)
      .docx  → Docx2txtLoader (toàn bộ file = 1 Document)
      others → TextLoader (thử nhiều encoding: utf-8, utf-8-sig, latin-1)

    Mỗi Document được đính kèm metadata: source_file, source_path,
    relative_path (tương đối với raw_data_dir), file_type, loader_index.
    File không hỗ trợ sẽ bị bỏ qua; lỗi tải sẽ được log và bỏ qua.
    """
    docs: list[Document] = []
    root = Path(raw_data_dir)

    logger.info("Loading raw documents from: %s", root)

    if not root.exists():
        logger.warning("Raw data directory does not exist: %s", root)
        return docs

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif suffix == ".docx":
                loaded = Docx2txtLoader(str(path)).load()
            elif suffix in SUPPORTED_TEXT_EXTENSIONS:
                loaded = _safe_load_text(path)
            else:
                logger.info("Skipping unsupported file: %s", path)
                continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load file: %s", path)
            continue

        relative_path = str(path.relative_to(
            root)) if path.is_relative_to(root) else path.name

        for index, doc in enumerate(loaded):
            doc.metadata.update(
                {
                    "source_file": path.name,
                    "source_path": str(path),
                    "relative_path": relative_path,
                    "file_type": suffix.replace(".", ""),
                    "loader_index": index,
                }
            )

        docs.extend(loaded)

    logger.info("Loaded document units: %s", len(docs))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Cắt nhỏ tài liệu thành chunk dùng RecursiveCharacterTextSplitter.

    Separator ưu tiên heading Markdown („\n# “, „\n## “ …) để giữ
    cấu trúc tài liệu kỹ thuật vốn có heading phân cấp rõ ràng.
    Mỗi chunk được đính thêm: chunk_index, content_hash (SHA-256),
    chunk_size, chunk_overlap, embedding_model để hỗ trợ tra cứu và debug.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(docs)

    for chunk_index, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")).hexdigest()
        chunk.metadata.update(
            {
                "chunk_index": chunk_index,
                "content_hash": content_hash,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL,
            }
        )

    logger.info("Created chunks: %s", len(chunks))
    return chunks


def make_point_id(chunk: Document) -> str:
    """
    UUID xác định (deterministic) để /ingest trở thành idempotent với các chunk không thay đổi.
    Chạy lại /ingest trên cùng file sẽ cập nhật cùng point ID thay vì tạo điểm trùng lặp.
    """
    raw = "|".join(
        [
            str(chunk.metadata.get("relative_path", "")),
            str(chunk.metadata.get("loader_index", "")),
            str(chunk.metadata.get("chunk_index", "")),
            str(chunk.metadata.get("content_hash", "")),
            str(chunk.metadata.get("embedding_model", "")),
        ]
    )

    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def batched(items: list[Document], batch_size: int) -> Iterable[list[Document]]:
    """Phân trang danh sách *items* thành các lô kích thước *batch_size*."""
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def _extract_entities_for_document(doc_text: str, llm: ChatOllama) -> list[dict]:
    """
    Gọi LLM để trích xuất thực thể có tên từ văn bản tổng hợp của tài liệu.
    Trả về danh sách dict {name, type}.  Lỗi trả về [] (không gây dừng ingest).
    Mỗi Document gọi một lần (không phải mỗi Chunk) để giảm thời gian ingest.
    """
    prompt = (
        "Extract named entities (features, components, APIs, services, concepts) "
        "from this technical text.\n"
        'Return ONLY a JSON array: [{"name": "...", "type": "Feature|Component|API|Service|Concept|Other"}]\n'
        "Return [] if none found. No explanation — just the JSON array.\n\n"
        f"Text:\n{doc_text[:2000]}"
    )
    try:
        content = llm.invoke(prompt).content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if 0 <= start < end:
            return json.loads(content[start:end])
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
    return []


def get_embeddings() -> OllamaEmbeddings:
    """Trả về instance OllamaEmbeddings dùng *EMBEDDING_MODEL* hiện tại."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# ── Helpers suy luận metadata chunk ──────────────────────────────────────────

_DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("prd",          "prd"),
    ("brd",          "brd"),
    ("schema",       "schema"),
    ("api-",         "api"),
    ("-api-",        "api"),
    ("architecture", "architecture"),
    ("hardening",    "security"),
    ("audit",        "audit"),
    ("portal",       "portal"),
    ("marketplace",  "marketplace"),
    ("terminology",  "glossary"),
    ("settings",     "settings"),
    ("billing",      "billing"),
    ("auth-",        "auth"),
    ("-auth-",       "auth"),
    ("partner",      "partner"),
    ("meeting",      "meeting-notes"),
    ("substrate",    "infrastructure"),
    ("foundation",   "infrastructure"),
    ("intelligence", "intelligence"),
]


def _infer_module(relative_path: str, project: str) -> str:
    """
    Trích xuất tên module từ cấu trúc thư mục tương đối.
    project/module/file.md  → trả về module.
    file.md (phẳng)         → trả về project.
    """
    parts = Path(relative_path).parts
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2 and not Path(parts[0]).suffix:
        return parts[0]
    return project


def _infer_doc_type(source_file: str, relative_path: str) -> str:
    """Suy luận loại tài liệu (prd, schema, api...) từ tên file và đường dẫn."""
    combined = (source_file + " " + relative_path).lower()
    for keyword, doc_type in _DOC_TYPE_PATTERNS:
        if keyword in combined:
            return doc_type
    return "document"


def _infer_chunk_type(content: str) -> str:
    """
    Heuristic phân loại chunk theo nội dung:
      table     — nhiều dòng chứa ký tự |
      code      — dòng đầu bắt đầu bằng ký hiệu code
      paragraph — mặc định
    """
    lines = [ln for ln in content.strip().splitlines() if ln.strip()]
    if not lines:
        return "paragraph"
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    if pipe_lines >= 2 and pipe_lines / len(lines) > 0.4:
        return "table"
    code_starters = (
        "```", "    ", "\t",
        "def ", "class ", "function ", "import ", "from ",
        "const ", "var ", "let ", "public ", "private ",
        "SELECT ", "INSERT ", "UPDATE ", "CREATE ",
    )
    if any(lines[0].startswith(s) for s in code_starters):
        return "code"
    indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
    if len(lines) >= 3 and indented / len(lines) > 0.5:
        return "code"
    return "paragraph"


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Tạo Qdrant collection nếu chưa tồn tại.
    Sử dụng COSINE similarity với kích thước vector được phát hiện tự động từ model embedding.
    Idempotent: gọi nhiều lần không gây lỗi.
    """
    if client.collection_exists(collection_name):
        return

    logger.info(
        "Creating Qdrant collection '%s' with vector size %s",
        collection_name,
        vector_size,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def _ingest_project(project: str, reset: bool = False) -> dict:
    collection_name = get_collection_name(project)
    project_dir = str(Path(RAW_DATA_DIR) / project)

    logger.info("Ingesting project '%s' → collection '%s'",
                project, collection_name)
    logger.info("Reset mode: %s | dir: %s", reset, project_dir)

    docs = load_documents(project_dir)

    if not docs:
        return {
            "project": project,
            "collection": collection_name,
            "status": "empty",
            "message": f"No supported files found in {project_dir}",
            "documents": 0,
            "chunks": 0,
        }

    chunks = split_documents(docs)
    embeddings = get_embeddings()
    client = QdrantClient(url=QDRANT_URL)

    if reset and client.collection_exists(collection_name):
        logger.warning("Deleting collection before reset: %s", collection_name)
        client.delete_collection(collection_name)

    if reset and GRAPH_ENABLED and _graph_module is not None:
        try:
            _graph_module.delete_project_graph(project)
        except Exception as exc:
            logger.warning("Graph delete failed (non-fatal): %s", exc)

    logger.info("Creating sample embedding to detect vector size...")
    sample_vector = embeddings.embed_query("health check")
    _ensure_collection(client, collection_name, vector_size=len(sample_vector))

    total_upserted = 0
    # (chunk, qdrant_point_id) — dùng để lưu vào Neo4j graph
    all_pairs: list[tuple] = []

    for batch_no, chunk_batch in enumerate(batched(chunks, UPSERT_BATCH_SIZE), start=1):
        logger.info("Embedding batch %s: %s chunks",
                    batch_no, len(chunk_batch))

        texts = [chunk.page_content for chunk in chunk_batch]
        vectors = embeddings.embed_documents(texts)

        _now = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=make_point_id(chunk),
                vector=vector,
                payload={
                    **chunk.metadata,
                    "page_content": chunk.page_content,
                    "project": project,
                    "module": _infer_module(
                        chunk.metadata.get("relative_path", ""), project
                    ),
                    "doc_type": _infer_doc_type(
                        chunk.metadata.get("source_file", ""),
                        chunk.metadata.get("relative_path", ""),
                    ),
                    "chunk_type": _infer_chunk_type(chunk.page_content),
                    "language": "vi",
                    "status": "active",
                    "created_at": _now,
                },
            )
            for chunk, vector in zip(chunk_batch, vectors)
        ]

        client.upsert(collection_name=collection_name,
                      points=points, wait=True)
        all_pairs.extend(zip(chunk_batch, [str(p.id) for p in points]))
        total_upserted += len(points)
        logger.info("Upserted %s/%s chunks", total_upserted, len(chunks))

    count_result = client.count(collection_name=collection_name, exact=True)
    logger.info("Project '%s' done. Points: %s", project, count_result.count)

    # ── Tích hợp Neo4j graph ──────────────────────────────────────────────────
    graph_chunks_saved = 0
    if GRAPH_ENABLED and _graph_module is not None:
        try:
            _graph_module.setup_schema()
            graph_chunks_saved = _graph_module.save_project_graph(
                project, all_pairs)

            if GRAPH_ENTITY_EXTRACTION and CHAT_MODEL:
                logger.info(
                    "Entity extraction enabled for project '%s'", project)
                llm = ChatOllama(model=CHAT_MODEL,
                                 base_url=OLLAMA_BASE_URL, temperature=0)
                # Mỗi Document gọi LLM một lần (không phải mỗi Chunk) — nhanh hơn nhiều
                doc_pairs: dict[str, list[tuple]] = {}
                for chunk, pid in all_pairs:
                    sf = chunk.metadata.get("source_file", "unknown")
                    doc_pairs.setdefault(sf, []).append((chunk, pid))
                for sf, pairs in doc_pairs.items():
                    # Lấy tối đa 5 chunk đầu làm đại diện văn bản tài liệu
                    doc_text = " ".join(c.page_content for c, _ in pairs[:5])
                    entities = _extract_entities_for_document(doc_text, llm)
                    for chunk, pid in pairs:
                        chunk_entities = [
                            e for e in entities
                            if (e.get("name") or "").lower() in chunk.page_content.lower()
                        ]
                        if chunk_entities:
                            _graph_module.save_entities(
                                pid, chunk_entities, project)
                logger.info(
                    "Entity extraction complete for project '%s'", project)
        except Exception as exc:
            logger.warning("Graph integration failed (non-fatal): %s", exc)

    return {
        "project": project,
        "collection": collection_name,
        "status": "ok",
        "documents": len(docs),
        "chunks": len(chunks),
        "upserted": total_upserted,
        "points_count": count_result.count,
        "graph_chunks_saved": graph_chunks_saved,
        "idempotent": True,
        "reset": reset,
    }


def ingest(project: str | None = None, reset: bool = False) -> dict:
    """
    Ingest tài liệu vào Qdrant.
    - project=None  → ingest TẤT CẢ thư mục project con trong RAW_DATA_DIR.
    - project="foo" → chỉ ingest data/raw/foo/ vào collection yan_raw_docs__foo.
    """
    if project is not None:
        return _ingest_project(project, reset=reset)

    projects = get_projects()
    if not projects:
        return {
            "status": "empty",
            "message": (
                f"No project subdirectories found in {RAW_DATA_DIR}. "
                "Create subdirectories like data/raw/auth/, data/raw/marketplace/, etc."
            ),
            "projects": {},
        }

    logger.info("Ingesting %s project(s): %s", len(projects), projects)

    results: dict = {}
    for proj in projects:
        results[proj] = _ingest_project(proj, reset=reset)

    return {
        "status": "ok",
        "projects_ingested": list(results.keys()),
        "total_documents": sum(r.get("documents", 0) for r in results.values()),
        "total_chunks": sum(r.get("chunks", 0) for r in results.values()),
        "reset": reset,
        "details": results,
    }
