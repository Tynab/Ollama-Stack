import hashlib
import json
import logging
import os
import uuid
from collections.abc import Iterable
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
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
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

        points = [
            PointStruct(
                id=make_point_id(chunk),
                vector=vector,
                payload={
                    **chunk.metadata,
                    "page_content": chunk.page_content,
                    "project": project,
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
