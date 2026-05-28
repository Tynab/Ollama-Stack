"""
app.py — YAN SDLC Agent Orchestrator API  (cổng 8091)

Endpoints
---------
GET  /health                 Thông tin service: Ollama URL, RAG URL, số agent.
GET  /agents                 Liệt kê cấu hình tất cả agent (step, model, depends_on).
POST /agent/{role}           Gọi đồng bộ một bước agent đơn lẻ.
                             Body: AgentStepRequest  →  AgentStepResponse
POST /workflow/run           Gửi workflow SDLC 13 agents để chạy nền.
                             Body: WorkflowRunRequest  →  {workflow_id, status}
GET  /workflow/{workflow_id} Kiểm tra trạng thái hoặc lấy kết quả đã hoàn thành.
                             Response: WorkflowRecord
GET  /workflows              Liệt kê workflow gần đây (mới nhất trước).

Vòng đời workflow
-----------------
  POST /workflow/run  →  status=pending  (record lưu, BG task xếp hàng)
     └─> nền          →  status=running  (LangGraph invoke bắt đầu)
           └─> xong   →  status=completed | failed

Ghi chú Concurrency
-------------------
- _store_lock bảo vệ workflow_store khỏi đọc/ghi đồng thời.
- get_workflow() dùng double-checked locking để đồ thị chỉ được biên dịch
  một lần dù nhiều request đến lúc khởi động.
- FastAPI BackgroundTasks chạy workflow trong thread-pool thread;
  _run_workflow_task cập nhật WorkflowRecord trực tiếp (không cần re-insert
  vì dict là kiểu tham chiếu).
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from artifacts import ARTIFACT_ROLES as _ARTIFACT_ROLES, extract_and_save as _extract_artifacts, list_artifacts as _list_artifacts, read_artifact as _read_artifact
from agents import AGENTS, WORKFLOW_STEPS
from workflow import (
    OLLAMA_BASE_URL,
    RAG_API_URL,
    RAG_TOP_K,
    SDLCState,
    get_workflow,
    run_single_step,
)


def _require_env(name: str) -> str:
    """Trả về giá trị biến môi trường *name*, raise RuntimeError nếu không tồn tại."""
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

logger = logging.getLogger("agent-api")

app = FastAPI(title="YAN SDLC Agent Orchestrator", version="1.0.0")

_STATIC_DIR = Path(__file__).parent / "static"


# ── Mô hình Request / Response ─────────────────────────────────────────────

class AgentStepRequest(BaseModel):
    user_input: str = Field(..., min_length=1,
                            description="Mục tiêu kinh doanh hoặc context đầu vào cho agent")
    project: str | None = Field(
        None, description="RAG project filter (ví dụ: 'yanlib')")
    extra_context: str | None = Field(
        None, description="Context bổ sung để chèn vào")
    prev_outputs: dict[str, str] | None = Field(
        None, description="Output của các bước trước để chèn vào context (role → text)"
    )
    tech_stack: list[str] | None = Field(
        None, description="Danh sách tech stack bắt buộc. Ví dụ: ['nestjs', 'reactjs', 'mongodb']")
    rag_enabled: bool = Field(
        True, description="Có query RAG knowledge base không")
    rag_top_k: int = Field(RAG_TOP_K, ge=1, le=20)


class AgentStepResponse(BaseModel):
    role: str
    name: str
    model: str
    output: str


class WorkflowRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1,
                            description="Mục tiêu kinh doanh / ý tưởng project để chạy qua SDLC")
    project: str | None = Field(
        None, description="RAG project filter (ví dụ: 'yanlib')")
    rag_enabled: bool = Field(True)
    rag_top_k: int = Field(RAG_TOP_K, ge=1, le=20)
    tech_stack: list[str] | None = Field(
        None,
        description=(
            "Danh sách tech stack bắt buộc (ngôn ngữ, framework, DB, infra). "
            "Ví dụ: ['nestjs', 'reactjs', 'mongodb', 'k8s', 'react native']. "
            "TA Agent sẽ thiết kế kiến trúc dựa trên stack này; các agent khác "
            "cũng sẽ bám sát đúng các công nghệ này."
        ),
    )


class WorkflowStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class WorkflowRecord(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    project: str | None
    user_input: str
    tech_stack: list[str] | None = None
    step_outputs: dict[str, str] = Field(default_factory=dict)
    completed_steps: list[str] = Field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, list] = Field(default_factory=dict)
    created_at: str
    completed_at: str | None = None


# ── Lưu trữ workflow trong bộ nhớ (khóa theo workflow_id) ──────────────────────
# Dùng cho local stack; môi trường production nên dùng Redis hoặc database.
# _store_lock đồng bộ hóa mọi thao tác đọc/ghi để tránh race condition.

workflow_store: dict[str, WorkflowRecord] = {}
_store_lock = threading.Lock()

_MAX_STORED_WORKFLOWS = 50  # xóa entry cũ nhất khi đạt giới hạn

# ── Tài nguyên bộ nhớ sự kiện ────────────────────────────────────────────────────

MEMORY_DIR: str = os.environ.get("MEMORY_DIR", "/data/memory")
_MAX_INPUT_CHARS: int = 10_000  # Giới hạn input để phòng context overflow


def _sanitize_input(text: str) -> str:
    """
    Kiểm tra và làm sạch input người dùng tại system boundary.
    Loại bỏ ký tự điều khiển (ngoại trừ newline/tab) và cắt ngắn nếu quá dài.
    Theo nguyên tắc SecOps: validate input trước khi đưa vào LLM context.
    """
    sanitized = "".join(ch for ch in text if ch >= " " or ch in "\n\r\t")
    if len(sanitized) > _MAX_INPUT_CHARS:
        logger.warning(
            "Input truncated from %d to %d chars", len(sanitized), _MAX_INPUT_CHARS
        )
        sanitized = sanitized[:_MAX_INPUT_CHARS]
    return sanitized.strip()


def _log_workflow_run(
    workflow_id: str,
    project: str | None,
    user_input: str,
    completed_steps: list[str],
    status: str,
    error: str | None,
    duration_seconds: float,
) -> None:
    """
    Ghi thông tin workflow run ra file JSONL làm seed dữ liệu episodic memory.
    Non-fatal: lỗi ghi file không làm gián đoạn workflow.
    """
    try:
        log_dir = Path(MEMORY_DIR) / "episodic"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "workflow_id": workflow_id,
            "project": project,
            "user_input": user_input[:500],
            "completed_steps": completed_steps,
            "steps_count": len(completed_steps),
            "status": status,
            "error": error,
            "duration_seconds": round(duration_seconds, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_file = log_dir / "workflow_runs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Workflow run logged: %s", workflow_id)
    except Exception as exc:
        logger.warning("Workflow run logging failed (non-fatal): %s", exc)


def _store_workflow(record: WorkflowRecord) -> None:
    """Lưu *record* vào store, xóa entry cũ nhất khi store đầy."""
    with _store_lock:
        if len(workflow_store) >= _MAX_STORED_WORKFLOWS:
            oldest_key = next(iter(workflow_store))
            del workflow_store[oldest_key]
        workflow_store[record.workflow_id] = record


# ── Chạy workflow nền ───────────────────────────────────────────────────────────────

def _run_workflow_task(workflow_id: str, req: WorkflowRunRequest) -> None:
    """
    Task nền: chạy toàn bộ SDLC LangGraph workflow và cập nhật
    WorkflowRecord trực tiếp. Được gọi bởi FastAPI BackgroundTasks.

    Việc cập nhật *record* an toàn vì:
    - Chỉ có duy nhất một background task cho mỗi workflow_id.
    - Các endpoint đọc chỉ đọc các trường được gán nguyên tử (status, timestamps).
    """
    record = workflow_store.get(workflow_id)
    if record is None:
        return

    record.status = WorkflowStatus.running
    logger.info("Workflow %s started", workflow_id)

    # user_input đã được sanitize khi enqueue — dùng lại từ record để tránh xử lý hai lần.
    sanitized_input = record.user_input

    initial_state: SDLCState = {
        "project": req.project,
        "user_input": sanitized_input,
        "tech_stack": req.tech_stack,
        "rag_enabled": req.rag_enabled,
        "rag_top_k": req.rag_top_k,
        "rag_api_url": RAG_API_URL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "step_outputs": {},
        "completed_steps": [],
        "error": None,
    }

    _start = time.monotonic()
    try:
        final_state: SDLCState = {}  # type: ignore[assignment]
        for chunk in get_workflow().stream(initial_state, stream_mode="updates"):
            # chunk = {node_name: partial_state_dict}  (stream_mode="updates")
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    # Cập nhật real-time để GET /workflow/{id} phản ánh tiến trình
                    if "step_outputs" in node_output:
                        record.step_outputs.update(node_output["step_outputs"])
                        # Trích xuất artifact cho các role có file code (non-fatal)
                        for _role, _out in node_output["step_outputs"].items():
                            if (
                                _role in _ARTIFACT_ROLES
                                and _out
                                and not _out.startswith("[LỖI")
                            ):
                                _arts = _extract_artifacts(_role, _out, workflow_id)
                                if _arts:
                                    record.artifacts[_role] = _arts
                    if "completed_steps" in node_output:
                        record.completed_steps = list(
                            dict.fromkeys(
                                record.completed_steps + node_output["completed_steps"]
                            )
                        )
                    if node_output.get("error"):
                        record.error = node_output["error"]
                    final_state.update(node_output)
        record.status = WorkflowStatus.failed if record.error else WorkflowStatus.completed
    except Exception as exc:
        logger.exception("Workflow %s thất bại", workflow_id)
        record.status = WorkflowStatus.failed
        record.error = str(exc)
    finally:
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _duration = time.monotonic() - _start
        logger.info("Workflow %s hoàn thành với status=%s",
                    workflow_id, record.status)
        _log_workflow_run(
            workflow_id=workflow_id,
            project=req.project,
            user_input=sanitized_input,
            completed_steps=record.completed_steps,
            status=record.status.value,
            error=record.error,
            duration_seconds=_duration,
        )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    """Trả về trạng thái service và cấu hình runtime. Dùng bởi Docker healthcheck."""
    return {
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "rag_api_url": RAG_API_URL,
        "agents": len(AGENTS),
        "workflow_steps": WORKFLOW_STEPS,
    }


@app.get("/agents")
def list_agents() -> dict[str, Any]:
    """Liệt kê cấu hình tất cả agent: step_id, name, model và chuỗi phụ thuộc."""
    return {
        role: {
            "step_id": cfg.step_id,
            "name": cfg.name,
            "model": cfg.model,
            "depends_on": cfg.depends_on,
        }
        for role, cfg in AGENTS.items()
    }


@app.post("/agent/{role}", response_model=AgentStepResponse)
def run_agent_step(role: str, req: AgentStepRequest) -> AgentStepResponse:
    """
    Chạy đồng bộ một bước agent đơn lẻ.

    Hữu ích để test từng agent riêng lẻ hoặc ghép nối bước thủ công
    khi không cần chạy toàn bộ workflow tuần tự.
    Truyền *prev_outputs* để chèn context từ các bước trước.
    """
    if role not in AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Role '{role}' không tồn tại. Hợp lệ: {list(AGENTS.keys())}",
        )

    agent = AGENTS[role]
    logger.info("Bước đơn lẻ: role=%s project=%s", role, req.project)

    try:
        output = run_single_step(
            role=role,
            user_input=_sanitize_input(req.user_input),
            project=req.project,
            extra_context=req.extra_context,
            prev_outputs=req.prev_outputs,
            tech_stack=req.tech_stack,
            rag_enabled=req.rag_enabled,
            rag_top_k=req.rag_top_k,
            ollama_base_url=OLLAMA_BASE_URL,
            rag_api_url=RAG_API_URL,
        )
    except Exception as exc:
        logger.exception("Bước agent thất bại: role=%s", role)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentStepResponse(
        role=role,
        name=agent.name,
        model=agent.model,
        output=output,
    )


@app.post("/workflow/run")
def start_workflow(req: WorkflowRunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Gửi workflow SDLC 13 agent để chạy bất đồng bộ.

    Trả về ngay với workflow_id. Poll GET /workflow/{id} để kiểm tra trạng thái.
    Workflow chạy tuần tự: BA → PM → SA → TA → Designer → FE → Mobile → DBA → BE → DA → Tech Lead → Tester → DevSecOps.
    Mỗi bước nhận output đã cắt ngắn của các bước phụ thuộc làm context.
    """
    workflow_id = str(uuid.uuid4())
    # Sanitize trước khi lưu vào record — đảm bảo API response và JSONL log nhất quán.
    sanitized_input = _sanitize_input(req.user_input)
    record = WorkflowRecord(
        workflow_id=workflow_id,
        status=WorkflowStatus.pending,
        project=req.project,
        user_input=sanitized_input,
        tech_stack=req.tech_stack,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _store_workflow(record)
    background_tasks.add_task(_run_workflow_task, workflow_id, req)
    logger.info("Workflow %s đã được xếp hàng", workflow_id)
    return {
        "workflow_id": workflow_id,
        "status": "pending",
        "message": f"Workflow đã được xếp hàng. Poll GET /workflow/{workflow_id} để kiểm tra trạng thái.",
    }


@app.get("/workflow/{workflow_id}", response_model=WorkflowRecord)
def get_workflow_status(workflow_id: str) -> WorkflowRecord:
    """Trả về trạng thái hiện tại của một lần chạy workflow (pending / running / completed / failed)."""
    with _store_lock:
        record = workflow_store.get(workflow_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' không tìm thấy. Có thể đã bị xóa khỏi store hoặc chưa bắt đầu.",
        )
    return record


@app.get("/workflows")
def list_workflows() -> dict[str, Any]:
    """Liệt kê các workflow gần đây (mới nhất trước, tối đa _MAX_STORED_WORKFLOWS bản ghi)."""
    with _store_lock:
        snapshot = list(workflow_store.values())
    return {
        "count": len(snapshot),
        "workflows": [
            {
                "workflow_id": r.workflow_id,
                "status": r.status,
                "project": r.project,
                "completed_steps": len(r.completed_steps),
                "total_steps": len(WORKFLOW_STEPS),
                "created_at": r.created_at,
                "completed_at": r.completed_at,
            }
            for r in reversed(snapshot)
        ],
    }


@app.get("/workflow/{workflow_id}/artifacts")
def get_workflow_artifacts(workflow_id: str) -> dict[str, Any]:
    """Liệt kê artifacts đã lưu cho workflow (metadata only, không bao gồm nội dung file)."""
    with _store_lock:
        record = workflow_store.get(workflow_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' không tìm thấy.",
        )
    # Merge in-memory artifacts with disk (in case container was restarted)
    disk_arts = _list_artifacts(workflow_id)
    merged = {**disk_arts, **record.artifacts}
    return {"workflow_id": workflow_id, "artifacts": merged}


@app.get("/workflow/{workflow_id}/artifacts/{role}/{path:path}")
def get_artifact_file(
    workflow_id: str,
    role: str,
    path: str,
    download: bool = False,
) -> Any:
    """
    Trả về nội dung file artifact.
    GET ?download=1  → tải về dưới dạng binary.
    GET              → trả về JSON {path, language, content}.
    """
    from fastapi.responses import Response as _Resp

    # Validate role to prevent path traversal — only known artifact roles are valid.
    if role not in _ARTIFACT_ROLES:
        raise HTTPException(status_code=404, detail="File không tìm thấy.")

    result = _read_artifact(workflow_id, role, path)
    if result is None:
        raise HTTPException(status_code=404, detail="File không tìm thấy.")
    content, language = result
    if download:
        # Strip path separators and quote chars to prevent header injection.
        safe_filename = Path(path).name.replace('"', "").replace("\n", "").replace("\r", "")
        return _Resp(
            content=content.encode("utf-8"),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    return {"path": path, "language": language, "content": content}


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def workflow_ui() -> FileResponse:
    """Phục vụ SDLC Workflow UI (single-page application)."""
    return FileResponse(_STATIC_DIR / "workflow.html")
