"""
workflow.py — Bộ điều phối SDLC workflow dùng LangGraph.

Cấu trúc đồ thị (tuần tự nghiêm ngặt)
--------------------------------------
  pm → ba → sa → qa_shiftleft → devops_env
     → be → fe → qa_exec → devops_release → pm_closure → END

Mỗi node thực thi ba giai đoạn:
  1. Tổng hợp context  — chèn output (đã cắt ngắn) từ các bước phụ thuộc.
  2. Bổ sung RAG       — gọi HTTP tùy chọn đến rag-api /ask để lấy kiến thức ngữ cảnh.
  3. Gọi LLM           — gọi ChatOllama với system prompt của agent + context đã tổng hợp.

Quản lý State (LangGraph reducers)
-----------------------------------
  step_outputs   : Annotated[dict, operator.or_]   — mỗi node gộp slice của mình
  completed_steps: Annotated[list, operator.add]   — mỗi node nối thêm vai trò vào list
  Các trường còn lại là TypedDict thông thường (ghi đè lần cuối thắng).

Giới hạn context window
------------------------
  Mỗi output phụ thuộc được cắt tối đa MAX_PREV_OUTPUT_CHARS (3 000 ký tự).
  Kết quả RAG cũng được cắt theo cùng giới hạn.
  Tổng context mỗi bước ≈ (|deps| + 1) × 3 000 + system_prompt + user_input.
  Ollama context length được đặt là 32 768 tokens (OLLAMA_CONTEXT_LENGTH).

Concurrency
-----------
  _workflow_lock dùng double-checked locking để đồ thị được biên dịch đúng một lần
  dù nhiều request đến đồng thời.
"""

import logging
import os
import threading
from typing import Annotated, TypedDict
import operator

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agents import AGENTS, MAX_PREV_OUTPUT_CHARS, WORKFLOW_STEPS, AgentConfig

logger = logging.getLogger("agent-workflow")

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
RAG_API_URL: str = os.environ.get("RAG_API_URL", "http://rag-api:8090")
RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "5"))
# Context window Ollama (tokens). Tăng nếu model hỗ trợ window lớn hơn.
OLLAMA_NUM_CTX: int = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "32768"))
# Timeout (giây) cho mỗi lần gọi RAG /ask.
# 120s là quá thấp khi Ollama phải cold-start load model; dùng 300s mặc định.
RAG_TIMEOUT: int = int(os.environ.get("RAG_TIMEOUT", "300"))


# ── LangGraph State ───────────────────────────────────────────────────────────

class SDLCState(TypedDict):
    """State dùng chung được truyền qua các LangGraph node."""

    project: str | None           # RAG collection filter (optional)
    user_input: str               # original business goal / request
    rag_enabled: bool             # whether to query RAG per step
    rag_top_k: int                # number of RAG results per query
    rag_api_url: str              # rag-api base URL
    ollama_base_url: str          # Ollama base URL

    # operator.or_  → gộp dict:  {**hiện_tại, **node_return}
    step_outputs: Annotated[dict[str, str], operator.or_]

    # operator.add  → nối list: hiện_tại + node_return
    completed_steps: Annotated[list[str], operator.add]

    error: str | None             # thông báo lỗi đầu tiên; None nếu không có lỗi


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_rag(rag_api_url: str, question: str, project: str | None, top_k: int) -> str:
    """
    Gọi endpoint /ask của rag-api và trả về văn bản trả lời.
    Trả về chuỗi rỗng nếu có lỗi (không gây dừng workflow).
    """
    try:
        payload: dict = {"question": question, "top_k": top_k}
        if project:
            payload["project"] = project
        resp = requests.post(
            f"{rag_api_url}/ask",
            json=payload,
            timeout=RAG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as exc:
        logger.warning("RAG query failed (non-fatal): %s", exc)
        return ""


def _truncate(text: str, max_chars: int = MAX_PREV_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... truncated at {max_chars} chars ...]"


def _call_agent(
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
) -> str:
    """Gọi LLM với system prompt của agent + context đã tổng hợp."""
    llm = ChatOllama(
        model=agent.model,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=OLLAMA_NUM_CTX,
    )
    response = llm.invoke([
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=context),
    ])
    return str(response.content)


# ── Factory tạo node ───────────────────────────────────────────────────────────────

def _build_node(role: str):
    """
    Trả về hàm LangGraph node cho vai trò SDLC đã cho.

    Hàm trả về dùng closure để nắm giữ *agent*, mỗi node có config riêng
    mà không cần tra cứu global lúc chạy.
    __name__ được đặt tường minh để LangGraph hiển thị tên vai trò
    thay vì tên hàm chung 'node_fn'.
    """
    agent = AGENTS[role]

    def node_fn(state: SDLCState) -> dict:
        logger.info("Step %d | %s | model=%s",
                    agent.step_id, agent.name, agent.model)

        step_outputs: dict[str, str] = state.get("step_outputs", {})

        # 1. Tổng hợp context
        context_parts: list[str] = [
            f"## Business Goal / User Input\n{state['user_input']}"
        ]

        for dep in agent.depends_on:
            if dep in step_outputs:
                dep_name = AGENTS[dep].name
                context_parts.append(
                    f"\n## {dep_name} Output\n{_truncate(step_outputs[dep])}"
                )

        # 2. Bổ sung RAG (tùy chọn)
        if state.get("rag_enabled") and state.get("rag_api_url"):
            _hint = agent.rag_query_hint or agent.name
            rag_question = f"{_hint}: {state['user_input']}"
            rag_text = _query_rag(
                state["rag_api_url"],
                rag_question,
                state.get("project"),
                state.get("rag_top_k", RAG_TOP_K),
            )
            if rag_text:
                context_parts.append(
                    f"\n## RAG Knowledge Base Context\n{_truncate(rag_text)}"
                )

        context = "\n".join(context_parts)

        # 3. Gọi LLM
        error_msg: str | None = None
        try:
            output = _call_agent(agent, state.get(
                "ollama_base_url", OLLAMA_BASE_URL), context)
        except Exception as exc:
            logger.error("Agent '%s' failed: %s", role, exc)
            output = f"[ERROR in {role}] {exc}"
            error_msg = f"Step '{role}' failed: {exc}"

        logger.info("Step %d | %s | output_len=%d",
                    agent.step_id, agent.name, len(output))

        # Trả về từng slice — reducer xử lý gộp/nối.
        # Luôn bao gồm "error" để cập nhật trường khi có lỗi.
        result: dict = {
            "step_outputs": {role: output},
            "completed_steps": [role],
        }
        if error_msg:
            result["error"] = error_msg
        return result

    node_fn.__name__ = f"{role}_node"
    return node_fn


# ── Xây dựng đồ thị ───────────────────────────────────────────────────────────────

def build_workflow():
    """
    Biên dịch SDLC StateGraph.

    Node được thêm theo thứ tự WORKFLOW_STEPS; cạnh được thêm tuần tự
    để đồ thị phản ánh luồng SDLC kiểu Waterfall/Agile truyền thống.
    Trả về LangGraph runnable đã biên dịch.
    """
    graph: StateGraph = StateGraph(SDLCState)

    for role in WORKFLOW_STEPS:
        graph.add_node(role, _build_node(role))

    graph.set_entry_point(WORKFLOW_STEPS[0])

    for i in range(len(WORKFLOW_STEPS) - 1):
        graph.add_edge(WORKFLOW_STEPS[i], WORKFLOW_STEPS[i + 1])

    graph.add_edge(WORKFLOW_STEPS[-1], END)

    return graph.compile()


# ── Singleton ─────────────────────────────────────────────────────────────────
# Double-checked locking đảm bảo đồ thị chỉ được biên dịch đúng một lần
# dù nhiều request đến đồng thời lúc khởi động.

_workflow = None
_workflow_lock = threading.Lock()


def get_workflow():
    global _workflow
    if _workflow is None:
        with _workflow_lock:
            if _workflow is None:  # kiểm tra lại bên trong lock
                _workflow = build_workflow()
    return _workflow


# ── Chạy từng bước đơn lẻ (dùng bởi endpoint /agent/{role}) ─────────────────

def run_single_step(
    role: str,
    user_input: str,
    project: str | None = None,
    extra_context: str | None = None,
    prev_outputs: dict[str, str] | None = None,
    rag_enabled: bool = True,
    rag_top_k: int = RAG_TOP_K,
    ollama_base_url: str = OLLAMA_BASE_URL,
    rag_api_url: str = RAG_API_URL,
) -> str:
    """
    Chạy một bước agent đơn lẻ mà không cần chạy toàn bộ workflow.

    Xây dựng cùng context mà workflow node sẽ thấy
    (output phụ thuộc + extra_context tùy chọn + RAG tùy chọn),
    sau đó gọi LLM trực tiếp.  Hữu ích cho:
    - Test từng agent riêng lẻ qua POST /agent/{role}.
    - Chạy từng bước thủ công khi người dùng dán output trước vào *prev_outputs*.
    """
    if role not in AGENTS:
        raise ValueError(
            f"Unknown role '{role}'. Valid roles: {list(AGENTS.keys())}")

    agent = AGENTS[role]
    step_outputs: dict[str, str] = prev_outputs or {}

    context_parts: list[str] = [f"## Business Goal / User Input\n{user_input}"]

    for dep in agent.depends_on:
        if dep in step_outputs:
            dep_name = AGENTS[dep].name
            context_parts.append(
                f"\n## {dep_name} Output\n{_truncate(step_outputs[dep])}"
            )

    if extra_context:
        context_parts.append(f"\n## Additional Context\n{extra_context}")

    if rag_enabled and rag_api_url:
        _hint = agent.rag_query_hint or agent.name
        rag_question = f"{_hint}: {user_input}"
        rag_text = _query_rag(rag_api_url, rag_question, project, rag_top_k)
        if rag_text:
            context_parts.append(
                f"\n## RAG Knowledge Base Context\n{_truncate(rag_text)}"
            )

    context = "\n".join(context_parts)
    return _call_agent(agent, ollama_base_url, context)
