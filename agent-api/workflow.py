"""
workflow.py — Bộ điều phối SDLC workflow dùng LangGraph.

Cấu trúc đồ thị (tuần tự nghiêm ngặt)
--------------------------------------
    ba -> pm -> sa -> ta -> designer -> tl -> fe -> mobile -> dba -> be -> da -> tech_lead -> tester -> devsecops -> clarifier -> END

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

import functools
import json as _json
import logging
import os
import re
import threading
from datetime import datetime
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
RAG_TIMEOUT: int = int(os.environ.get("RAG_TIMEOUT", "600"))
# Timeout (giây) cho mỗi lần gọi Ollama LLM.
OLLAMA_REQUEST_TIMEOUT: int = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "1200"))

# Model dùng để lập kế hoạch danh sách file (chat model, không phải code-completion model).
CODING_PLANNER_MODEL: str = os.environ.get(
    "CODING_PLANNER_MODEL",
    os.environ.get("BA_MODEL", "granite3.3:2b"),
)
# Số file tối đa mỗi role trong vòng lặp per-file.
_MAX_FILES_PER_ROLE: int = int(os.environ.get("MAX_FILES_PER_ROLE", "6"))


# ── Quy tắc chung được chèn vào system prompt của mọi agent ─────────────────
COMMON_AGENT_RULES: str = """\
Critical rules — apply to every response:
1. Use ONLY the provided User Input, Previous Agent Outputs, Required Tech Stack, and RAG Knowledge Base Context. Do not invent information from outside these sources.
2. Do not invent business requirements, APIs, database fields, UI screens, permissions, SLA rules, cost figures, or infrastructure config unless explicitly marked as [Proposed].
3. Classify important statements with one of these labels:
   - [Confirmed]: directly supported by input or context.
   - [Assumption]: reasonable inference, not explicitly stated — must be flagged.
   - [Open Question]: requires PO/PM/Tech confirmation before proceeding.
   - [Proposed]: suggested implementation option, not yet decided.
4. If information is missing, place it under Open Questions instead of silently filling the gap.
5. Do not repeat large sections from previous agents. Produce only the artifact owned by your role.
6. Keep IDs consistent across artifacts (req ID, user story ID, test case ID).
7. Prefer concise Markdown tables for implementation-ready outputs.
8. If sources are available in RAG context, reference the source file name in a Notes/Source column.
9. If actual source code is not provided, clearly label your output as Design Review, not Code Review. Do not invent file names, line numbers, or PR comments.
10. Complete sections in order. If context budget is exhausted before all sections are done, mark remaining sections as [Deferred — insufficient input] and stop cleanly. Do not produce partial sentences or half-filled tables.
11. LANGUAGE: Respond in English only. Do not write in Vietnamese, even if the user input or project context is in Vietnamese. All section headings, labels, table headers, and prose must be in English.
12. CROSS-AGENT CITATIONS: Whenever a decision, design choice, or data element traces back to a prior agent's output, cite it explicitly using the format "Agent §Section" (e.g., "per BA §3 FR-01", "per SA §3 /api/auth/login", "per TL §4 FE Task #3", "per Designer §5 Screen S-02"). Do NOT silently consume upstream information without citation.
13. INTRA-DOCUMENT LINKAGE: For every section in your output, add a brief note stating which other sections within this document it connects to, using "→ see §N" notation (e.g., "→ see §5 API Integration Map", "→ see §3 Component Breakdown"). This makes the dependency graph within your output explicit.
14. DEPTH OVER SUMMARY: Every section must contain complete, actionable, implementation-ready detail — not a summary or placeholder. Tables must have real data rows derived from the provided context. Bullet points must be specific (names, values, IDs), not generic descriptions. If you find yourself writing a generic statement like "handle errors appropriately", replace it with the exact error codes, HTTP statuses, and recovery actions required.
15. PLATFORM CONVENTIONS ARE BINDING: If the RAG Knowledge Base Context contains project-specific coding conventions, library names, naming rules, architectural patterns (e.g. three-seam pattern, scope=platform rows, Module Federation topology, tenantId-first index invariant), platform-provided shared modules (common-lib, guard decorators, base repositories), or Kafka topic names — treat ALL of these as HARD CONSTRAINTS that OVERRIDE generic best practices. Do not invent a new abstraction when the platform already provides one. Do not use a generic library when the platform mandates a specific one. Do not invent service names, Kafka topic names, route prefixes, port numbers, or decorator names — use only what the RAG context documents. If RAG documents a naming invariant (e.g. "tenantId is field #1 in every compound index"), apply it everywhere without exception.
"""


# ── Trạng thái LangGraph ───────────────────────────────────────────────────────────

class SDLCState(TypedDict):
    """Trạng thái dùng chung được truyền qua các LangGraph node."""

    project: str | None           # Bộ lọc RAG collection (tùy chọn)
    user_input: str               # Mục tiêu kinh doanh / yêu cầu gốc
    rag_enabled: bool             # Có truy vấn RAG mỗi bước không
    rag_top_k: int                # Số kết quả RAG mỗi truy vấn
    rag_api_url: str              # URL cơ sở của rag-api
    ollama_base_url: str          # URL cơ sở của Ollama

    # operator.or_  → gộp dict:  {**hiện_tại, **node_return}
    step_outputs: Annotated[dict[str, str], operator.or_]

    # operator.add  → nối list: hiện_tại + node_return
    completed_steps: Annotated[list[str], operator.add]

    tech_stack: list[str] | None    # danh sách công nghệ bắt buộc (ngôn ngữ, framework, DB, infra)

    error: str | None             # Thông báo lỗi đầu tiên; None nếu không có lỗi


# ── Hàm tiện ích ───────────────────────────────────────────────────────────────────

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
    """Cắt ngắn text im lặng — không thêm marker để tránh LLM tái tạo marker đó."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# Models that emit <think>...</think> chain-of-thought blocks — stripped from output.
# Values are substrings matched against model names (e.g. "qwen3" matches "qwen3.6:35b").
_REASONING_MODELS: frozenset[str] = frozenset({"phi4-mini-reasoning", "phi4-reasoning", "qwq", "deepseek-r1", "qwen3"})


def _strip_thinking(text: str) -> str:
    """Xóa các block <think>...</think> mà reasoning model phát ra, trả về phần nội dung thực sự."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _get_num_ctx(model: str) -> int:
    """Trả về num_ctx phù hợp với model.

    Một số model nhỏ được clamp ở 4 096 để giảm loop/hallucination
    và timeout trong bối cảnh pipeline nhiều bước.
    """
    if "codegemma:2b" in model.lower():
        return min(4096, OLLAMA_NUM_CTX)
    return OLLAMA_NUM_CTX


# Giới hạn chars tối đa MỖI dep output theo role (để tránh overflow context window)
_PER_DEP_CHARS: dict[str, int] = {
    "tech_lead":  800,    # 5 deps × 800 = 4 000 chars ≈ 1 000 tokens
    "devsecops": 1_000,  # 4 deps × 1 000 = 4 000 chars ≈ 1 000 tokens
    "tester":    3_000,  # 5 deps × 3 000 = 15 000 chars ≈ 3 700 tokens
    "clarifier": 1_500,  # 14 deps × 1 500 = 21 000 chars ≈ 5 250 tokens
}


# Roles sinh code/config theo từng file riêng biệt (loop-per-file approach).
# Không inject _FILE_OUTPUT_INSTRUCTION nữa vì một số model có thể lặp lại template verbatim.
# NOTE: "da" is intentionally absent — DA runs via _call_agent (produces a full analysis
# report in one shot). artifacts.py ARTIFACT_ROLES *does* include "da" so any code blocks
# in the DA output are still extracted to disk after the fact.
_ARTIFACT_ROLES: frozenset[str] = frozenset(
    {"fe", "mobile", "be", "dba", "devsecops", "tech_lead"}
)

# Template prompt cho bước lập kế hoạch file
_PLAN_PROMPT_TMPL = """\
You are a {role_name}. Based on the task and tech stack, list the source code / config files you will create.
Return ONLY a JSON array (no prose, no markdown wrapper):
[{{"filename": "src/components/Login.tsx", "description": "Login form component", "language": "typescript"}}]
Rules:
- Produce ONLY essential, non-trivial files. Minimum 2, maximum {max_files}. Do NOT pad to the limit — only list files that are truly needed.
- Use relative paths from project root.
- "language" must be a valid code fence name (typescript, python, yaml, sql, dockerfile, etc).
{extra}
Task: {user_input}
Tech stack: {tech_stack}
Context summary:
{context_summary}
"""

# Template prompt cho từng file
_FILE_GEN_PROMPT_TMPL = """\
Write the complete {language} source code for: {filename}
Purpose: {description}
Tech stack: {tech_stack}
{extra}

Strict rules:
- Respond with one triple-backtick code block containing the full working implementation.
- Do not include author/date/version/license/copyright metadata headers.
- Do not repeat identical import lines or boilerplate blocks.
- Prefer concise, production-ready code over placeholders.
"""


def _detect_db_type(tech_stack: list[str] | None) -> str:
    """Nhận diện loại DB từ tech_stack. Trả về 'nosql', 'sql', 'both', hoặc 'unknown'."""
    if not tech_stack:
        return "unknown"
    _nosql_kw = {"mongodb", "mongo", "nosql", "dynamodb", "cassandra",
                 "redis", "firestore", "couchdb", "elasticsearch"}
    _sql_kw   = {"postgresql", "mysql", "postgres", "sql server",
                 "sqlite", "mssql", "oracle", "mariadb"}
    combined  = " ".join(t.lower() for t in tech_stack)
    has_nosql = any(kw in combined for kw in _nosql_kw)
    has_sql   = any(kw in combined for kw in _sql_kw)
    if has_nosql and has_sql:
        return "both"
    if has_nosql:
        return "nosql"
    if has_sql:
        return "sql"
    return "unknown"


@functools.lru_cache(maxsize=32)
def _fence_pattern(language: str) -> re.Pattern:
    """Compile và cache pattern regex tích hợp sẵn cho ngôn ngữ chỉ định."""
    return re.compile(
        rf"```(?:{re.escape(language)}|{re.escape(language.lower())})\s*\n(.*?)(?:\n```\s*$|\n```\s*\Z)",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )


_FENCE_FALLBACK_RE = re.compile(
    r"```\w*\s*\n(.*?)(?:\n```\s*$|\n```\s*\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)


def _strip_code_fence(content: str, language: str) -> str:
    """Loại bỏ code fence bên ngoài để các fence lồng nhau không phá vỡ regex/markdown."""
    # Try to extract content between ```lang ... ``` (cached pattern per language).
    for pat in (_fence_pattern(language), _FENCE_FALLBACK_RE):
        m = pat.search(content)
        if m:
            return m.group(1).rstrip()
    # Fallback: strip leading/trailing fence lines
    lines = content.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _deloop(text: str, max_repeats: int = 8) -> str:
    """Cắt ngắn các dòng lặp vô hạn để giảm thiểu lỗi lặp vòng của model."""
    lines = text.split("\n")
    out: list[str] = []
    prev: str | None = None
    count = 0
    for ln in lines:
        key = ln.rstrip()
        if key and key == prev:
            count += 1
            if count <= max_repeats:
                out.append(ln)
            elif count == max_repeats + 1:
                out.append("// ... (repeated pattern truncated)")
        else:
            prev  = key
            count = 1
            out.append(ln)
    return "\n".join(out)


def _compute_extra_instruction(role: str, tech_stack: list[str] | None) -> str:
    """Sinh câu lệnh bổ sung riêng theo role để hướng dẫn việc tạo artifact."""
    if not tech_stack:
        if role == "tech_lead":
            return (
                "Write ARCHITECTURE.md or INTEGRATION.md with technical review, "
                "integration decisions, code standards, and specific code change recommendations."
            )
        return ""

    combined = " ".join(t.lower() for t in tech_stack)
    db_type  = _detect_db_type(tech_stack)

    if role == "dba":
        if db_type == "nosql":
            return (
                "Use MongoDB Mongoose models (TypeScript .ts files). "
                "NO SQL CREATE TABLE or DDL. "
                "Embedded documents and arrays instead of FK/JOINs."
            )
        if db_type == "sql":
            return "Use SQL DDL with CREATE TABLE, FOREIGN KEY references, and CREATE INDEX."
        if db_type == "both":
            return (
                "Create both SQL DDL (schema.sql) AND Mongoose models (.ts). "
                "SQL for relational data, MongoDB for document data."
            )

    if role == "fe":
        if "vue" in combined:
            return (
                "Write real Vue 3 component with Composition API, TypeScript, and template markup. "
                "No license headers."
            )
        if "angular" in combined:
            return "Write real Angular component with TypeScript, decorators, and template. No license headers."
        if "svelte" in combined:
            return "Write real Svelte component with TypeScript and reactive syntax. No license headers."
        # Default: React / Next.js
        return (
            "Write real React/TypeScript component with JSX markup, hooks, and props interface. "
            "No license headers."
        )

    if role == "mobile":
        if "react native" in combined or "expo" in combined:
            return (
                "Write real React Native component with TypeScript, JSX UI code, and StyleSheet. "
                "No license headers."
            )
        if "flutter" in combined or "dart" in combined:
            return "Write real Flutter/Dart widget with actual UI code. No license headers."
        # Default: offer both options
        return (
            "Write real Flutter/Dart widget or React Native component with actual UI code. "
            "No license headers."
        )

    if role == "be":
        if "fastapi" in combined or (
            "python" in combined and "django" not in combined and "flask" not in combined
        ):
            return (
                "Write real FastAPI route with Pydantic models and async business logic. "
                "No license headers."
            )
        if "django" in combined:
            return (
                "Write real Django view or DRF ViewSet with serializers and business logic. "
                "No license headers."
            )
        if "flask" in combined:
            return "Write real Flask route with request parsing and business logic. No license headers."
        if "express" in combined and "nest" not in combined:
            return (
                "Write real Express.js route with TypeScript types and business logic. "
                "No license headers."
            )
        if "spring" in combined or "java" in combined or "kotlin" in combined:
            return (
                "Write real Spring Boot service and controller with annotations and business logic. "
                "No license headers."
            )
        if "go" in combined or "golang" in combined:
            return "Write real Go HTTP handler with struct types and business logic. No license headers."
        # Default: NestJS
        return (
            "Write real NestJS service, controller, or DTO with actual business logic. "
            "No license headers."
        )

    if role == "devsecops":
        return (
            "Write real Kubernetes manifests (Deployment, Service, Ingress, ConfigMap, Secret), "
            "Dockerfile, and CI/CD pipeline YAML with working config. "
            "All sensitive env vars must go in K8s Secret objects referenced via secretKeyRef — "
            "never in plain env: blocks. No license headers."
        )
    if role == "tech_lead":
        return (
            "Write ARCHITECTURE.md or INTEGRATION.md with technical review, "
            "integration decisions, code standards, and specific code change recommendations."
        )
    return ""


def _plan_code_files(
    role: str,
    agent: AgentConfig,
    ollama_base_url: str,
    user_input: str,
    tech_stack: list[str] | None,
    context_summary: str,
    extra_instruction: str = "",
) -> list[dict]:
    """Dùng CODING_PLANNER_MODEL để lấy danh sách file cần tạo."""
    planner = ChatOllama(
        model=CODING_PLANNER_MODEL,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=min(8192, OLLAMA_NUM_CTX),
        request_timeout=float(OLLAMA_REQUEST_TIMEOUT),
    )
    stack_str = ", ".join(tech_stack) if tech_stack else "not specified"
    prompt = _PLAN_PROMPT_TMPL.format(
        role_name=agent.name,
        max_files=_MAX_FILES_PER_ROLE,
        user_input=user_input[:500],
        tech_stack=stack_str,
        context_summary=context_summary[:1000],
        extra=f"Constraint: {extra_instruction}" if extra_instruction else "",
    )
    try:
        resp = planner.invoke([HumanMessage(content=prompt)])
        content = str(resp.content)
        m = re.search(r"\[.*?\]", content, re.DOTALL)
        if m:
            files = _json.loads(m.group(0))
            return [
                f for f in files
                if isinstance(f, dict) and "filename" in f
            ][:_MAX_FILES_PER_ROLE]
    except Exception as exc:
        logger.warning("File planning failed for %s: %s", role, exc)
    return []


def _generate_one_file(
    filename: str,
    description: str,
    language: str,
    agent: AgentConfig,
    ollama_base_url: str,
    tech_stack: list[str] | None,
    extra_instruction: str = "",
    context_snippet: str = "",
) -> str:
    """Tạo nội dung một file bằng coding model, cắt fence bên ngoài và loại vòng lặp."""
    llm = ChatOllama(
        model=agent.model,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=_get_num_ctx(agent.model),
        request_timeout=float(OLLAMA_REQUEST_TIMEOUT),
    )
    stack_str = ", ".join(tech_stack) if tech_stack else "not specified"
    ctx_block  = f"\n\nReview context (previous agents):\n{context_snippet}" if context_snippet else ""
    prompt = _FILE_GEN_PROMPT_TMPL.format(
        filename=filename,
        description=description,
        language=language,
        tech_stack=stack_str,
        extra=extra_instruction,
    ) + ctx_block
    try:
        resp = llm.invoke([
            SystemMessage(content=(
                COMMON_AGENT_RULES
                + "\n\n---\n\n"
                + agent.system_prompt
                + "\n\nWhen generating a single file, output only the requested file content. "
                  "Do not repeat other sections or produce a full project overview."
            )),
            HumanMessage(content=prompt),
        ])
        result = str(resp.content)
        result = _strip_code_fence(result, language)   # remove nested fences
        result = _deloop(result)                        # remove infinite loops
        return result
    except Exception as exc:
        logger.warning("File gen failed for %s (%s): %s", agent.name, filename, exc)
        return f"// Error generating {filename}: {exc}"


def _generate_artifacts_multi_turn(
    role: str,
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
    user_input: str,
    tech_stack: list[str] | None,
    extra_instruction: str = "",
) -> str:
    """
    Sinh code/artifact theo 2 pha:
    1. Lập kế hoạch — lấy danh sách file dùng CODING_PLANNER_MODEL (granite3.3:2b).
    2. Loop — tạo từng file riêng bằng coding model với context nhỏ và tập trung.
    Trả về markdown tổng hợp với ### FILE: sections.
    """
    context_summary = context[-1200:]
    files = _plan_code_files(
        role, agent, ollama_base_url, user_input, tech_stack, context_summary, extra_instruction
    )

    if not files:
        logger.warning("No files planned for %s, fallback to single call", role)
        return _call_agent(agent, ollama_base_url, context[-2000:])

    logger.info(
        "Planned %d files for %s: %s",
        len(files), role, [f.get("filename") for f in files],
    )

    parts: list[str] = [f"## {agent.name}\n"]
    # Review roles benefit from seeing previous agent outputs
    _review_roles = {"tech_lead", "devsecops"}
    ctx_snippet = context[-1500:] if role in _review_roles else ""
    for file_info in files:
        filename    = file_info.get("filename", "unknown")
        description = file_info.get("description", "")
        language    = file_info.get("language", "text")
        logger.info("Generating file %s for role=%s", filename, role)
        file_content = _generate_one_file(
            filename, description, language, agent, ollama_base_url, tech_stack,
            extra_instruction, ctx_snippet,
        )
        parts.append(f"\n### FILE: {filename}\n```{language}\n{file_content}\n```\n")

    return "\n".join(parts)


def _call_agent(
    agent: AgentConfig,
    ollama_base_url: str,
    context: str,
) -> str:
    """Gọi LLM với system prompt của agent + context đã tổng hợp.

    Nếu model trả về output rỗng hoặc quá ngắn (< 30 ký tự), thực hiện
    retry với context rút gọn (1 500 ký tự cuối) để xử lý các model nhỏ
    có context window hạn chế.
    """
    llm = ChatOllama(
        model=agent.model,
        base_url=ollama_base_url,
        temperature=0.1,
        num_ctx=_get_num_ctx(agent.model),
        request_timeout=float(OLLAMA_REQUEST_TIMEOUT),
    )
    # Compute once — reused in both the primary call and the retry branch.
    is_reasoning = any(m in agent.model.lower() for m in _REASONING_MODELS)

    # Build system message once — reused in primary call and retry to avoid duplication.
    sys_msg  = SystemMessage(content=COMMON_AGENT_RULES + "\n\n---\n\n" + agent.system_prompt)
    response = llm.invoke([sys_msg, HumanMessage(content=context)])
    result   = str(response.content)

    # Strip chain-of-thought blocks from reasoning models.
    if is_reasoning:
        result = _strip_thinking(result)

    if len(result.strip()) < 30:
        logger.warning(
            "Agent '%s' (model=%s) trả về output rất ngắn (%d chars) — retry với context rút gọn.",
            agent.name, agent.model, len(result.strip()),
        )
        trimmed = context[-1500:] if len(context) > 1500 else context
        response = llm.invoke([sys_msg, HumanMessage(content=trimmed)])
        result = str(response.content)
        if is_reasoning:
            result = _strip_thinking(result)

    return result


# ── Context builder — shared by _build_node and run_single_step ──────────────────


def _build_context_parts(
    role: str,
    agent: AgentConfig,
    user_input: str,
    step_outputs: dict[str, str],
    tech_stack: list[str] | None,
    dep_max_chars: int,
    rag_enabled: bool = False,
    rag_api_url: str = "",
    project: str | None = None,
    rag_top_k: int = RAG_TOP_K,
    extra_context: str | None = None,
) -> list[str]:
    """
    Lắp ráp danh sách context-parts cho bất kỳ lần gọi agent nào.
    Tập trung toàn bộ logic dùng chung giữa _build_node và run_single_step, đảm bảo
    cả hai đường code đều chèn định danh context giống nhau — bao gồm gợi ý ngày
    hiện tại cho các role lập kế hoạch và gợi ý kiểu DB cho DBA/DA.
    """
    parts: list[str] = [
        f"## Mục tiêu kinh doanh / Yêu cầu đầu vào\n{user_input}"
    ]

    # Current date — planning roles need this to avoid past-dated timelines.
    if role in {"pm", "ba", "sa", "ta"}:
        now = datetime.now()
        parts.append(
            f"\n## Ngày hiện tại\n"
            f"{now.strftime('%d/%m/%Y')} (tháng {now.month} năm {now.year}). "
            f"Mọi timeline, sprint, milestone PHẢI bắt đầu từ ngày này trở về sau. "
            f"Không dùng bất kỳ ngày nào trong quá khứ."
        )

    if tech_stack:
        stack_lines = "\n".join(f"- {item}" for item in tech_stack)
        parts.append(
            f"\n## Công nghệ bắt buộc\n"
            f"Các công nghệ sau BẮT BUỘC phải sử dụng. Không đề xuất lựa chọn khác "
            f"trừ khi được yêu cầu rõ ràng.\n{stack_lines}"
        )

    for dep in agent.depends_on:
        if dep in step_outputs:
            dep_name = AGENTS[dep].name
            parts.append(
                f"\n## Kết quả {dep_name}\n{_truncate(step_outputs[dep], dep_max_chars)}"
            )

    if extra_context:
        parts.append(f"\n## Context bổ sung\n{extra_context}")

    if rag_enabled and rag_api_url:
        _hint = agent.rag_query_hint or agent.name
        rag_text = _query_rag(rag_api_url, f"{_hint}: {user_input}", project, rag_top_k)
        if rag_text:
            parts.append(f"\n## Context từ Knowledge Base\n{_truncate(rag_text)}")

    # DB-type hints steer DBA/DA to the correct schema dialect.
    if role in {"dba", "da"} and tech_stack:
        db_type = _detect_db_type(tech_stack)
        if role == "dba":
            if db_type == "nosql":
                parts.append(
                    "\n## DB Schema: NoSQL Only\n"
                    "Tech stack chỉ dùng NoSQL (MongoDB). Tạo Mongoose schema files (.ts). "
                    "KHÔNG tạo SQL DDL hay CREATE TABLE. "
                    "Dùng embedded documents và arrays thay cho JOINs/FK."
                )
            elif db_type == "sql":
                parts.append(
                    "\n## DB Schema: SQL Relational\n"
                    "Tech stack dùng relational DB. Tạo SQL DDL với CREATE TABLE, "
                    "FOREIGN KEY relationships, và CREATE INDEX."
                )
            elif db_type == "both":
                parts.append(
                    "\n## DB Schema: Mixed (SQL + NoSQL)\n"
                    "Tech stack dùng cả SQL và NoSQL. Tạo SQL DDL schema VÀ "
                    "Mongoose models riêng cho từng data store."
                )
        elif role == "da":
            if db_type == "nosql":
                parts.append(
                    "\n## Data Analysis: NoSQL Only\n"
                    "Tech stack chỉ dùng MongoDB NoSQL. Sử dụng aggregation pipeline "
                    "($match, $group, $project, $lookup). "
                    "Nếu cần cross-collection analysis: BE export CSV trước, DA đọc CSV bằng Python/pandas. "
                    "KHÔNG dùng SQL SELECT/FROM/WHERE."
                )
            elif db_type == "sql":
                parts.append(
                    "\n## Data Analysis: SQL\n"
                    "Tech stack dùng relational DB. Phân tích bằng SQL queries với "
                    "GROUP BY, window functions, aggregates, CTEs."
                )

    return parts


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
        tech_stack = state.get("tech_stack")
        # Giới hạn chars mỗi dep cho các role có nhiều dependencies
        dep_max_chars = _PER_DEP_CHARS.get(role, MAX_PREV_OUTPUT_CHARS)

        # 1. Tổng hợp context
        context_parts = _build_context_parts(
            role=role,
            agent=agent,
            user_input=state["user_input"],
            step_outputs=step_outputs,
            tech_stack=tech_stack,
            dep_max_chars=dep_max_chars,
            rag_enabled=state.get("rag_enabled", False),
            rag_api_url=state.get("rag_api_url", ""),
            project=state.get("project"),
            rag_top_k=state.get("rag_top_k", RAG_TOP_K),
        )

        # Tính extra_instruction cho _generate_artifacts_multi_turn
        extra_instruction = _compute_extra_instruction(role, tech_stack)

        context = "\n".join(context_parts)
        ollama_url = state.get("ollama_base_url", OLLAMA_BASE_URL)

        # 3. Gọi LLM
        error_msg: str | None = None
        try:
            if role in _ARTIFACT_ROLES:
                # Sinh code theo từng file riêng biệt (planner + loop)
                output = _generate_artifacts_multi_turn(
                    role, agent, ollama_url, context,
                    state["user_input"], tech_stack, extra_instruction,
                )
            else:
                output = _call_agent(agent, ollama_url, context)
        except Exception as exc:
            logger.error("Agent '%s' gặp lỗi: %s", role, exc)
            output = f"[LỖI trong {role}] {exc}"
            error_msg = f"Bước '{role}' thất bại: {exc}"

        logger.info("Step %d | %s | output_len=%d",
                    agent.step_id, agent.name, len(output))

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
    để đồ thị phản ánh luồng SDLC theo mô hình Waterfall/Agile truyền thống.
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
# dù nhiều request đến cùng lúc khi khởi động.

_workflow = None
_workflow_lock = threading.Lock()


def get_workflow():
    global _workflow
    if _workflow is None:
        with _workflow_lock:
            if _workflow is None:  # kiểm tra lại bên trong lock
                _workflow = build_workflow()
    return _workflow


# ── Chạy từng bước đơn lẻ (dùng bởi endpoint /agent/{role}) ────────────────

def run_single_step(
    role: str,
    user_input: str,
    project: str | None = None,
    extra_context: str | None = None,
    prev_outputs: dict[str, str] | None = None,
    tech_stack: list[str] | None = None,
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

    # Apply same per-dep char limits as the workflow node to avoid context overflow.
    dep_max_chars = _PER_DEP_CHARS.get(role, MAX_PREV_OUTPUT_CHARS)
    context_parts = _build_context_parts(
        role=role,
        agent=agent,
        user_input=user_input,
        step_outputs=step_outputs,
        tech_stack=tech_stack,
        dep_max_chars=dep_max_chars,
        rag_enabled=rag_enabled,
        rag_api_url=rag_api_url,
        project=project,
        rag_top_k=rag_top_k,
        extra_context=extra_context,
    )
    context = "\n".join(context_parts)
    if role in _ARTIFACT_ROLES:
        extra_instruction = _compute_extra_instruction(role, tech_stack)
        return _generate_artifacts_multi_turn(
            role,
            agent,
            ollama_base_url,
            context,
            user_input,
            tech_stack,
            extra_instruction,
        )
    return _call_agent(agent, ollama_base_url, context)
