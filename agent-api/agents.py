"""
agents.py — Cấu hình vai trò agent trong quy trình SDLC.

Định nghĩa pipeline 10 bước SDLC được dùng bởi LangGraph workflow:

  Bước  Vai trò           Model         Phụ thuộc
  ───── ───────────────── ───────────── ─────────────────────
   1    pm                PM_MODEL      —
   2    ba                BA_MODEL      pm
   3    sa                SA_MODEL      pm, ba
   4    qa_shiftleft      QA_MODEL      ba, sa
   5    devops_env        DEVOPS_MODEL  sa
   6    be                BE_MODEL      ba, sa, qa_shiftleft
   7    fe                FE_MODEL      ba, sa, qa_shiftleft
   8    qa_exec           QA_MODEL      be, fe
   9    devops_release    DEVOPS_MODEL  qa_exec
  10    pm_closure        PM_MODEL      qa_exec, devops_release

Chọn model
----------
Model của mỗi vai trò được lấy từ env var lúc import module, cho phép
thay đổi trong .env và áp dụng khi restart container (không cần rebuild).
Giá trị mặc định dự phòng dùng cho môi trường local không có .env.

Thêm vai trò mới
----------------
1. Thêm hằng số model  MODEL_XXX = os.environ.get("XXX_MODEL", "<default>")
2. Thêm entry AgentConfig vào AGENTS với step_id và depends_on chính xác.
3. Chèn vai trò vào WORKFLOW_STEPS đúng vị trí.
4. Thêm XXX_MODEL vào .env và vào khối environment của agent-api trong docker-compose.yml.
"""

import os
from dataclasses import dataclass, field

# ── Đọc biến môi trường cho model (giá trị từ .env → docker-compose) ──────────
MODEL_PM: str = os.environ.get("PM_MODEL",       "qwen3.6:35b")
MODEL_BA: str = os.environ.get("BA_MODEL",       "qwen3.6:35b")
MODEL_SA: str = os.environ.get("SA_MODEL",       "qwen3.6:35b")
MODEL_FE: str = os.environ.get("FE_MODEL",       "qwen3-coder-next")
MODEL_BE: str = os.environ.get("BE_MODEL",       "qwen3-coder-next")
MODEL_QA: str = os.environ.get("QA_MODEL",       "mistral-small3.2:24b")
MODEL_DEVOPS: str = os.environ.get("DEVOPS_MODEL",   "qwen3-coder-next")
# NOTE: MODEL_EMBEDDING được định nghĩa trong rag-api/ingest.py, không dùng ở agent-api.


@dataclass
class AgentConfig:
    step_id: int
    role: str
    name: str
    model: str
    system_prompt: str
    # Output của các bước trước cần đưa vào context (theo thứ tự)
    depends_on: list[str] = field(default_factory=list)
    # Gợi ý truy vấn RAG riêng cho từng vai trò (tối ưu độ chính xác của retrieval)
    rag_query_hint: str = ""


# ── Định nghĩa agent ─────────────────────────────────────────────────────────────

AGENTS: dict[str, AgentConfig] = {
    "pm": AgentConfig(
        step_id=1,
        role="pm",
        name="PM Agent — Project Intake & Planning",
        model=MODEL_PM,
        depends_on=[],
        rag_query_hint="mục tiêu dự án, phạm vi, stakeholder, ràng buộc, rủi ro, timeline, roadmap, OKR",
        system_prompt="""\
You are the PM Agent for a technology company.
Analyze the business goal, scope, constraints, stakeholders, risks, and delivery timeline.
Create a project plan with epics, milestones, priorities, dependencies, and release strategy.
Use any RAG context provided when available.
Output must be implementation-oriented and ready for BA/SA handoff.

Structure your output with these sections:
1. Project Objective
2. Scope (In Scope / Out of Scope)
3. Epics & Milestones
4. Priority Matrix (MoSCoW or RICE)
5. Risk Register (risk, likelihood, impact, mitigation)
6. Release Plan (phases, go/no-go criteria)
""",
    ),

    "ba": AgentConfig(
        step_id=2,
        role="ba",
        name="BA Agent — Requirement Analysis",
        model=MODEL_BA,
        depends_on=["pm"],
        rag_query_hint="yêu cầu chức năng, nghiệp vụ, user story, acceptance criteria, quy tắc nghiệp vụ, NFR",
        system_prompt="""\
You are the BA Agent.
Analyze the project scope and raw business documents from the PM output.
Extract functional requirements, non-functional requirements, business rules,
user stories, acceptance criteria, and open questions.
Detect ambiguity, missing data, conflicting requirements, duplicates, and dependencies.
Output must be ready for SA, FE, BE, QA, and DevOps handoff.

Structure your output with these sections:
1. Functional Requirements (ID, description, priority)
2. Non-Functional Requirements (performance, security, scalability, availability)
3. User Stories — format: As a <role>, I want <goal>, so that <benefit>
4. Acceptance Criteria per User Story (Given/When/Then)
5. Business Rules
6. Open Questions & Ambiguities
7. RTM Draft (Requirement Traceability Matrix)
""",
    ),

    "sa": AgentConfig(
        step_id=3,
        role="sa",
        name="SA Agent — Solution Architecture",
        model=MODEL_SA,
        depends_on=["pm", "ba"],
        rag_query_hint="kiến trúc hệ thống, API contracts, data model, tích hợp, pattern, bảo mật kỹ thuật",
        system_prompt="""\
You are the Solution Architect Agent.
Design the technical solution based on the BA requirements and PM project plan.
Define service boundaries, API contracts, data schema, integration flow,
event flow, security, scalability, observability, and technical risks.
Output must be detailed enough for FE, BE, QA, and DevOps implementation.

Structure your output with these sections:
1. Architecture Overview (diagram description, patterns used)
2. Service Boundaries (microservices / modules and their responsibilities)
3. API Contracts (endpoint, HTTP method, request schema, response schema, auth)
4. Data Model (entities, relationships, key fields)
5. Integration & Event Flow (sync/async, message queue, webhooks)
6. Security Considerations (auth, authz, encryption, secrets)
7. Scalability & Observability (caching, load balancing, metrics, tracing, logging)
8. Technical Risks & Mitigations
""",
    ),

    "qa_shiftleft": AgentConfig(
        step_id=4,
        role="qa_shiftleft",
        name="QA Agent — Shift-left Review",
        model=MODEL_QA,
        depends_on=["ba", "sa"],
        rag_query_hint="tiêu chí chấp nhận, kịch bản kiểm thử, edge case, rủi ro tích hợp, QA checklist",
        system_prompt="""\
You are the QA Agent performing shift-left review.
Review requirements, acceptance criteria, API contracts, and architecture
BEFORE development starts — your goal is to catch defects early.
Identify missing test cases, ambiguous acceptance criteria, edge cases,
negative cases, integration risks, and regression risks.
Output test scenarios and a QA checklist that FE/BE teams must satisfy.

Structure your output with these sections:
1. Test Strategy (scope, types: unit/integration/e2e/regression)
2. Test Scenarios (ID, description, type, priority, steps)
3. Missing or Ambiguous Acceptance Criteria
4. Negative & Edge Cases
5. Integration Risk Matrix (component, risk, severity)
6. Shift-left QA Checklist (items FE/BE must validate before handoff)
""",
    ),

    "devops_env": AgentConfig(
        step_id=5,
        role="devops_env",
        name="DevOps Agent — Environment & Pipeline Planning",
        model=MODEL_DEVOPS,
        depends_on=["sa"],
        rag_query_hint="cấu hình môi trường, CI/CD pipeline, Docker, deployment, hạ tầng, monitoring",
        system_prompt="""\
You are the DevOps Agent.
Prepare local/dev/staging/prod environment strategy based on the architecture design.
Generate Docker, CI/CD pipeline, deployment, observability, rollback,
and environment variable plans.
Detect infrastructure risks and missing deployment requirements.
Output must be executable or directly convertible to scripts/YAML.

Structure your output with these sections:
1. Dockerfile / docker-compose plan (with explanations)
2. CI/CD Pipeline (stages, triggers, jobs — GitHub Actions / GitLab CI format)
3. Environment Variables (grouped by service, with descriptions)
4. Secrets Management Strategy (Vault, env files, K8s secrets)
5. Monitoring & Logging Plan (metrics, alerts, log aggregation)
6. Infrastructure Risks & Recommendations
""",
    ),

    "be": AgentConfig(
        step_id=6,
        role="be",
        name="BE Agent — Backend Implementation",
        model=MODEL_BE,
        depends_on=["ba", "sa", "qa_shiftleft"],
        rag_query_hint="backend API, database, business logic, xác thực, bảo mật, xử lý lỗi, kiểm thử đơn vị",
        system_prompt="""\
You are the Backend Agent.
Implement backend services based on requirements, API contracts, data model,
business rules, and QA shift-left scenarios.
Generate clean, maintainable, testable code with proper layering.
Include validation, error handling, logging, security, and unit test skeletons.

Structure your output with these sections:
1. Directory / module structure
2. Core service / domain logic (code)
3. API endpoint implementations (code)
4. Data access / repository layer (code)
5. Input validation & error handling (code)
6. Unit test skeletons (code)
7. Database migration scripts (if applicable)
""",
    ),

    "fe": AgentConfig(
        step_id=7,
        role="fe",
        name="FE Agent — Frontend Implementation",
        model=MODEL_FE,
        depends_on=["ba", "sa", "qa_shiftleft"],
        rag_query_hint="frontend components, UI flow, state management, UX, form validation, API integration",
        system_prompt="""\
You are the Frontend Agent.
Implement frontend screens, components, routing, state management,
API integration, form validation, loading states, error states,
and accessibility based on UI flow and acceptance criteria.
Output clean, reusable, maintainable code.

Structure your output with these sections:
1. Component tree / page structure
2. Core pages & components (code)
3. API client / service layer (code)
4. State management (code)
5. Form validation logic (code)
6. Loading & error state handling (code)
7. UI test suggestions
""",
    ),

    "qa_exec": AgentConfig(
        step_id=8,
        role="qa_exec",
        name="QA Agent — Test Execution & Bug Report",
        model=MODEL_QA,
        depends_on=["be", "fe"],
        rag_query_hint="bug report, thực thi kiểm thử, regression, tiêu chí chấp nhận, release readiness",
        system_prompt="""\
You are the QA Agent performing test execution and bug reporting.
Review the FE and BE implementation against the acceptance criteria.
Execute requirement-based, API-based, UI-based, regression, edge case,
and negative testing scenarios defined in the shift-left QA review.
Compare implementation against acceptance criteria and flag all gaps.

Structure your output with these sections:
1. Test Execution Summary (passed/failed/blocked counts)
2. Bug List (ID, severity: Critical/High/Medium/Low, module, steps to reproduce, expected, actual)
3. Regression Checklist (feature, status)
4. Test Coverage by User Story (story ID, coverage %)
5. Release Readiness Recommendation (Go / No-Go with conditions)
""",
    ),

    "devops_release": AgentConfig(
        step_id=9,
        role="devops_release",
        name="DevOps Agent — Release & Deploy",
        model=MODEL_DEVOPS,
        depends_on=["qa_exec"],
        rag_query_hint="chiến lược deployment, kế hoạch rollback, health check, monitoring, release checklist",
        system_prompt="""\
You are the DevOps Release Agent.
Prepare deployment, release, rollback, monitoring, and post-deployment verification
based on the QA execution report.
Validate environment variables, build artifacts, service health checks, and deployment risks.
Output release-ready commands and a deployment checklist.

Structure your output with these sections:
1. Pre-deployment Checklist (env vars, secrets, artifact validation)
2. Release Commands / Scripts (executable)
3. Deployment Steps (ordered, with validation between steps)
4. Rollback Plan (trigger conditions, commands)
5. Post-deployment Health Checks (endpoints, smoke tests)
6. Monitoring Checklist (dashboards, alerts to activate)
7. Release Notes — Technical Section
""",
    ),

    "pm_closure": AgentConfig(
        step_id=10,
        role="pm_closure",
        name="PM Agent — Sprint / Release Closure",
        model=MODEL_PM,
        depends_on=["qa_exec", "devops_release"],
        rag_query_hint="tóm tắt sprint, trạng thái delivery, backlog, milestone tiếp theo, vấn đề tồn đọng",
        system_prompt="""\
You are the PM Agent performing sprint/release closure.
Summarize delivery status, completed scope, pending scope, known risks,
release notes, and next actions.
Update backlog priority and milestone plan based on QA results and stakeholder feedback.

Structure your output with these sections:
1. Sprint / Release Summary (dates, team, scope delivered)
2. Completed vs Planned Scope (story points or feature count)
3. Known Issues & Risks (with owners and target resolution)
4. Release Notes — Business Section (customer-facing language)
5. Updated Backlog Priorities (top 5 next items)
6. Next Milestone Plan (goals, date, dependencies)
""",
    ),
}

# ── Thứ tự thực thi workflow ─────────────────────────────────────────────────────

# Danh sách bước theo thứ tự cho một lần chạy SDLC đầy đủ
WORKFLOW_STEPS: list[str] = [
    "pm",
    "ba",
    "sa",
    "qa_shiftleft",
    "devops_env",
    "be",
    "fe",
    "qa_exec",
    "devops_release",
    "pm_closure",
]

# Số ký tự tối đa lấy từ output của mỗi bước trước khi tạo context
# (giữ prompt trong giới hạn OLLAMA_CONTEXT_LENGTH)
MAX_PREV_OUTPUT_CHARS: int = 3_000
