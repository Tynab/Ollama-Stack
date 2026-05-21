"""
title: YAN SDLC Agent Workflow
author: YAN
description: >
  Khởi chạy SDLC Agent Workflow theo thứ tự PM → BA → SA → QA → DevOps → BE → FE → QA exec → DevOps release → PM closure.
  Hỗ trợ chạy full workflow hoặc từng agent đơn lẻ.
required_open_webui_version: 0.3.0
requirements: requests
version: 1.0.0
"""

import time

import requests
from pydantic import BaseModel, Field

_VALID_ROLES = [
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

_ROLE_DESCRIPTIONS = {
    "pm": "PM Agent — Project Intake & Planning",
    "ba": "BA Agent — Requirement Analysis",
    "sa": "SA Agent — Solution Architecture",
    "qa_shiftleft": "QA Agent — Shift-left Review",
    "devops_env": "DevOps Agent — Environment & Pipeline Planning",
    "be": "BE Agent — Backend Implementation",
    "fe": "FE Agent — Frontend Implementation",
    "qa_exec": "QA Agent — Test Execution & Bug Report",
    "devops_release": "DevOps Agent — Release & Deploy",
    "pm_closure": "PM Agent — Sprint / Release Closure",
}


class Tools:
    class Valves(BaseModel):
        agent_api_url: str = Field(
            default="http://agent-api:8091",
            description="URL của agent-api service (nội bộ Docker network)",
        )
        timeout: int = Field(
            default=600,
            description="Timeout cho single-step request (giây). Full workflow dùng polling.",
        )
        poll_interval: int = Field(
            default=15,
            description="Khoảng cách giữa các lần poll workflow status (giây)",
        )
        poll_max_attempts: int = Field(
            default=120,
            description="Số lần poll tối đa trước khi trả về timeout (120 × 15s = 30 phút)",
        )
        rag_enabled: bool = Field(
            default=True,
            description="Có query RAG knowledge base cho mỗi agent step không",
        )
        rag_top_k: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Số kết quả RAG trả về cho mỗi agent",
        )
        default_project: str | None = Field(
            default=None,
            description="RAG project mặc định. Để trống để search tất cả projects.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _base_url(self) -> str:
        return self.valves.agent_api_url.rstrip("/")

    def _handle_connection_error(self, exc: Exception) -> str:
        return (
            f"❌ Không kết nối được agent-api tại {self._base_url()}.\n"
            f"Kiểm tra service có đang chạy không: `docker ps | grep agent-api`\n"
            f"Chi tiết: {exc}"
        )

    # ── Tools ─────────────────────────────────────────────────────────────────

    def run_sdlc_workflow(
        self,
        user_input: str,
        project: str | None = None,
    ) -> str:
        """
        Chạy toàn bộ SDLC Workflow (PM → BA → SA → QA → DevOps → BE → FE → QA exec → Deploy → PM closure).
        Mỗi agent đọc output của agent trước và bổ sung RAG context từ knowledge base.

        :param user_input: Mô tả business goal / feature / project cần phân tích và implement
        :param project: Tên project RAG cần filter (ví dụ: 'yanlib'). Để trống để search tất cả.
        :return: Tóm tắt kết quả từng step hoặc link để xem chi tiết
        """
        resolved_project = project or self.valves.default_project
        payload: dict = {
            "user_input": user_input,
            "rag_enabled": self.valves.rag_enabled,
            "rag_top_k": self.valves.rag_top_k,
        }
        if resolved_project:
            payload["project"] = resolved_project

        # Submit workflow
        try:
            resp = requests.post(
                f"{self._base_url()}/workflow/run",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except requests.exceptions.HTTPError as exc:
            return f"❌ agent-api lỗi HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Lỗi khi submit workflow: {exc}"

        workflow_id: str = data["workflow_id"]
        result_lines = [
            f"✅ **Workflow đã được khởi chạy**",
            f"- **ID:** `{workflow_id}`",
            f"- **Project:** {resolved_project or 'all'}",
            f"- **Steps:** {' → '.join(_VALID_ROLES)}",
            f"\n⏳ Đang chạy... (có thể mất 15–45 phút tùy model và độ phức tạp)",
        ]

        # Poll for completion
        for attempt in range(self.valves.poll_max_attempts):
            time.sleep(self.valves.poll_interval)
            try:
                poll_resp = requests.get(
                    f"{self._base_url()}/workflow/{workflow_id}",
                    timeout=30,
                )
                poll_resp.raise_for_status()
                wf = poll_resp.json()
            except Exception as exc:
                result_lines.append(
                    f"\n⚠️ Poll attempt {attempt + 1} failed: {exc}")
                continue

            status = wf.get("status", "unknown")
            completed = wf.get("completed_steps", [])
            result_lines.append(
                f"📊 Attempt {attempt + 1}: status={status} | completed={completed}"
            )

            if status == "completed":
                result_lines.append("\n---\n## ✅ Workflow Hoàn thành\n")
                step_outputs: dict = wf.get("step_outputs", {})
                for role in _VALID_ROLES:
                    if role in step_outputs:
                        role_name = _ROLE_DESCRIPTIONS.get(role, role)
                        preview = step_outputs[role][:400].replace("\n", " ")
                        result_lines.append(f"### {role_name}\n{preview}...\n")
                result_lines.append(
                    f"\n💡 Dùng `get_workflow_result('{workflow_id}')` để xem output đầy đủ từng step."
                )
                return "\n".join(result_lines)

            if status == "failed":
                error = wf.get("error", "unknown error")
                result_lines.append(f"\n❌ **Workflow thất bại:** {error}")
                return "\n".join(result_lines)

        result_lines.append(
            f"\n⏰ Timeout sau {self.valves.poll_max_attempts} lần poll. "
            f"Dùng `get_workflow_result('{workflow_id}')` để kiểm tra sau."
        )
        return "\n".join(result_lines)

    def run_agent_step(
        self,
        role: str,
        user_input: str,
        extra_context: str | None = None,
        project: str | None = None,
    ) -> str:
        """
        Chạy một agent step đơn lẻ (không cần chạy full workflow).
        Dùng để test từng agent hoặc bổ sung output thủ công.

        :param role: Tên agent role. Hợp lệ: pm, ba, sa, qa_shiftleft, devops_env, be, fe, qa_exec, devops_release, pm_closure
        :param user_input: Business goal / context đầu vào cho agent
        :param extra_context: Context bổ sung (ví dụ: output từ step trước dán vào)
        :param project: RAG project filter. Để trống để search tất cả.
        :return: Output của agent role được chọn
        """
        if role not in _VALID_ROLES:
            return (
                f"❌ Role không hợp lệ: `{role}`\n"
                f"Các role hợp lệ: {', '.join(_VALID_ROLES)}"
            )

        resolved_project = project or self.valves.default_project
        payload: dict = {
            "user_input": user_input,
            "rag_enabled": self.valves.rag_enabled,
            "rag_top_k": self.valves.rag_top_k,
        }
        if resolved_project:
            payload["project"] = resolved_project
        if extra_context:
            payload["extra_context"] = extra_context

        try:
            resp = requests.post(
                f"{self._base_url()}/agent/{role}",
                json=payload,
                timeout=self.valves.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except requests.exceptions.Timeout:
            return (
                f"❌ Timeout sau {self.valves.timeout}s. "
                "Tăng timeout trong Valves hoặc dùng run_sdlc_workflow cho async."
            )
        except requests.exceptions.HTTPError as exc:
            return f"❌ agent-api lỗi HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Lỗi: {exc}"

        role_name = data.get("name", role)
        model = data.get("model", "unknown")
        output = data.get("output", "Không có output.")

        return (
            f"## {role_name}\n"
            f"**Model:** `{model}`  |  **Project:** {resolved_project or 'all'}\n\n"
            f"---\n\n{output}"
        )

    def get_workflow_result(
        self,
        workflow_id: str,
        role: str | None = None,
    ) -> str:
        """
        Lấy kết quả của một workflow đã chạy theo ID.
        Có thể filter để chỉ xem output của một role cụ thể.

        :param workflow_id: ID của workflow (trả về từ run_sdlc_workflow)
        :param role: Nếu muốn xem output của một step cụ thể (ví dụ: 'ba', 'sa'). Để trống để xem tóm tắt tất cả.
        :return: Kết quả workflow hoặc output của step được chọn
        """
        try:
            resp = requests.get(
                f"{self._base_url()}/workflow/{workflow_id}",
                timeout=30,
            )
            resp.raise_for_status()
            wf = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 404:
                return f"❌ Workflow `{workflow_id}` không tìm thấy. ID có thể đã hết hạn."
            return f"❌ agent-api lỗi HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Lỗi: {exc}"

        status = wf.get("status", "unknown")
        completed = wf.get("completed_steps", [])
        step_outputs: dict = wf.get("step_outputs", {})

        if role:
            if role not in _VALID_ROLES:
                return f"❌ Role không hợp lệ: `{role}`. Hợp lệ: {', '.join(_VALID_ROLES)}"
            if role not in step_outputs:
                return (
                    f"⚠️ Step `{role}` chưa có output.\n"
                    f"Workflow status: {status} | Completed: {completed}"
                )
            role_name = _ROLE_DESCRIPTIONS.get(role, role)
            return f"## {role_name}\n\n{step_outputs[role]}"

        # Summary of all steps
        lines = [
            f"## Workflow `{workflow_id}`",
            f"- **Status:** {status}",
            f"- **Project:** {wf.get('project') or 'all'}",
            f"- **Completed steps:** {len(completed)}/{len(_VALID_ROLES)}",
            f"- **Created:** {wf.get('created_at', 'N/A')}",
            f"- **Completed at:** {wf.get('completed_at', 'N/A')}",
        ]

        if wf.get("error"):
            lines.append(f"- **Error:** {wf['error']}")

        lines.append("\n---\n### Step Outputs\n")
        for r in _VALID_ROLES:
            if r in step_outputs:
                role_name = _ROLE_DESCRIPTIONS.get(r, r)
                preview = step_outputs[r][:300].replace("\n", " ")
                lines.append(f"**{role_name}**\n{preview}...\n")
            else:
                lines.append(
                    f"**{_ROLE_DESCRIPTIONS.get(r, r)}** — _(not run)_\n")

        lines.append(
            f"\n💡 Dùng `get_workflow_result('{workflow_id}', role='ba')` để xem full output từng step."
        )
        return "\n".join(lines)

    def list_agent_roles(self) -> str:
        """
        Liệt kê tất cả agent roles hợp lệ, model được dùng, và thứ tự chạy trong SDLC workflow.

        :return: Danh sách agent roles
        """
        try:
            resp = requests.get(f"{self._base_url()}/agents", timeout=15)
            resp.raise_for_status()
            agents = resp.json()
        except requests.exceptions.ConnectionError as exc:
            return self._handle_connection_error(exc)
        except Exception as exc:
            return f"❌ Lỗi: {exc}"

        rows = []
        for role in _VALID_ROLES:
            info = agents.get(role, {})
            rows.append(
                f"| **{info.get('step_id', '?')}** | `{role}` | {info.get('name', role)} | `{info.get('model', '?')}` |"
            )

        return (
            "## SDLC Agent Roles\n\n"
            "| Step | Role | Name | Model |\n"
            "|------|------|------|-------|\n"
            + "\n".join(rows)
        )
