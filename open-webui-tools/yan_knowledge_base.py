"""
title: YAN Knowledge Base
author: YAN
description: Query nội bộ — hỏi đáp từ các tài liệu PRD, spec, architecture đã ingest vào Qdrant qua rag-api.
required_open_webui_version: 0.3.0
requirements: requests
version: 1.0.0
"""

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        rag_api_url: str = Field(
            default="http://rag-api:8090",
            description="URL của rag-api service (nội bộ Docker network)",
        )
        timeout: int = Field(
            default=120,
            description="Timeout cho mỗi request (giây)",
        )
        top_k: int | None = Field(
            default=None,
            ge=1,
            le=20,
            description="Số kết quả RAG trả về. Để trống dùng default của rag-api (RAG_TOP_K env).",
        )
        default_project: str | None = Field(
            default=None,
            description="Project mặc định. Để trống để search tất cả projects.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def ask_internal_docs(self, question: str, project: str | None = None) -> str:
        """
        Hỏi-đáp với tài liệu nội bộ: PRD, spec, architecture, billing, auth, marketplace, v.v.
        Dùng khi cần tìm thông tin trong các tài liệu kỹ thuật của dự án.
        :param question: Câu hỏi cần trả lời
        :param project: Tên project cần query (ví dụ: auth, marketplace, billing). Để trống để search tất cả.
        :return: Câu trả lời kèm tên file nguồn
        """
        resolved_project = project or self.valves.default_project
        payload: dict = {"question": question}
        if resolved_project:
            payload["project"] = resolved_project
        if self.valves.top_k is not None:
            payload["top_k"] = self.valves.top_k

        try:
            resp = requests.post(
                f"{self.valves.rag_api_url}/ask",
                json=payload,
                timeout=self.valves.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return "❌ Không kết nối được rag-api. Kiểm tra service có đang chạy không."
        except requests.exceptions.Timeout:
            return f"❌ rag-api timeout sau {self.valves.timeout}s. Thử lại hoặc tăng timeout trong Valves."
        except requests.exceptions.HTTPError as exc:
            return f"❌ rag-api trả lỗi HTTP {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Lỗi không xác định: {exc}"

        answer = data.get("answer", "Không có câu trả lời.")
        sources = data.get("sources", [])

        if sources:
            seen: set = set()
            unique_files = [
                s["source_file"]
                for s in sources
                # type: ignore[func-returns-value]
                if s.get("source_file") and not (s["source_file"] in seen or seen.add(s["source_file"]))
            ]
            if unique_files:
                answer += "\n\n---\n**Nguồn:** " + " · ".join(unique_files)

        return answer
