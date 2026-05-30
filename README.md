# YAN Ollama Stack

Nền tảng AI cục bộ vận hành hoàn toàn offline: **Ollama** (suy luận LLM), **Qdrant** (vector DB), **Neo4j** (graph DB), **RAG API** (FastAPI), **Agent API** (điều phối SDLC), **Open WebUI** (chat UI), **Watchtower** (tự động cập nhật), và **Deunhealth** (giám sát healthcheck container).

---

## Stack Overview

| Service      | Image                              | Port (host)   | Mô tả                                                  |
|--------------|------------------------------------|---------------|--------------------------------------------------------|
| `ollama`     | `ollama/ollama:latest`             | 11434         | LLM & embedding inference                              |
| `qdrant`     | `qdrant/qdrant:latest`             | 6333 / 6334   | Vector database (REST / gRPC)                          |
| `neo4j`      | `neo4j:5-community`                | 7474 / 7687   | Knowledge graph (Neo4j Browser / Bolt)                 |
| `rag-api`    | build `./rag-api`                  | 8090          | FastAPI RAG service (ingest + ask)                     |
| `agent-api`  | build `./agent-api`                  | 8091          | FastAPI SDLC Agent Orchestrator (14-step SDLC workflow) |
| `open-webui` | `ghcr.io/open-webui/open-webui`    | 8085          | Chat UI kết nối Ollama & Qdrant                        |
| `watchtower` | `containrrr/watchtower`            | —             | Tự động pull & restart image mới nhất                 |
| `deunhealth` | `qmcgaw/deunhealth`                | 9999          | Khởi động lại container khi healthcheck thất bại       |

**Model theo biến môi trường** (cấu hình trong `.env` → mục `SDLC Agent Models`):

| Mục đích                                    | Env var(s)                                                        | Giá trị                   |
|---------------------------------------------|-------------------------------------------------------------------|---------------------------|
| Embedding                                   | `EMBEDDING_MODEL`                                                 | Lấy từ `.env`             |
| Chat — RAG API `/ask`                        | `CHAT_MODEL`                                                      | Lấy từ `.env`             |
| Reasoning agents (BA / PM / SA / TA / DA)   | `BA_MODEL` `PM_MODEL` `SA_MODEL` `TA_MODEL` `DA_MODEL`            | Lấy từ `.env`             |
| Team Lead planning agent                    | `TL_MODEL`                                                        | Lấy từ `.env`             |
| Coding agents (FE / Mobile / BE / DBA / Tech Lead / DevSecOps) | `FE_MODEL` `MOBILE_MODEL` `BE_MODEL` `DBA_MODEL` `TECH_LEAD_MODEL` `DEVSECOPS_MODEL` | Lấy từ `.env` |
| Creative agents (Tester / Designer)         | `TESTER_MODEL` `DESIGNER_MODEL`                                   | Lấy từ `.env`             |

> **Lưu ý:** Đổi `EMBEDDING_MODEL` yêu cầu re-ingest toàn bộ documents (`POST /ingest {"reset": true}`).

---

## Kiến trúc

```
User → Open WebUI (8085)
         │
         ├── yan_knowledge_base.py  →  RAG API (8090)
         │                                ├── Qdrant  (vector search)
         │                                └── Neo4j   (graph enrichment)
         │                                       └── Ollama (embedding + chat)
         │
         └── yan_agent_workflow.py  →  Agent API (8091)
                                           ├── LangGraph SDLC Workflow (14 bước)
                                           │     BA → PM → SA → TA → Designer → Team Lead
                                           │     → FE → Mobile → DBA → BE → DA
                                           │     → Tech Lead → Tester → DevSecOps
                                           └── Ollama (per-role models)
```

### SDLC Workflow (14 bước)

| Bước | Role            | Tên                                          | Model             | Phụ thuộc                              |
|------|-----------------|----------------------------------------------|-------------------|----------------------------------------|
| 1    | `ba`            | BA — Phân tích nghiệp vụ                     | BA_MODEL          | —                                      |
| 2    | `pm`            | PM — Quản lý dự án & Lập kế hoạch           | PM_MODEL          | ba                                     |
| 3    | `sa`            | SA — Kiến trúc giải pháp                    | SA_MODEL          | ba, pm                                 |
| 4    | `ta`            | TA — Kiến trúc kỹ thuật                     | TA_MODEL          | ba, sa                                 |
| 5    | `designer`      | Designer — Thiết kế UI/UX                   | DESIGNER_MODEL    | ba, sa, ta                             |
| 6    | `tl`            | Team Lead — Lập kế hoạch task kỹ thuật      | TL_MODEL          | ba, sa, ta, designer                   |
| 7    | `fe`            | FE — Kỹ thuật Frontend                      | FE_MODEL          | ba, sa, ta, designer, tl               |
| 8    | `mobile`        | Mobile — Phát triển Mobile                  | MOBILE_MODEL      | ba, sa, ta, designer, tl               |
| 9    | `dba`           | DBA — Kiến trúc cơ sở dữ liệu              | DBA_MODEL         | ba, sa, ta, tl                         |
| 10   | `be`            | BE — Triển khai Backend                     | BE_MODEL          | ba, sa, ta, fe, mobile, dba, tl        |
| 11   | `da`            | DA — Phân tích & Báo cáo dữ liệu           | DA_MODEL          | ba, sa, dba                            |
| 12   | `tech_lead`     | Tech Lead — Review code & Tiêu chuẩn        | TECH_LEAD_MODEL   | sa, fe, mobile, be, dba                |
| 13   | `tester`        | Tester — Kiểm thử & Đảm bảo chất lượng    | TESTER_MODEL      | be, fe, mobile, tech_lead, designer    |
| 14   | `devsecops`     | DevSecOps — Hạ tầng, CI/CD & Bảo mật       | DEVSECOPS_MODEL   | sa, ta, tech_lead, tester              |

Mỗi bước nhận output đã rút gọn từ các bước phụ thuộc và có thể bổ sung context từ RAG knowledge base.

### Kiến trúc hybrid RAG

```
/ask request
  ├── Qdrant  → vector search (cosine similarity, top-k chunks)
  └── Neo4j   → graph enrichment (entity co-occurrence traversal)
       └── merge → Ollama (chat model) → answer
```

- Mỗi `/ask` trước tiên lấy top-k chunks từ Qdrant, sau đó Neo4j tìm thêm chunks liên quan qua entity co-occurrence.
- Nếu `GRAPH_ENTITY_EXTRACTION=true`, khi ingest LLM sẽ extract entities từ mỗi document rồi lưu vào Neo4j.

---

## Tính năng mới

### Module-scoped RAG

Khi tổ chức tài liệu theo thư mục con `data/raw/{project}/{module}/`, mỗi chunk sẽ được gán `module` tự động:

```
data/raw/yanlib/auth/auth-prd.md    → module=auth
data/raw/yanlib/billing/schema.md   → module=billing
data/raw/yanlib/spec.md             → module=yanlib (flat file)
```

Sử dụng filter `module` trong `/ask` để giới hạn search:

```bash
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "JWT refresh token flow?", "project": "yanlib", "module": "auth"}' | jq
```

Mỗi `SourceItem` trong response chứa thêm: `module`, `doc_type` (prd/schema/api/architecture/…), `chunk_type` (paragraph/table/code).

### Role-specific RAG queries (rag_query_hint)

Mỗi SDLC agent có `rag_query_hint` riêng để xây dựng câu truy vấn RAG chính xác hơn thay vì dùng nguyên `user_input`. Ví dụ:
- **PM**: tìm *"mục tiêu dự án, phạm vi, stakeholder, ràng buộc, rủi ro, timeline, OKR"*
- **BA**: tìm *"yêu cầu chức năng, user story, acceptance criteria, quy tắc nghiệp vụ"*
- **SA**: tìm *"kiến trúc hệ thống, API contracts, data model, tích hợp, pattern, bảo mật"*

Thiết kế này cải thiện độ chính xác truy xuất cho từng vai trò SDLC mà không cần thay đổi `user_input`.

### Episodic Memory Logging

Mỗi lần workflow hoàn thành, agent-api tự động ghi log JSONL vào:

```
data/memory/episodic/workflow_runs.jsonl
```

Mỗi dòng là một JSON object:

```json
{
  "workflow_id": "a1b2c3d4-...",
  "project": "yanlib",
  "user_input": "Xây dựng checkout flow...",
  "completed_steps": ["ba","pm","sa","ta","designer","tl","fe","mobile","dba","be","da","tech_lead","tester","devsecops"],
  "steps_count": 14,
  "status": "completed",
  "error": null,
  "duration_seconds": 842.5,
  "timestamp": "2025-01-01T12:00:00+00:00"
}
```

File này có thể dùng làm dataset cho fine-tune, phân tích hiệu suất, hoặc làm context lịch sử cho các workflow kế tiếp.

### Input Sanitization

`user_input` ở `/agent/{role}` và `/workflow/run` được tự động:
- Loại bỏ ký tự điều khiển (trừ newline/tab)
- Cắt ngắn tại 10 000 ký tự nếu vượt quá

Giá trị sau sanitize được lưu vào `WorkflowRecord` và log.



## Yêu cầu hệ thống


- Docker Desktop (Windows) hoặc Docker Engine + Compose plugin (Linux/Mac)
- Tài nguyên CPU/RAM/GPU phù hợp với model và mức tải bạn cấu hình trong `.env`
- Nếu bật GPU: cài driver và runtime container tương thích trên máy host

---

## Cấu trúc thư mục

```
Ollama-Stack/
├── .env                          # Single source of truth cho tất cả config
├── docker-compose.yml
├── data/
│   ├── raw/                      # Tài liệu RAG — tổ chức theo project (và module tùy chọn)
│   │   ├── yanlib/               # → collection: yan_raw_docs__yanlib
│   │   │   ├── auth/             # → module=auth   (subdirectory = module)
│   │   │   ├── billing/          # → module=billing
│   │   │   ├── marketplace/      # → module=marketplace
│   │   │   └── *.md              # → module=yanlib (flat file = dùng project làm module)
│   │   └── <project>/            # Thêm bất kỳ project nào
│   └── memory/
│       └── episodic/
│           └── workflow_runs.jsonl  # Log tự động mỗi workflow run
├── rag-api/
│   ├── app.py                    # FastAPI endpoints (ingest, ask, projects, graph)
│   ├── ingest.py                 # Document ingestion: chunk → embed → Qdrant upsert
│   ├── graph.py                  # Neo4j GraphRAG layer
│   ├── requirements.txt
│   └── Dockerfile
├── agent-api/
│   ├── app.py                    # FastAPI SDLC Agent Orchestrator API
│   ├── agents.py                 # 14 cấu hình agent (model, system prompt, deps, rag_query_hint)
│   ├── workflow.py               # LangGraph StateGraph — 14-bước SDLC workflow
│   ├── requirements.txt
│   └── Dockerfile
└── open-webui-tools/
    ├── yan_knowledge_base.py     # Open WebUI tool: query RAG knowledge base
    └── yan_agent_workflow.py     # Open WebUI tool: run SDLC agent workflow
```

Mỗi subfolder trong `data/raw/` là một **project** độc lập → Qdrant collection riêng → tránh noise khi query.

Cấu trúc thư mục con trong project:
- `data/raw/{project}/{module}/file.md` → chunk có `module=<tên thư mục>`
- `data/raw/{project}/file.md` (flat) → chunk có `module=<project>`

Nhờ đó `/ask` có thể lọc kết quả theo module để tăng độ chính xác:

**File formats được hỗ trợ:** `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, `.json`, `.jsonl`, `.yaml`, `.yml`, `.xml`, `.html`, `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.cs`, `.sh`, `.sql`, `.log`

---

## Thiết lập ban đầu

### 1. Chuẩn bị `.env`

```bash
cat .env
```

Các biến quan trọng cần kiểm tra:

```env
# ─── SDLC Agent Models ───────────────────────────────────────────────────────
# Thay đổi model tại đây rồi restart service tương ứng (không cần rebuild)
EMBEDDING_MODEL=<embedding_model>
CHAT_MODEL=<chat_model>
CODING_PLANNER_MODEL=<coding_planner_model>

# Reasoning agents (BA / PM / SA / TA / DA)
BA_MODEL=<ba_model>
PM_MODEL=<pm_model>
SA_MODEL=<sa_model>
TA_MODEL=<ta_model>
DA_MODEL=<da_model>

# Team Lead planning agent
TL_MODEL=<tl_model>

# Coding agents (FE / Mobile / BE / DBA / Tech Lead / DevSecOps)
FE_MODEL=<fe_model>
MOBILE_MODEL=<mobile_model>
BE_MODEL=<be_model>
DBA_MODEL=<dba_model>
TECH_LEAD_MODEL=<tech_lead_model>
DEVSECOPS_MODEL=<devsecops_model>

# Creative/Test agents
DESIGNER_MODEL=<designer_model>
TESTER_MODEL=<tester_model>

# ─── RAG API ─────────────────────────────────────────────────────────────────
RAG_TOP_K=4

# ─── Agent API ───────────────────────────────────────────────────────────────
MEMORY_DIR=/data/memory          # thư mục ghi episodic log (workflow_runs.jsonl)
ARTIFACT_DIR=/data/artifacts     # thư mục lưu code artifacts do agent sinh ra

# ─── Bảo mật — bắt buộc đổi trước khi deploy ─────────────────────────────
NEO4J_PASSWORD=changeme_in_production
WATCHTOWER_HTTP_API_TOKEN=...

# ─── Ollama performance ───────────────────────────────────────────────────────
OLLAMA_MAX_LOADED_MODELS=<max_loaded_models>   # số model giữ đồng thời
OLLAMA_NUM_PARALLEL=<num_parallel_requests>    # số request Ollama xử lý song song
OLLAMA_KEEP_ALIVE=<model_keep_alive>           # thời gian giữ model trong bộ nhớ
OLLAMA_CONTEXT_LENGTH=<context_length_tokens>  # context window cho mỗi LLM call trong agent-api
OLLAMA_REQUEST_TIMEOUT=<timeout_seconds>       # timeout cho Ollama call trong agent-api
MAX_FILES_PER_ROLE=<max_files_per_role>        # số file tối đa mỗi coding agent sinh trong 1 workflow run

# Số batch embed/upsert chạy đồng thời khi ingest (asyncio pipeline).
# Nên điều chỉnh theo tài nguyên thực tế và mức song song mong muốn.
INGEST_EMBED_WORKERS=<ingest_embed_workers>
```

### 2. Khởi động stack

```bash
docker compose up -d
```

Lần khởi động đầu tiên sẽ pull toàn bộ image (dung lượng có thể vài GB). Theo dõi bằng:

```bash
docker compose logs -f
```

### 3. Pull models Ollama

```bash
# Pull tất cả model đang khai báo trong .env (loại trùng tự động)
grep -E '^(EMBEDDING_MODEL|CHAT_MODEL|CODING_PLANNER_MODEL|BA_MODEL|PM_MODEL|SA_MODEL|TA_MODEL|DA_MODEL|TL_MODEL|FE_MODEL|MOBILE_MODEL|BE_MODEL|DBA_MODEL|TECH_LEAD_MODEL|DEVSECOPS_MODEL|TESTER_MODEL|DESIGNER_MODEL)=' .env \
  | cut -d= -f2 \
  | sort -u \
  | xargs -I {} docker exec ollama ollama pull "{}"
```

Xác nhận các model đã sẵn sàng:

```bash
docker exec ollama ollama list
```

### 4. Tổ chức tài liệu

Đặt tài liệu vào thư mục con theo project (và tùy chọn theo module):

```
data/raw/yanlib/auth/auth-prd.md          # module=auth
data/raw/yanlib/billing/billing-plans.md  # module=billing
data/raw/yanlib/marketplace-schema.md     # module=yanlib (flat)
```

Hoặc dùng cấu trúc phẳng nếu không cần lọc theo module:

```
data/raw/myproject/spec-v1.md
data/raw/myproject/architecture.md
```

### 5. Ingest tài liệu vào Qdrant

```bash
# Ingest tất cả projects
curl -s -X POST http://localhost:8090/ingest | jq

# Hoặc ingest từng project
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "auth"}' | jq
```

### 6. Kiểm tra

```bash
# RAG API health
curl -s http://localhost:8090/health | jq

# Agent API health
curl -s http://localhost:8091/health | jq

# Xem agents và models đang dùng
curl -s http://localhost:8091/agents | jq

# Test RAG
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Auth flow hoạt động như thế nào?"}' | jq .answer
```

---

## Danh mục URL dịch vụ

| Service              | URL                                    | Mô tả                           |
|----------------------|----------------------------------------|---------------------------------|
| **Open WebUI**       | http://localhost:8085                  | Chat UI                         |
| **RAG API**          | http://localhost:8090                  | FastAPI root                    |
| **RAG API Docs**     | http://localhost:8090/docs             | Swagger UI                      |
| **RAG API ReDoc**    | http://localhost:8090/redoc            | ReDoc                           |
| **Agent API**        | http://localhost:8091                  | SDLC Agent Orchestrator root    |
| **Agent API UI**     | http://localhost:8091/ui               | SDLC Workflow UI                |
| **Agent API Docs**   | http://localhost:8091/docs             | Swagger UI                      |
| **Ollama API**       | http://localhost:11434                 | LLM inference API               |
| **Qdrant UI**        | http://localhost:6333/dashboard        | Vector DB dashboard             |
| **Qdrant REST**      | http://localhost:6333                  | Qdrant REST API                 |
| **Qdrant gRPC**      | localhost:6334                         | Qdrant gRPC                     |
| **Neo4j Browser**    | http://localhost:7474                  | Graph DB browser UI             |
| **Neo4j Bolt**       | bolt://localhost:7687                  | Neo4j Bolt (driver/tools)       |
| **Deunhealth**       | http://localhost:9999                  | Health watchdog                 |
| **Watchtower**       | http://localhost:8080 *(internal only)*| Metrics (nội bộ Docker)         |

---

## Agent API — Curl Reference

### Health & Discovery

```bash
# Kiểm tra agent-api đang chạy
curl -s http://localhost:8091/health | jq
```

```json
{
  "status": "ok",
  "ollama_base_url": "http://ollama:11434",
  "rag_api_url": "http://rag-api:8090",
  "agents": 14,
  "workflow_steps": ["ba","pm","sa","ta","designer","tl","fe","mobile","dba","be","da","tech_lead","tester","devsecops"]
}
```

```bash
# Xem tất cả agents và models đang dùng
curl -s http://localhost:8091/agents | jq
```

### Chạy single-step agent

```bash
# Chạy riêng BA agent
curl -s -X POST http://localhost:8091/agent/ba \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng hệ thống quản lý billing cho SaaS platform",
    "project": "billing",
    "rag_enabled": true,
    "rag_top_k": 5
  }' | jq .output

# Chạy PM agent với output BA đã có
curl -s -X POST http://localhost:8091/agent/pm \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng hệ thống quản lý billing cho SaaS platform",
    "project": "billing",
    "prev_outputs": {
      "ba": "<output của BA agent ở trên>"
    }
  }' | jq .output
```

### Chạy full SDLC Workflow (async)

```bash
# 1a. Submit workflow cơ bản → nhận workflow_id
curl -s -X POST http://localhost:8091/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng tính năng checkout và payment cho marketplace",
    "project": "marketplace",
    "rag_enabled": true,
    "rag_top_k": 5
  }' | jq

# 1b. Submit workflow với tech_stack bắt buộc (optional)
# Mỗi agent sẽ nhận danh sách này trong context "## Công nghệ bắt buộc"
curl -s -X POST http://localhost:8091/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Xây dựng tính năng checkout và payment cho marketplace",
    "project": "marketplace",
    "rag_enabled": true,
    "rag_top_k": 5,
    "tech_stack": [
      "Backend: NestJS (Node.js)",
      "Frontend: React + TypeScript",
      "Mobile: React Native",
      "Database: PostgreSQL",
      "Cache: Redis",
      "Message Queue: RabbitMQ",
      "Infra: Docker, Kubernetes, AWS ECS"
    ]
  }' | jq
```

```json
{
  "workflow_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Workflow đã được xếp hàng. Poll GET /workflow/a1b2c3d4-... để kiểm tra trạng thái."
}
```

```bash
# 2. Poll trạng thái (lặp lại theo chu kỳ, khoảng 15-45 phút tùy model)
curl -s http://localhost:8091/workflow/a1b2c3d4-... | jq .status

# 3. Khi status=completed, lấy output theo từng step
curl -s http://localhost:8091/workflow/a1b2c3d4-... | jq '.step_outputs.ba'
curl -s http://localhost:8091/workflow/a1b2c3d4-... | jq '.step_outputs.sa'

# 4. Xem danh sách workflow gần đây (tối đa 50 bản ghi)
curl -s http://localhost:8091/workflows | jq
```

**Workflow status values:**

| Status      | Ý nghĩa                                            |
|-------------|---------------------------------------------------|
| `pending`   | Đã vào hàng đợi, chưa bắt đầu                     |
| `running`   | LangGraph đang thực thi các bước                 |
| `completed` | Hoàn tất toàn bộ 14 bước                         |
| `failed`    | Có lỗi không xử lý được; kiểm tra trường `error` |

> **Lưu ý:** Nếu một bước gặp lỗi LLM (timeout, model chưa pull, v.v.), hệ thống sẽ ghi `[LỖI trong <role>] ...` vào `step_outputs` và trường `error`, nhưng workflow vẫn tiếp tục các bước kế tiếp (lỗi không dừng toàn pipeline).

---

## RAG API — Curl Reference

### Health

```bash
curl -s http://localhost:8090/health | jq
```

```json
{
  "status": "ok",
  "ollama_base_url": "http://ollama:11434",
  "qdrant_url": "http://qdrant:6333",
  "collection_prefix": "yan_raw_docs",
  "embedding_model": "<embedding_model_from_env>",
  "chat_model": "<chat_model_from_env>",
  "rag_top_k": 4,
  "graph": {
    "enabled": true,
    "neo4j_uri": "bolt://neo4j:7687",
    "entity_extraction": false,
    "neo4j_connected": true
  }
}
```

### Xem danh sách projects & trạng thái index

```bash
curl -s http://localhost:8090/projects | jq
```

```json
{
  "raw_data_dir": "/data/raw",
  "projects": {
    "auth": { "collection": "yan_raw_docs__auth", "indexed": true, "points_count": 142 },
    "billing": { "collection": "yan_raw_docs__billing", "indexed": false, "points_count": null }
  }
}
```

### Ingest

```bash
# Ingest tất cả projects
curl -s -X POST http://localhost:8090/ingest | jq

# Ingest 1 project cụ thể
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "marketplace"}' | jq

# Reset rồi ingest lại (xoá collection cũ trước)
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "auth", "reset": true}' | jq

# Reset tất cả projects
curl -s -X POST http://localhost:8090/reset-ingest | jq

# Reset 1 project qua endpoint reset-ingest
curl -s -X POST http://localhost:8090/reset-ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "billing"}' | jq
```

### Ask

**Request body**

| Field      | Type            | Mô tả                                                              |
|------------|-----------------|--------------------------------------------------------------------|
| `question` | `string` (bắt buộc) | Câu hỏi                                                       |
| `project`  | `string\|null`  | Lọc theo project (null = search tất cả)                           |
| `module`   | `string\|null`  | Lọc theo module trong project (null = search toàn project)        |
| `top_k`    | `int\|null`     | Số kết quả (null = dùng `RAG_TOP_K` env, mặc định 4)              |

**Các trường trong `sources[]`:** `score`, `project`, `module`, `doc_type`, `chunk_type`, `source_file`, `relative_path`, `file_type`, `chunk_index`, `preview`

```bash
# Truy vấn toàn bộ (tìm trên tất cả collection và hợp nhất kết quả)
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Mô tả auth flow?"}' | jq

# Truy vấn trong một project cụ thể
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Schema billing như thế nào?", "project": "yanlib"}' | jq

# Truy vấn theo project và module (tăng độ chính xác)
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Auth flow hoạt động như thế nào?", "project": "yanlib", "module": "auth"}' | jq

# Tùy chỉnh top_k
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Marketplace pack architecture?", "project": "yanlib", "top_k": 8}' | jq

# Chỉ lấy trường answer
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Partner platform là gì?"}' | jq .answer
```

### Qdrant — kiểm tra collections

```bash
# Liệt kê tất cả collections
curl -s http://localhost:6333/collections | jq

# Chi tiết 1 collection
curl -s http://localhost:6333/collections/yan_raw_docs__auth | jq

# Số points trong collection
curl -s http://localhost:6333/collections/yan_raw_docs__auth/points/count \
  -H "Content-Type: application/json" \
  -d '{"exact": true}' | jq

# Xoá collection thủ công
curl -s -X DELETE http://localhost:6333/collections/yan_raw_docs__auth | jq
```

### Graph (Neo4j)

```bash
# Trạng thái Neo4j và thống kê số lượng node
curl -s http://localhost:8090/graph/status | jq
```

```json
{
  "enabled": true,
  "connected": true,
  "neo4j_uri": "bolt://neo4j:7687",
  "entity_extraction": false,
  "stats": {
    "projects": 3,
    "documents": 18,
    "chunks": 742,
    "entities": 0
  }
}
```

```bash
# Xem entities đã trích xuất trong một project (yêu cầu GRAPH_ENTITY_EXTRACTION=true)
curl -s http://localhost:8090/graph/projects/auth/entities | jq
```

### Ollama

```bash
# Danh sách model đã pull
curl -s http://localhost:11434/api/tags | jq .models[].name

# Kiểm tra Ollama đang hoạt động
curl -s http://localhost:11434/

# Chat trực tiếp (không qua lớp RAG)
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<chat_model>",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }' | jq .message.content
```

---

## Quy trình cập nhật tài liệu

```bash
# 1. Thêm file vào đúng thư mục con
cp my-new-spec.md data/raw/marketplace/

# 2. Ingest lại project (idempotent: chunk đã tồn tại sẽ upsert, không tạo bản sao)
curl -s -X POST http://localhost:8090/ingest \
  -H "Content-Type: application/json" \
  -d '{"project": "marketplace"}' | jq .upserted

# 3. Kiểm tra lại truy vấn
curl -s -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "project": "marketplace"}' | jq .answer
```

> **Idempotent:** Ingest cùng một file nhiều lần sẽ upsert vào cùng point ID (deterministic UUID từ path + content hash), không tạo bản ghi trùng lặp.

---

## Open WebUI — Custom Tools

### Thiết lập

1. Mở **Admin Panel → Tools → "+"**
2. Dán nội dung file tool, lưu, rồi bật trong phần model settings

### Tool 1: `yan_knowledge_base.py` — RAG Query

Truy vấn trực tiếp knowledge base từ giao diện chat.

**Valves:**

| Valve             | Default                  | Mô tả                                            |
|-------------------|--------------------------|--------------------------------------------------|
| `rag_api_url`     | `http://rag-api:8090`    | URL rag-api (nội bộ Docker network)              |
| `timeout`         | `120`                    | Timeout giây                                     |
| `top_k`           | `null`                   | Số kết quả RAG (null = dùng `RAG_TOP_K` của API) |
| `default_project` | `null`                   | Project mặc định (null = search tất cả)          |
| `default_module`  | `null`                   | Module mặc định để lọc chunk (null = toàn project) |

**Ví dụ sử dụng trong chat:**
```
Tìm trong knowledge base: auth flow hoạt động như thế nào?
```

### Tool 2: `yan_agent_workflow.py` — SDLC Workflow

Chạy toàn bộ SDLC workflow hoặc chạy từng agent độc lập.

**Valves:**

| Valve               | Default                   | Mô tả                                                   |
|---------------------|---------------------------|----------------------------------------------------------|
| `agent_api_url`     | `http://agent-api:8091`   | URL agent-api (nội bộ Docker network)                   |
| `timeout`           | `600`                     | Timeout cho single-step request (giây)                  |
| `poll_interval`     | `15`                      | Khoảng cách poll workflow status (giây)                 |
| `poll_max_attempts` | `120`                     | Số poll tối đa (120 × 15s = 30 phút)                   |
| `rag_enabled`       | `true`                    | Query RAG cho mỗi agent step                            |
| `rag_top_k`         | `5`                       | Số kết quả RAG mỗi agent                               |
| `default_project`   | `null`                    | Project mặc định                                        |

**Functions:**

| Function              | Mô tả                                                      |
|-----------------------|------------------------------------------------------------|
| `run_sdlc_workflow`   | Chạy đầy đủ 14 bước, poll đến khi hoàn tất, trả về summary |
| `run_agent_step`      | Chạy 1 agent role đơn lẻ (sync)                           |
| `get_workflow_result` | Lấy kết quả workflow theo ID, filter theo role             |
| `list_agent_roles`    | Liệt kê tất cả roles, models, thứ tự chạy                 |

**Ví dụ sử dụng trong chat:**
```
Chạy SDLC workflow cho: xây dựng tính năng payment gateway cho marketplace, project=marketplace
```

---

## Điều chỉnh model

Tất cả model được quản lý trong `.env` tại mục `SDLC Agent Models`.  
Sau khi cập nhật `.env`, chỉ cần restart service, không cần rebuild image:

```bash
# Cập nhật model trong .env, sau đó:
docker compose restart agent-api   # cho SDLC agent models
docker compose restart rag-api     # cho CHAT_MODEL hoặc EMBEDDING_MODEL

# Nếu đổi EMBEDDING_MODEL: bắt buộc re-ingest toàn bộ tài liệu
curl -s -X POST http://localhost:8090/reset-ingest | jq
curl -s -X POST http://localhost:8090/ingest | jq
```

---

## Rebuild sau khi cập nhật mã nguồn

```bash
# Rebuild và restart agent-api
docker compose up -d --build agent-api

# Rebuild và restart rag-api
docker compose up -d --build rag-api

# Rebuild cả hai
docker compose up -d --build agent-api rag-api
```

---

## Watchtower - Auto-update

Watchtower tự pull image mới và restart container theo lịch (`WATCHTOWER_SCHEDULE` trong `.env`, mặc định mỗi phút).

```bash
# Kích hoạt cập nhật thủ công qua HTTP API
curl -s -H "Authorization: Bearer ${WATCHTOWER_HTTP_API_TOKEN}" \
  http://localhost:8080/v1/update

# Xem metrics hiện tại
curl -s -H "Authorization: Bearer ${WATCHTOWER_HTTP_API_TOKEN}" \
  http://localhost:8080/v1/metrics
```

> **Lưu ý:** `rag-api` và `agent-api` đang dùng local build với `watchtower.enable=false`, nên Watchtower không tự cập nhật. Sử dụng `docker compose up -d --build` khi cần cập nhật.

---

## Vận hành Docker

```bash
# Khởi động toàn bộ dịch vụ
docker compose up -d

# Dừng toàn bộ dịch vụ
docker compose down

# Rebuild rag-api sau khi cập nhật mã nguồn
docker compose up -d --build rag-api

# Theo dõi logs theo thời gian thực
docker compose logs -f agent-api
docker compose logs -f rag-api

# Khởi động lại một service
docker compose restart agent-api

# Theo dõi mức sử dụng tài nguyên
docker stats

# Xóa volumes (NGUY HIỂM: xóa toàn bộ dữ liệu Ollama + Qdrant + Neo4j)
docker compose down -v
```

---

## Troubleshooting

| Triệu chứng | Nguyên nhân | Hướng xử lý |
|---|---|---|
| `rag-api` / `agent-api` crash khi start | Thiếu env var bắt buộc | `docker compose logs rag-api` hoặc `agent-api` |
| `/ask` trả về 404 "chưa được ingest" | Chưa chạy `/ingest` | `POST /ingest {"project": "..."}` |
| `/ingest` trả về `"status": "empty"` | Không có subfolder trong `data/raw/` | Tạo subfolder và đặt file vào đó |
| `/ask` với `module` không trả kết quả | Module chưa tồn tại trong collection | Kiểm tra tên module = tên thư mục con trong project |
| Ingest chậm hoặc mất nhiều thời gian | `INGEST_EMBED_WORKERS` quá thấp | Tăng `INGEST_EMBED_WORKERS` bằng `OLLAMA_NUM_PARALLEL`; restart `rag-api` |
| Embedding chậm / timeout | Model chưa được pull | Pull lại model đang cấu hình trong `.env` bằng lệnh ở mục **Pull models Ollama** |
| SDLC workflow `status=failed` | Xem field `error` | `curl .../workflow/{id} \| jq .error` |
| SDLC step output bắt đầu bằng `[LỖI` | LLM timeout hoặc model chưa pull | Kiểm tra model đã pull, tăng `OLLAMA_REQUEST_TIMEOUT` trong `.env` |
| Output chứa `<think>...</think>` thô | Model reasoning chưa được nhận diện | Kiểm tra tên model khớp rule nhận diện trong `agent-api/workflow.py` |
| `user_input` bị cắt ngắn trong workflow | Input > 10 000 ký tự | Input được sanitize tự động; chia nhỏ nếu cần |
| `workflow_runs.jsonl` không được tạo | `MEMORY_DIR` không mount | Kiểm tra volume `./data/memory:/data/memory` trong `docker-compose.yml` |
| Open WebUI không gọi được rag-api | URL sai (dùng `localhost`) | Valve `rag_api_url` = `http://rag-api:8090` |
| Open WebUI không gọi được agent-api | URL sai (dùng `localhost`) | Valve `agent_api_url` = `http://agent-api:8091` |
| Qdrant collection không tồn tại | Collection bị xoá hoặc chưa ingest | Chạy lại `POST /ingest` |
| Neo4j không kết nối được | Container chưa sẵn sàng | Đợi ~30s, kiểm tra `docker compose logs neo4j` |
| `/graph/status` → `connected: false` | Sai `NEO4J_PASSWORD` | `.env` phải khớp với lần đầu Neo4j khởi tạo |
| `graph_chunks_saved: 0` sau ingest | `GRAPH_ENABLED=false` trong `.env` | Đổi thành `true` + `docker compose up -d --build rag-api` |
| Workflow mất > 30 phút | Model nặng hoặc tài nguyên thiếu | Giảm mức song song, chọn model nhẹ hơn, hoặc tăng `poll_max_attempts` |
| `workflow_id` not found sau vài giờ | In-memory store bị evict (tối đa 50) | Tăng `_MAX_STORED_WORKFLOWS` trong `agent-api/app.py` hoặc lưu kết quả ngay sau khi hoàn tất |

---

## Neo4j — Cypher Queries

Mở Neo4j Browser tại [http://localhost:7474](http://localhost:7474) (login: `neo4j` / giá trị `NEO4J_PASSWORD` trong `.env`).

```cypher
// Xem toàn bộ graph của 1 project
MATCH path = (:Project {name:"auth"})-[:HAS_DOCUMENT]->(:Document)-[:HAS_CHUNK]->(:Chunk)
RETURN path LIMIT 50;

// Thống kê node counts
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC;

// Top entities được đề cập nhiều nhất (cần GRAPH_ENTITY_EXTRACTION=true)
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN e.name, e.type, count(c) AS mentions
ORDER BY mentions DESC LIMIT 20;

// Chunks đề cập một entity cụ thể
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name:"Billing Plan"})
RETURN c.source_file, c.chunk_index, c.text LIMIT 10;

// Entities co-occur (xuất hiện cùng nhau trong chunks)
MATCH (e1:Entity)-[:CO_OCCURS_WITH]-(e2:Entity)
RETURN e1.name, e2.name LIMIT 30;

// Xoá toàn bộ graph (reset thủ công)
MATCH (n) DETACH DELETE n;
```