# 辩衡 Dialectica · 部署指南

作者：晨星

---

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | >= 3.10（开发/验证用 3.13） |
| GPU | **不需要** |
| 网络 | **不需要**（默认离线链路） |
| 磁盘 | 核心栈 ~120 MB；加生产栈（含 ONNX 模型）~2 GB |
| OS | Windows / Linux / macOS（锁版清单已剔除 Windows 专有包） |

## 2. 三种部署形态

### 形态 A：离线零依赖（默认，最快验证）

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt       # Linux/macOS: .venv/bin/pip
python scripts/verify.py
```

链路：Mock LLM（确定性抽取式）+ 哈希嵌入（IDF 加权）+ 自研 BM25 + 内存余弦索引 + 启发式重排。
用途：CI、回归、离线审计。**不提供真实语义能力**。

### 形态 B：本地 CPU 全栈（推荐自托管）

```bash
pip install -r requirements-prod.txt
```

```bash
export DIALECTICA_EMBEDDING_BACKEND=fastembed
export DIALECTICA_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
export DIALECTICA_VECTOR_BACKEND=faiss
export DIALECTICA_RERANK_BACKEND=cross-encoder
export DIALECTICA_RERANK_MODEL=Xenova/bge-reranker-base
export DIALECTICA_LLM_BACKEND=llamacpp
export DIALECTICA_GGUF_PATH=/models/qwen2.5-7b-instruct-q4_k_m.gguf
export DIALECTICA_N_THREADS=4
```

首次运行会下载 ONNX 模型（约 100 MB）。**离线环境请提前下载并缓存**。

> `N_THREADS` 不要设成 CPU 核数。小量化模型是内存带宽受限，线程过多会让吞吐掉 4 倍。锁在 2-4。

### 形态 C：远端模型

```bash
export DIALECTICA_LLM_BACKEND=ollama
export DIALECTICA_OLLAMA_HOST=http://127.0.0.1:11434
export DIALECTICA_LLM_MODEL=qwen2.5:7b
```

或任意 OpenAI 兼容端点（vLLM / OpenAI / DeepSeek / Together）：

```bash
export DIALECTICA_LLM_BACKEND=openai
export DIALECTICA_OPENAI_BASE_URL=https://api.openai.com/v1
export DIALECTICA_OPENAI_API_KEY=sk-...
export DIALECTICA_LLM_MODEL=gpt-4o-mini
```

## 3. 启动服务

```bash
python -m dialectica.cli serve --host 0.0.0.0 --port 8000
# 或
uvicorn dialectica.api.app:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

控制台：浏览器打开 `web/console.html`（单文件，零依赖，双击即可），把 API 地址指向服务。

## 4. Docker

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "dialectica.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t dialectica .
docker run -p 8000:8000 dialectica
```

生产镜像建议用多阶段构建，并把 `requirements.lock.txt` 换成 `requirements-prod.txt` 后再 `pip freeze` 一次锁版。

## 5. CI

`.github/workflows/ci.yml` 在 Python 3.11 / 3.12 / 3.13 上跑：

1. `pip install -r requirements.txt -r requirements-dev.txt`
2. `ruff check src tests`
3. `pytest tests -q`
4. `python scripts/verify.py`
5. `python scripts/scan_emoji.py`（P0 门禁）

全部无网络依赖。

## 6. 已知环境陷阱（本机实测）

| 症状 | 根因 | 对策 |
|---|---|---|
| `pip` 报 SOCKS 相关错误 | 机器有 SOCKS5 隧道 | 先 `pip install PySocks` |
| `pip` 被守卫拦截写 site-packages | 沙箱守卫 | `pip install --target <dir>` + `PYTHONPATH` |
| npm 装包 404 | registry 被指向鸿蒙源 | `--registry=https://registry.npmmirror.com` |
| 访问 localhost 报 `WinError 10054` | httpx 把本机流量送进代理 | 指向 localhost 的 client 一律 `trust_env=False`（本仓库 `OllamaLLM` 已这样写） |
| 后台起的服务过一会儿消失 | 作业对象随父 shell 退出被杀 | 用 `subprocess.Popen` 自包含脚本（见 `scripts/verify.py`） |
| 端口泄漏导致下次跑失败 | 子进程未回收 | `taskkill /pid <pid> /t /f`（`verify.py` 的 `kill()` 已实现） |
| 源码里 emoji 变成乱码 | 控制台代码页 mangled | 检测类代码只用 `ord()` 码点范围，源文件不放 emoji 字面量 |

## 7. 配置管理

复制 `.env.example` 为 `.env`，按需修改。所有配置项见 `docs/SPEC.md` 第 3 节。

## 8. 容量与性能

| 后端 | 适用规模 | 备注 |
|---|---|---|
| `MemoryIndex` | <= 约 20k 切片 | 精确余弦扫描，零依赖 |
| `FaissIndex` | 10 万级 | IndexFlatIP，精确；内存约 `dim × 4 B × N`（dim=512 时 100 万条约 2 GB） |
| `HashingEmbedder` | 任意 | dim=2048 时每条约 16 KB（Python float） |
| `fastembed` bge-small-zh | 任意 | 512 维，ONNX int8 可选 |

辩衡的瓶颈在**内存带宽**而非算力：检索是向量扫描，生成是 llama.cpp。加线程通常不提速。
