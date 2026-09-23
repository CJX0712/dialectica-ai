"""Self-contained end-to-end verification.

Unit tests alone prove nothing about whether the process actually boots and
serves a request. This script:

  1. runs the unit suite (subprocess)
  2. boots the real HTTP server on a free port (subprocess)
  3. polls /health until 200 -- printing the server stderr on timeout instead
     of silently skipping
  4. exercises the success flow and every error flow over real HTTP
  5. tears the server down with taskkill (Windows leaves the port held otherwise)
  6. runs the P0 emoji gate

Author: 晨星
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

CORPUS = [
    "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。"
    "绕行方式是使用 pip install --target 指定目录，再通过 PYTHONPATH 加载。",
    "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support，需要先安装 PySocks。",
    "npm registry 被指向了鸿蒙源 ohpm.openharmony.cn，缺少通用包。"
    "安装时需要显式指定 --registry=https://registry.npmmirror.com。",
]

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(f"{name}{(' :: ' + detail) if detail else ''}")
        print(f"  FAIL  {name} {detail}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http(method: str, url: str, payload=None, timeout: float = 30.0):
    """Localhost-only HTTP. Proxies are disabled explicitly: the machine runs a
    SOCKS5 tunnel and anything else sends loopback traffic into it."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            status, _ = http("GET", url, timeout=3)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
        return
    except Exception:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/pid", str(proc.pid), "/t", "/f"],
            capture_output=True,
        )
    else:  # pragma: no cover
        proc.kill()


def run_unit_tests(python: str) -> None:
    print("\n[1/4] 单元测试")
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [python, "-m", "pytest", os.path.join(ROOT, "tests"), "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    tail = (r.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print("   ", line)
    check("pytest 全绿", r.returncode == 0, f"exit={r.returncode}")


def run_e2e(python: str) -> None:
    print("\n[2/4] 端到端服务链路")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        PYTHONPATH=SRC,
        PYTHONIOENCODING="utf-8",
        DIALECTICA_PORT=str(port),
        DIALECTICA_HOST="127.0.0.1",
    )
    script = (
        "import uvicorn;"
        "from dialectica.api.app import app;"
        "uvicorn.run(app, host='127.0.0.1', port=" + str(port) + ", log_level='warning')"
    )
    proc = subprocess.Popen(
        [python, "-c", script],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        ok = wait_healthy(f"{base}/health", proc)
        if not ok:
            err = ""
            try:
                err = (proc.stderr.read() or "")[-2000:]
            except Exception:
                pass
            check("服务启动并响应 /health", False, f"stderr={err}")
            return
        check("服务启动并响应 /health", True)

        status, body = http("POST", f"{base}/ingest", {"texts": CORPUS})
        check("POST /ingest 返回 200", status == 200 and body.get("chunks", 0) >= 3, str(body))

        status, body = http("POST", f"{base}/debate", {"query": "沙箱里为什么无法安装 Python 包"})
        check("POST /debate 返回 200", status == 200, str(body)[:200])
        check("辩论产出非空答案", bool(body.get("answer")), str(body.get("answer"))[:120])
        check("答案通过收敛判定", bool(body.get("converged")))
        check("证据跨度非空", len(body.get("evidence", [])) > 0)
        check("断言全部已锚定", all(c.get("anchored") for c in body.get("claims", [])))

        status, body = http("POST", f"{base}/ask", {"query": "12*(3+4)"})
        check("确定性工具路由命中计算器", status == 200 and "84" in str(body.get("answer", "")), str(body))

        status, _ = http("POST", f"{base}/debate", {"query": "完全不相关的查询"})
        check("空语料之外不误报 422", status == 200)

        status, body = http("POST", f"{base}/reset", {})
        check("POST /reset 清空语料", status == 200)
        status, _ = http("POST", f"{base}/ingest", {"texts": []})
        check("空入库被拒绝 400", status == 400)

        status, body = http("POST", f"{base}/ingest", {"texts": CORPUS})
        status, body = http("POST", f"{base}/evaluate", {})
        check("POST /evaluate 覆盖全部用例", status == 200 and body.get("total") == 3, str(body)[:200])
        check("基准通过率 >= 2/3", body.get("passed", 0) >= 2, str(body.get("passed")))

        status, body = http("GET", f"{base}/trace")
        check("GET /trace 有事件记录", status == 200 and len(body.get("events", [])) > 0)
    finally:
        kill(proc)
    print("  服务已回收")


def run_cli(python: str) -> None:
    print("\n[3/4] CLI 冒烟")
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
    corpus_path = os.path.join(ROOT, ".verify_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(CORPUS))
    r = subprocess.run(
        [python, "-m", "dialectica.cli", "ingest", "--file", corpus_path],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    check("cli ingest 成功", r.returncode == 0, (r.stderr or "")[-200:])

    r = subprocess.run(
        [python, "-m", "dialectica.cli", "ask", "12*(3+4)"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    check("cli ask 工具路由", r.returncode == 0 and "84" in (r.stdout or ""), (r.stdout or "")[:120])

    r = subprocess.run(
        [python, "-m", "dialectica.cli", "health"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    check("cli health 输出 JSON", r.returncode == 0 and '"status"' in (r.stdout or ""))
    if os.path.exists(corpus_path):
        os.remove(corpus_path)


def run_emoji_gate(python: str) -> None:
    print("\n[4/4] P0 门禁：emoji 扫描")
    r = subprocess.run(
        [python, os.path.join(ROOT, "scripts", "scan_emoji.py")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    check("仓库无 emoji 作功能图标", r.returncode == 0, (r.stdout or "")[-400:])


def main() -> int:
    python = sys.executable
    print(f"Dialectica 验证 · Python {python}")
    run_unit_tests(python)
    run_e2e(python)
    run_cli(python)
    run_emoji_gate(python)
    print(f"\n通过：{len(PASSED)} / 失败：{len(FAILED)}")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
