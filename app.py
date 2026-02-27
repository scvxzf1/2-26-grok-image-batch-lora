import asyncio
import base64
import errno
import hashlib
import ipaddress
import io
import json
import logging
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image


DEFAULT_PORT = 12607
SINGLE_RETRY_MAX_CONCURRENCY = 10
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
PASSWORD_PLACEHOLDER = "********"


def now_ts() -> int:
    return int(time.time())


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def parse_env_int(name: str, default_value: int, min_value: int, max_value: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default_value
    try:
        parsed = int(raw)
    except Exception:
        return default_value
    return clamp_int(parsed, min_value, max_value)


MAX_DEBUG_LOG_CHARS = parse_env_int("MAX_DEBUG_LOG_CHARS", 400000, 20000, 2000000)
MAX_TASK_DEBUG_LOG_CHARS = parse_env_int("MAX_TASK_DEBUG_LOG_CHARS", 500000, 20000, 3000000)
MAX_SSE_RAW_LOG_CHARS = parse_env_int("MAX_SSE_RAW_LOG_CHARS", 200000, 20000, 1000000)
MAX_SSE_OBJECTS = parse_env_int("MAX_SSE_OBJECTS", 200, 20, 2000)
MAX_SOURCE_IMAGE_BYTES = parse_env_int("MAX_SOURCE_IMAGE_BYTES", 20 * 1024 * 1024, 256 * 1024, 200 * 1024 * 1024)
MAX_CANDIDATE_IMAGE_BYTES = parse_env_int(
    "MAX_CANDIDATE_IMAGE_BYTES", 20 * 1024 * 1024, 256 * 1024, 200 * 1024 * 1024
)
MAX_PRESET_IMPORT_BYTES = parse_env_int("MAX_PRESET_IMPORT_BYTES", 2 * 1024 * 1024, 64 * 1024, 20 * 1024 * 1024)
MAX_HTTP_REDIRECTS = parse_env_int("MAX_HTTP_REDIRECTS", 3, 0, 10)
CACHE_TTL_HOURS = parse_env_int("CACHE_TTL_HOURS", 168, 0, 24 * 365)
CACHE_MAX_BYTES = parse_env_int("CACHE_MAX_BYTES", 2 * 1024 * 1024 * 1024, 0, 200 * 1024 * 1024 * 1024)


def parse_env_bool(name: str, default_value: bool) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default_value
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default_value


def parse_config_bool(value: Any, default_value: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default_value


def parse_env_set(name: str) -> set:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return set()
    return {item.strip().lower().rstrip(".") for item in raw.split(",") if item.strip()}


ALLOW_PRIVATE_API_URL = parse_env_bool("ALLOW_PRIVATE_API_URL", False)
ALLOW_PRIVATE_DOWNLOAD_URL = parse_env_bool("ALLOW_PRIVATE_DOWNLOAD_URL", False)
ALLOW_REMOTE_DEBUG_LOG_API = parse_env_bool("ALLOW_REMOTE_DEBUG_LOG_API", False)
API_URL_HOST_ALLOWLIST = parse_env_set("API_URL_HOST_ALLOWLIST")
DOWNLOAD_URL_HOST_ALLOWLIST = parse_env_set("DOWNLOAD_URL_HOST_ALLOWLIST")


def detect_mode(api_mode: str, api_url: str) -> str:
    if api_mode and api_mode != "auto":
        return api_mode
    return "responses" if "/responses" in (api_url or "").lower() else "chat_completions"


def select_token_field(token_field: str, mode: str) -> str:
    if token_field and token_field != "auto":
        return token_field
    return "max_output_tokens" if mode == "responses" else "max_tokens"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [f"http://127.0.0.1:{DEFAULT_PORT}", f"http://localhost:{DEFAULT_PORT}"]


def host_in_allowlist(hostname: str, allowlist: set) -> bool:
    if not allowlist:
        return True
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    if host in allowlist:
        return True
    return any(host.endswith(f".{item}") for item in allowlist)


def resolve_host_ips(hostname: str) -> List[ipaddress._BaseAddress]:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return []
    try:
        return [ipaddress.ip_address(host)]
    except Exception:
        pass

    addresses: List[ipaddress._BaseAddress] = []
    seen = set()
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:
        return []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_text = str(sockaddr[0])
        if ip_text in seen:
            continue
        seen.add(ip_text)
        try:
            addresses.append(ipaddress.ip_address(ip_text))
        except Exception:
            continue
    return addresses


def is_non_public_ip(address: ipaddress._BaseAddress) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_network_url(url: str, purpose: str, allow_private: bool, allowlist: set) -> None:
    target = (url or "").strip()
    if not target:
        raise ValueError(f"{purpose} 为空")
    parsed = urlparse(target)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"{purpose} 仅允许 http/https 协议")
    hostname = (parsed.hostname or "").strip()
    if not hostname:
        raise ValueError(f"{purpose} 缺少主机名")
    if not host_in_allowlist(hostname, allowlist):
        raise ValueError(f"{purpose} 主机不在白名单: {hostname}")
    if allow_private:
        return
    ips = resolve_host_ips(hostname)
    blocked = [str(ip) for ip in ips if is_non_public_ip(ip)]
    if blocked:
        raise ValueError(f"{purpose} 不允许访问内网/本地地址: {hostname} -> {', '.join(blocked[:3])}")


def is_private_api_url_unlocked(config: Dict[str, Any]) -> bool:
    if ALLOW_PRIVATE_API_URL:
        return True
    return parse_config_bool(config.get("unlock_local_api_url", False), False)


def is_private_download_url_unlocked(config: Dict[str, Any]) -> bool:
    if ALLOW_PRIVATE_DOWNLOAD_URL:
        return True
    return parse_config_bool(config.get("unlock_local_download_url", False), False)


def resolve_bind_host(host: str) -> str:
    normalized = (host or "").strip()
    if normalized in {"", "0.0.0.0", "::"}:
        return normalized or "0.0.0.0"
    return normalized


def try_bind_port(host: str, port: int) -> Tuple[bool, Optional[OSError]]:
    bind_host = resolve_bind_host(host)
    family = socket.AF_INET6 if ":" in bind_host and bind_host != "0.0.0.0" else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
        return True, None
    except OSError as exc:
        return False, exc


def parse_pids_from_lsof(raw: str) -> List[int]:
    pids: List[int] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def parse_pids_from_fuser(raw: str) -> List[int]:
    content = raw or ""
    if ":" in content:
        content = content.split(":", 1)[1]
    return [int(token) for token in re.findall(r"\b\d+\b", content)]


def parse_pids_from_ss(raw: str) -> List[int]:
    return [int(token) for token in re.findall(r"pid=(\d+)", raw or "")]


def compact_text(raw: str, limit: int = 220) -> str:
    text = (raw or "").strip().replace("\r", " ").replace("\n", " | ")
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def has_ss_listener(raw: str) -> bool:
    for line in (raw or "").splitlines():
        line = line.strip()
        if line:
            return True
    return False


def find_listen_pids(port: int) -> Tuple[List[int], bool, List[str]]:
    candidates: List[int] = []
    diagnostics: List[str] = []
    listener_detected = False

    probes: List[Tuple[List[str], Any, bool]] = [
        (["ss", "-H", "-ltnp", f"sport = :{port}"], parse_pids_from_ss, True),
        (["ss", "-H", "-ltn", f"sport = :{port}"], lambda _raw: [], True),
        (["lsof", "-nP", "-ti", f"TCP:{port}", "-sTCP:LISTEN"], parse_pids_from_lsof, False),
        (["fuser", "-n", "tcp", str(port)], parse_pids_from_fuser, False),
    ]
    for command, parser, check_listener in probes:
        cmd_text = " ".join(command)
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            diagnostics.append(f"{cmd_text}: 命令不存在")
            continue
        except Exception as exc:
            diagnostics.append(f"{cmd_text}: 执行失败 ({exc})")
            continue

        raw = f"{result.stdout}\n{result.stderr}"
        if check_listener and has_ss_listener(result.stdout):
            listener_detected = True
        parsed_pids = parser(raw)
        if parsed_pids:
            listener_detected = True
        if parsed_pids:
            diagnostics.append(f"{cmd_text}: 命中 PID -> {sorted(set(parsed_pids))}")
        else:
            compact = compact_text(raw)
            if compact:
                diagnostics.append(f"{cmd_text}: 未解析到 PID，输出 -> {compact}")
            else:
                diagnostics.append(f"{cmd_text}: 无输出 (rc={result.returncode})")

        for pid in parsed_pids:
            if pid != os.getpid():
                candidates.append(pid)

    return sorted(set(candidates)), listener_detected, diagnostics


def get_process_name(pid: int) -> str:
    proc_comm = Path(f"/proc/{pid}/comm")
    try:
        if proc_comm.exists():
            name = proc_comm.read_text(encoding="utf-8", errors="ignore").strip()
            if name:
                return name
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            check=False,
        )
        name = (result.stdout or "").strip()
        if name:
            return name
    except Exception:
        pass
    return "unknown"


def format_process_list(pids: List[int]) -> str:
    if not pids:
        return "未知"
    items: List[str] = []
    for pid in pids:
        items.append(f"{pid}({get_process_name(pid)})")
    return ", ".join(items)


def ask_yes_no(title: str, message: str) -> bool:
    print(title)
    print(message)
    try:
        user_input = input("请输入 yes 或 no: ").strip().lower()
    except EOFError:
        return False
    return user_input in {"y", "yes", "是", "确认", "继续"}


def kill_pids(pids: List[int]) -> None:
    if not pids:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    deadline = time.time() + 3.0
    while time.time() < deadline:
        alive: List[int] = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                alive.append(pid)
        if not alive:
            return
        time.sleep(0.2)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def ensure_port_ready_or_exit(host: str, port: int) -> bool:
    ok, bind_error = try_bind_port(host, port)
    if ok:
        return True

    bind_host = resolve_bind_host(host)
    pids, listener_detected, diagnostics = find_listen_pids(port)
    family = "IPv6" if ":" in bind_host and bind_host != "0.0.0.0" else "IPv4"

    error_detail = "未知错误"
    if bind_error is not None:
        errno_prefix = f"[Errno {bind_error.errno}] " if bind_error.errno is not None else ""
        error_detail = f"{errno_prefix}{bind_error.strerror or str(bind_error)}"

    if bind_error is not None and bind_error.errno != errno.EADDRINUSE:
        print("端口绑定失败：")
        print(f"- 目标地址: {bind_host}:{port}")
        print(f"- 绑定协议: {family}")
        print(f"- 绑定错误: {error_detail}")
        if diagnostics:
            print("- 诊断明细:")
            for item in diagnostics:
                print(f"  - {item}")
        return False

    if not listener_detected:
        print("端口预检提示占用，但未检测到监听进程，自动继续启动。")
        if diagnostics:
            for item in diagnostics:
                print(f"- 诊断: {item}")
        return True

    print("端口冲突：")
    print(f"- 目标地址: {bind_host}:{port}")
    print(f"- 绑定协议: {family}")
    print(f"- 绑定错误: {error_detail}")

    if pids:
        occupied = format_process_list(pids)
        question = (
            f"检测到端口 {port} 已被占用。\n"
            f"占用进程: {occupied}\n"
            "是否关闭占用进程并继续启动？"
        )
        if not ask_yes_no("端口冲突", question):
            print("用户取消，程序退出。")
            return False
        kill_pids(pids)
        time.sleep(0.3)
        retry_ok, _ = try_bind_port(host, port)
        if retry_ok:
            print("已释放端口，继续启动。")
            return True
        print("尝试关闭占用进程后，端口仍不可用。")
        return False
    else:
        print("- 占用进程: 未识别到 PID（可能是其他用户/系统进程）")

    if diagnostics:
        print("- 诊断明细:")
        for item in diagnostics:
            print(f"  - {item}")

    print("- 处理建议: 请先手动释放端口，或改用其他端口（例如 PORT=12608）")
    return False


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__(level=logging.INFO)
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.log_queue.put_nowait(message)
        except Exception:
            return


def mount_runtime_log_queue(
    log_queue: "queue.Queue[str]",
) -> List[Tuple[logging.Logger, logging.Handler, List[logging.Handler], bool]]:
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(message)s", datefmt="%H:%M:%S")
    mounted: List[Tuple[logging.Logger, logging.Handler, List[logging.Handler], bool]] = []
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        old_handlers = list(logger.handlers)
        old_propagate = logger.propagate
        logger.handlers = []
        logger.propagate = False
        handler = QueueLogHandler(log_queue)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        mounted.append((logger, handler, old_handlers, old_propagate))
    return mounted


def unmount_runtime_log_queue(
    mounted: List[Tuple[logging.Logger, logging.Handler, List[logging.Handler], bool]],
) -> None:
    for logger, handler, old_handlers, old_propagate in mounted:
        logger.removeHandler(handler)
        logger.handlers = old_handlers
        logger.propagate = old_propagate


def run_with_cli_dashboard(host: str, port: int) -> int:
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    runtime_logs: "queue.Queue[str]" = queue.Queue()
    mounted = mount_runtime_log_queue(runtime_logs)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    view_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    access_url = f"http://{view_host}:{port}"
    recent_logs: List[str] = []

    def pump_logs() -> bool:
        changed = False
        while True:
            try:
                line = runtime_logs.get_nowait()
            except queue.Empty:
                break
            recent_logs.append(line)
            changed = True
        if len(recent_logs) > 800:
            del recent_logs[:-800]
        return changed

    def run_status_label() -> str:
        if server.started and not server.should_exit:
            return "运行中"
        if server.should_exit:
            return "停止中"
        return "启动中"

    def run_plain_dashboard() -> int:
        print("image-batch-console CLI 状态界面", flush=True)
        print(f"地址: {access_url}", flush=True)
        print("操作: Ctrl+C 停止服务并退出", flush=True)
        last_status = ""
        try:
            while server_thread.is_alive():
                pump_logs()
                status = run_status_label()
                if status != last_status:
                    print(f"[状态] {status}", flush=True)
                    last_status = status
                while recent_logs:
                    print(recent_logs.pop(0), flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            server.should_exit = True
        return 0

    def run_curses_dashboard() -> int:
        import curses

        def draw_screen(stdscr: Any) -> None:
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.timeout(150)
            last_signature: Tuple[Any, ...] = ()

            while True:
                logs_changed = pump_logs()
                key = stdscr.getch()
                if key in (ord("q"), ord("Q")):
                    server.should_exit = True

                height, width = stdscr.getmaxyx()
                status = run_status_label()
                log_rows = max(6, height - 8)
                visible_logs = recent_logs[-log_rows:] if recent_logs else ["(暂无日志)"]
                signature = (status, len(visible_logs), width, height, visible_logs[-1] if visible_logs else "")

                if logs_changed or signature != last_signature:
                    stdscr.erase()
                    header_lines = [
                        "image-batch-console CLI 状态界面",
                        f"状态: {status}",
                        f"地址: {access_url}",
                        "操作: 按 q 退出（或 Ctrl+C）",
                    ]
                    for row, line in enumerate(header_lines):
                        if row >= height:
                            break
                        stdscr.addnstr(row, 0, line, max(1, width - 1))

                    split_row = min(len(header_lines), max(0, height - 1))
                    if split_row < height:
                        try:
                            stdscr.hline(split_row, 0, "-", max(1, width - 1))
                        except curses.error:
                            pass

                    start_row = split_row + 1
                    for index, log_line in enumerate(visible_logs):
                        row = start_row + index
                        if row >= height:
                            break
                        stdscr.addnstr(row, 0, str(log_line), max(1, width - 1))
                    stdscr.refresh()
                    last_signature = signature

                if not server_thread.is_alive():
                    break

        try:
            curses.wrapper(draw_screen)
            return 0
        except KeyboardInterrupt:
            server.should_exit = True
            return 0
        except Exception as exc:
            print(f"CLI TUI 启动失败，已降级为普通日志模式: {exc}")
            return run_plain_dashboard()

    try:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return run_curses_dashboard()
        return run_plain_dashboard()
    finally:
        if server_thread.is_alive():
            server.should_exit = True
            server_thread.join(timeout=5)
        unmount_runtime_log_queue(mounted)
        print("服务已退出。")


def sanitize_config_for_client(config: Dict[str, Any]) -> Dict[str, Any]:
    safe_config = dict(config)
    if safe_config.get("api_key"):
        safe_config["api_key"] = PASSWORD_PLACEHOLDER
    else:
        safe_config["api_key"] = ""
    return safe_config


def merge_api_key_if_empty(incoming: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(incoming)
    api_key = (result.get("api_key") or "").strip()
    if not api_key or api_key == PASSWORD_PLACEHOLDER:
        result["api_key"] = (fallback.get("api_key") or "").strip()
    return result


def safe_filename_component(value: str, fallback: str = "result") -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", value).strip()
    cleaned = cleaned.replace("..", "_")
    if not cleaned:
        return fallback
    return cleaned


def load_json(path: Path, default_value: Any) -> Any:
    if not path.exists():
        return default_value
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def is_probable_image_url(value: str) -> bool:
    lower_value = value.lower()
    if lower_value.startswith("data:image/"):
        return True
    if any(lower_value.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return True
    return any(token in lower_value for token in ["/image", "image=", "img=", "format=png", "format=jpg", "format=jpeg"])


def normalize_base64(value: str) -> str:
    value = value.strip()
    padding = len(value) % 4
    if padding:
        value += "=" * (4 - padding)
    return value


def decode_data_url(data_url: str) -> bytes:
    if ";base64," not in data_url:
        raise ValueError("仅支持 base64 data url")
    _, payload = data_url.split(";base64,", 1)
    return base64.b64decode(normalize_base64(payload))


def read_image_info(content: bytes) -> Tuple[str, int, int]:
    with Image.open(io.BytesIO(content)) as image:
        fmt = (image.format or "PNG").lower()
        width, height = image.size
    return fmt, width, height


def parse_markdown_image_urls(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"!\[[^\]]*\]\((https?://[^)\s]+|data:image/[^\s\"'<>]+)\)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return [item.strip() for item in matches]


def parse_plain_image_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"(https?://[^\s\"'<>]+|data:image/[^\s\"'<>]+)", text, flags=re.IGNORECASE)
    result_urls = []
    for raw_url in urls:
        url = raw_url.rstrip("),.;]")
        if is_probable_image_url(url):
            result_urls.append(url)
    return result_urls


def build_url_candidate(value: str, key: str) -> Dict[str, str]:
    normalized_value = (value or "").strip()
    if normalized_value.lower().startswith("data:image/"):
        candidate_type = "data_url"
    else:
        candidate_type = "url"
    return {"type": candidate_type, "value": normalized_value, "key": key}


def extract_after_think_tag(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def parse_sse_event_payload(event_lines: List[str]) -> str:
    if not event_lines:
        return ""
    has_data_prefix = any(line.lstrip().startswith("data:") for line in event_lines)
    payload_lines: List[str] = []
    for raw_line in event_lines:
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            payload_lines.append(line[len("data:"):].strip())
            continue
        if not has_data_prefix:
            payload_lines.append(line)
    return "\n".join(payload_lines).strip()


def parse_sse_from_lines(lines: List[str]) -> Dict[str, Any]:
    event_buffer: List[str] = []
    event_count = 0
    parse_errors = 0
    done = False
    text_parts: List[str] = []
    parsed_objects: List[Dict[str, Any]] = []
    usage: Dict[str, Any] = {}

    def flush_event() -> None:
        nonlocal event_buffer, event_count, parse_errors, done, usage
        if not event_buffer:
            return
        payload = parse_sse_event_payload(event_buffer)
        event_buffer = []
        if not payload:
            return
        if payload == "[DONE]":
            done = True
            return
        try:
            event_obj = json.loads(payload)
        except Exception:
            parse_errors += 1
            return

        if not isinstance(event_obj, dict):
            parse_errors += 1
            return

        event_count += 1
        if len(parsed_objects) < MAX_SSE_OBJECTS:
            parsed_objects.append(event_obj)
        if isinstance(event_obj.get("usage"), dict):
            usage = event_obj["usage"]

        choices = event_obj.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)

    for raw_line in lines:
        if raw_line is None:
            continue
        line = str(raw_line).rstrip("\r")
        if line == "":
            flush_event()
            if done:
                break
            continue
        event_buffer.append(line)
    flush_event()

    return {
        "ok": bool(event_count > 0 or done),
        "event_count": event_count,
        "parse_errors": parse_errors,
        "done": done,
        "text": "".join(text_parts),
        "objects": parsed_objects,
        "usage": usage,
    }


def parse_sse_from_text(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    parsed = parse_sse_from_lines(lines)
    parsed["raw_text"] = text
    return parsed


def parse_sse_from_response(response: requests.Response, cancel_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    event_buffer: List[str] = []
    event_count = 0
    parse_errors = 0
    done = False
    text_parts: List[str] = []
    parsed_objects: List[Dict[str, Any]] = []
    usage: Dict[str, Any] = {}
    raw_parts: List[str] = []
    raw_chars = 0
    raw_truncated = False
    cancelled = False

    def append_raw(line: str) -> None:
        nonlocal raw_chars, raw_truncated
        if raw_chars >= MAX_SSE_RAW_LOG_CHARS:
            raw_truncated = True
            return
        left = MAX_SSE_RAW_LOG_CHARS - raw_chars
        clipped = line[:left]
        raw_parts.append(clipped)
        raw_chars += len(clipped) + 1
        if len(line) > len(clipped):
            raw_truncated = True

    def flush_event() -> None:
        nonlocal event_buffer, event_count, parse_errors, done, usage
        if not event_buffer:
            return
        payload = parse_sse_event_payload(event_buffer)
        event_buffer = []
        if not payload:
            return
        if payload == "[DONE]":
            done = True
            return
        try:
            event_obj = json.loads(payload)
        except Exception:
            parse_errors += 1
            return

        if not isinstance(event_obj, dict):
            parse_errors += 1
            return

        event_count += 1
        if len(parsed_objects) < MAX_SSE_OBJECTS:
            parsed_objects.append(event_obj)
        if isinstance(event_obj.get("usage"), dict):
            usage = event_obj["usage"]

        choices = event_obj.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)

    for raw_line in response.iter_lines(decode_unicode=True):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break
        if raw_line is None:
            continue
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        append_raw(line)
        line = line.rstrip("\r")
        if line == "":
            flush_event()
            if done:
                break
            continue
        event_buffer.append(line)
    flush_event()

    raw_text = "\n".join(raw_parts)
    if raw_truncated:
        raw_text = f"{raw_text}\n...<SSE日志已截断>..."

    return {
        "ok": bool(event_count > 0 or done),
        "event_count": event_count,
        "parse_errors": parse_errors,
        "done": done,
        "cancelled": cancelled,
        "text": "".join(text_parts),
        "objects": parsed_objects,
        "usage": usage,
        "raw_text": raw_text,
    }


def build_sse_summary(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "object": "sse_summary",
        "event_count": parsed.get("event_count", 0),
        "parse_errors": parsed.get("parse_errors", 0),
        "done": parsed.get("done", False),
        "usage": parsed.get("usage", {}),
        "text_preview": (parsed.get("text", "") or "")[:4000],
    }


def redact_sensitive_text(text: str) -> str:
    if not text:
        return ""
    redacted = text
    redacted = re.sub(r'("api_key"\s*:\s*")([^"]+)(")', r"\1***\3", redacted, flags=re.IGNORECASE)
    redacted = re.sub(
        r'(authorization"\s*:\s*")([^"]+)(")',
        r"\1***\3",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1***", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"\bsk-[A-Za-z0-9._-]{12,}\b", "***", redacted)
    return redacted


def compose_debug_log(parts: List[str]) -> str:
    merged = "\n\n".join([part for part in parts if part and str(part).strip()])
    merged = redact_sensitive_text(merged)
    return merged[:MAX_DEBUG_LOG_CHARS]


def collect_image_candidates(obj: Any, result: List[Dict[str, str]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if isinstance(value, str):
                if value.lower().startswith("data:image/"):
                    result.append({"type": "data_url", "value": value, "key": key_lower})
                elif key_lower in {"b64_json", "image_base64", "base64", "image_b64"} and len(value) > 100:
                    result.append({"type": "base64", "value": value, "key": key_lower})
                elif key_lower in {"url", "image_url"} and is_probable_image_url(value):
                    result.append(build_url_candidate(value, key_lower))
                else:
                    for candidate in parse_markdown_image_urls(value):
                        result.append(build_url_candidate(candidate, f"{key_lower}_markdown"))
                    for candidate in parse_plain_image_urls(value):
                        result.append(build_url_candidate(candidate, f"{key_lower}_text"))
            else:
                collect_image_candidates(value, result)
    elif isinstance(obj, list):
        for item in obj:
            collect_image_candidates(item, result)
    elif isinstance(obj, str):
        for candidate in parse_markdown_image_urls(obj):
            result.append(build_url_candidate(candidate, "raw_markdown"))
        for candidate in parse_plain_image_urls(obj):
            result.append(build_url_candidate(candidate, "raw_text"))


class ApiConfig(BaseModel):
    system_prompt: str = ""
    user_prompt: str = ""
    api_key: str = ""
    api_url: str = "https://api.openai.com/v1/chat/completions"
    max_tokens: int = 65535
    temperature: float = 0.7
    model_name: str = "grok-imagine-1.0-edit"
    concurrency_limit: int = 3
    max_retries: int = 2
    retry_interval: float = 1.5
    timeout_seconds: int = 120
    api_mode: str = "auto"
    token_field: str = "auto"
    unlock_local_api_url: bool = False
    unlock_local_download_url: bool = False

    def sanitize(self) -> Dict[str, Any]:
        api_url = self.api_url.strip()
        if not api_url:
            raise ValueError("api_url 不能为空")
        api_key = (self.api_key or "").strip()
        if api_key == PASSWORD_PLACEHOLDER:
            api_key = ""
        return {
            "system_prompt": self.system_prompt or "",
            "user_prompt": self.user_prompt or "",
            "api_key": api_key,
            "api_url": api_url,
            "max_tokens": clamp_int(int(self.max_tokens), 1, 256000),
            "temperature": clamp_float(float(self.temperature), 0.0, 2.0),
            "model_name": self.model_name.strip() or "gpt-4o",
            "concurrency_limit": clamp_int(int(self.concurrency_limit), 1, 64),
            "max_retries": clamp_int(int(self.max_retries), 0, 10),
            "retry_interval": clamp_float(float(self.retry_interval), 0.1, 60.0),
            "timeout_seconds": clamp_int(int(self.timeout_seconds), 5, 1800),
            "api_mode": self.api_mode if self.api_mode in {"auto", "chat_completions", "responses"} else "auto",
            "token_field": self.token_field if self.token_field in {"auto", "max_tokens", "max_completion_tokens", "max_output_tokens", "none"} else "auto",
            "unlock_local_api_url": bool(self.unlock_local_api_url),
            "unlock_local_download_url": bool(self.unlock_local_download_url),
        }


class StartBatchRequest(BaseModel):
    input_path: str
    config: ApiConfig


class SaveCurrentConfigRequest(BaseModel):
    config: ApiConfig


class SavePresetRequest(BaseModel):
    name: str
    config: ApiConfig


class ApplyPresetRequest(BaseModel):
    name: str


class SaveAllRequest(BaseModel):
    output_dir: str
    filename_template: str = "{src_stem}_{result_index}"
    replace_from: str = ""
    replace_to: str = ""
    create_txt: bool = False
    txt_template: str = "{src_name}"
    only_selected: bool = True
    only_first_if_multiple: bool = False


class SaveSingleRequest(BaseModel):
    output_dir: str
    filename_template: str = "{src_stem}_{result_index}"
    replace_from: str = ""
    replace_to: str = ""
    create_txt: bool = False
    txt_template: str = "{src_name}"


class AppState:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.cache_dir = self.data_dir / "cache"
        self.state_json = self.data_dir / "state.json"
        self.preset_json = self.data_dir / "presets.json"
        ensure_dir(self.cache_dir)

        self.lock = threading.RLock()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_order: List[str] = []
        self.running = False
        self.stop_requested = False
        self.run_token = 0
        self.active_single_retry_task_ids: set[str] = set()
        self.current_config = ApiConfig().sanitize()
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        with self.lock:
            raw_state = load_json(self.state_json, {})
            raw_tasks = raw_state.get("tasks", {})
            self.tasks = {}
            for task_id, task in raw_tasks.items():
                if not isinstance(task, dict):
                    continue
                task.setdefault("results", [])
                task.setdefault("status", "pending")
                task.setdefault("attempts", 0)
                task.setdefault("last_error", "")
                task.setdefault("last_warning", "")
                task.setdefault("http_status", 0)
                task.setdefault("updated_at", now_ts())
                self.tasks[task_id] = task

            self.task_order = [task_id for task_id in raw_state.get("task_order", []) if task_id in self.tasks]
            for task_id in self.tasks:
                if task_id not in self.task_order:
                    self.task_order.append(task_id)

            config = raw_state.get("current_config")
            if isinstance(config, dict):
                try:
                    self.current_config = ApiConfig(**config).sanitize()
                except Exception:
                    self.current_config = ApiConfig().sanitize()
            else:
                self.current_config = ApiConfig().sanitize()

            raw_presets = load_json(self.preset_json, {})
            if isinstance(raw_presets, dict):
                self.presets = {}
                for preset_name, preset_config in raw_presets.items():
                    if not isinstance(preset_name, str) or not isinstance(preset_config, dict):
                        continue
                    try:
                        self.presets[preset_name] = ApiConfig(**preset_config).sanitize()
                    except Exception:
                        continue
            else:
                self.presets = {}

    def save_state(self) -> None:
        with self.lock:
            payload = {
                "tasks": self.tasks,
                "task_order": self.task_order,
                "current_config": self.current_config,
            }
        save_json(self.state_json, payload)

    def save_presets(self) -> None:
        with self.lock:
            payload = self.presets
        save_json(self.preset_json, payload)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            tasks = []
            status_counter = {
                "pending": 0,
                "processing": 0,
                "done": 0,
                "partial_success": 0,
                "failed": 0,
                "stopped": 0,
            }
            for task_id in self.task_order:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                task_status = task.get("status", "pending")
                if task_status not in status_counter:
                    task_status = "pending"
                status_counter[task_status] += 1
                result_items = []
                for result in task.get("results", []):
                    result_items.append({
                        "id": result.get("id"),
                        "selected": bool(result.get("selected", True)),
                        "deleted": bool(result.get("deleted", False)),
                        "width": result.get("width", 0),
                        "height": result.get("height", 0),
                        "source_type": result.get("source_type", ""),
                        "hash": result.get("hash", ""),
                        "image_url": f"/api/tasks/{task_id}/results/{result.get('id')}/image?t={task.get('updated_at', 0)}",
                    })
                tasks.append({
                    "id": task_id,
                    "source_path": task.get("source_path", ""),
                    "source_name": task.get("source_name", ""),
                    "status": task.get("status", "pending"),
                    "attempts": task.get("attempts", 0),
                    "http_status": task.get("http_status", 0),
                    "last_error": task.get("last_error", ""),
                    "last_warning": task.get("last_warning", ""),
                    "has_debug_log": bool(task.get("debug_log", "")),
                    "updated_at": task.get("updated_at", 0),
                    "source_image_url": f"/api/tasks/{task_id}/source?t={task.get('updated_at', 0)}",
                    "result_count": len(task.get("results", [])),
                    "selected_count": len([x for x in task.get("results", []) if x.get("selected", True) and not x.get("deleted", False)]),
                    "results": result_items,
                })
            return {
                "running": self.running,
                "stop_requested": self.stop_requested,
                "single_retry_running": len(self.active_single_retry_task_ids),
                "task_count": len(tasks),
                "progress": {
                    "total": len(tasks),
                    "pending": status_counter["pending"],
                    "processing": status_counter["processing"],
                    "done": status_counter["done"],
                    "partial_success": status_counter["partial_success"],
                    "failed": status_counter["failed"],
                    "stopped": status_counter["stopped"],
                    "finished": status_counter["done"] + status_counter["partial_success"] + status_counter["failed"] + status_counter["stopped"],
                    "percent": round(
                        (
                            (status_counter["done"] + status_counter["partial_success"] + status_counter["failed"] + status_counter["stopped"])
                            / len(tasks)
                            * 100
                        ),
                        2,
                    )
                    if tasks
                    else 0.0,
                },
                "tasks": tasks,
                "current_config": sanitize_config_for_client(self.current_config),
                "presets": {name: sanitize_config_for_client(cfg) for name, cfg in self.presets.items()},
            }


def collect_referenced_result_paths(state: AppState) -> set:
    referenced = set()
    with state.lock:
        for task in state.tasks.values():
            if not isinstance(task, dict):
                continue
            for result in task.get("results", []):
                if not isinstance(result, dict):
                    continue
                raw_path = (result.get("file_path") or "").strip()
                if not raw_path:
                    continue
                try:
                    referenced.add(str(Path(raw_path).resolve()))
                except Exception:
                    continue
    return referenced


def cleanup_orphan_cache_files(state: AppState, reason: str = "", remove_all_unreferenced: bool = False) -> Dict[str, Any]:
    cache_dir = state.cache_dir
    if not cache_dir.exists() or not cache_dir.is_dir():
        return {"removed": 0, "freed_bytes": 0, "reason": reason or "unknown"}

    referenced = collect_referenced_result_paths(state)
    entries: List[Dict[str, Any]] = []
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except Exception:
            continue
        resolved = str(path.resolve())
        entries.append(
            {
                "path": path,
                "resolved": resolved,
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "referenced": resolved in referenced,
            }
        )

    removed = 0
    freed_bytes = 0
    total_bytes = sum(item["size"] for item in entries)
    now_value = time.time()
    ttl_seconds = CACHE_TTL_HOURS * 3600 if CACHE_TTL_HOURS > 0 else 0

    # 1) 先按策略清理未被引用缓存（可强制全清）
    for item in entries:
        if item["referenced"]:
            continue
        if not remove_all_unreferenced:
            if ttl_seconds <= 0:
                continue
            age = now_value - item["mtime"]
            if age < ttl_seconds:
                continue
        try:
            item["path"].unlink(missing_ok=True)
            removed += 1
            freed_bytes += item["size"]
            total_bytes -= item["size"]
            item["deleted"] = True
        except Exception:
            continue

    # 2) 再按总大小上限清理未被引用缓存（从最旧开始）
    if CACHE_MAX_BYTES > 0 and total_bytes > CACHE_MAX_BYTES:
        candidates = sorted(
            [item for item in entries if not item.get("deleted") and not item["referenced"]],
            key=lambda x: x["mtime"],
        )
        for item in candidates:
            if total_bytes <= CACHE_MAX_BYTES:
                break
            try:
                item["path"].unlink(missing_ok=True)
                removed += 1
                freed_bytes += item["size"]
                total_bytes -= item["size"]
            except Exception:
                continue

    return {
        "removed": removed,
        "freed_bytes": freed_bytes,
        "reason": reason or "unknown",
        "remove_all_unreferenced": bool(remove_all_unreferenced),
        "cache_total_bytes": total_bytes,
    }


def get_resource_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_data_dir(runtime_base_dir: Path) -> Path:
    raw = os.getenv("APP_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return runtime_base_dir / "web_data"


RESOURCE_BASE_DIR = get_resource_base_dir()
RUNTIME_BASE_DIR = get_runtime_base_dir()
DATA_DIR = get_data_dir(RUNTIME_BASE_DIR)
TEMPLATE_DIR = RESOURCE_BASE_DIR / "templates"
STATIC_DIR = RESOURCE_BASE_DIR / "static"
STATE = AppState(DATA_DIR)
try:
    cleanup_orphan_cache_files(STATE, reason="startup", remove_all_unreferenced=True)
except Exception:
    pass
ALLOWED_ORIGINS = parse_allowed_origins()

app = FastAPI(title="图片批处理调度台", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
ACTIVE_RUN_TASK: Any = None
ACTIVE_RUN_CANCEL_EVENT: Optional[threading.Event] = None


def register_active_run(task: Any, cancel_event: threading.Event) -> None:
    global ACTIVE_RUN_TASK, ACTIVE_RUN_CANCEL_EVENT
    with STATE.lock:
        ACTIVE_RUN_TASK = task
        ACTIVE_RUN_CANCEL_EVENT = cancel_event


def finalize_run(run_token: int, running_task: Any) -> None:
    global ACTIVE_RUN_TASK, ACTIVE_RUN_CANCEL_EVENT
    with STATE.lock:
        if ACTIVE_RUN_TASK is running_task:
            ACTIVE_RUN_TASK = None
            ACTIVE_RUN_CANCEL_EVENT = None
        if STATE.run_token == run_token:
            STATE.running = False
            STATE.stop_requested = False


def discover_images(input_path: str) -> List[Path]:
    target = Path(input_path).expanduser()
    if not target.exists():
        raise ValueError("输入路径不存在")

    if target.is_file():
        if target.suffix.lower() in IMAGE_EXTENSIONS:
            return [target.resolve()]
        raise ValueError("输入文件不是支持的图片格式")

    images = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path.resolve())
    if not images:
        raise ValueError("目录中未找到图片文件")
    return images


def build_payload(config: Dict[str, Any], source_image_b64: str) -> Dict[str, Any]:
    mode = detect_mode(config["api_mode"], config["api_url"])
    token_field = select_token_field(config["token_field"], mode)

    if mode == "responses":
        payload: Dict[str, Any] = {
            "model": config["model_name"],
            "temperature": config["temperature"],
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": config["system_prompt"]}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": config["user_prompt"] or ""},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{source_image_b64}"},
                    ],
                },
            ],
        }
    else:
        payload = {
            "model": config["model_name"],
            "temperature": config["temperature"],
            "messages": [
                {"role": "system", "content": config["system_prompt"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": config["user_prompt"] or ""},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{source_image_b64}"}},
                    ],
                },
            ],
        }

    if token_field != "none":
        payload[token_field] = config["max_tokens"]
    return payload


def should_retry(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS


def download_candidate_content(
    candidate: Dict[str, str],
    timeout_seconds: int,
    max_retries: int,
    retry_interval: float,
    cancel_event: Optional[threading.Event] = None,
    allow_private_download: bool = False,
) -> bytes:
    candidate_type = candidate["type"]
    value = candidate["value"]

    if cancel_event and cancel_event.is_set():
        raise ValueError("任务被紧急停止")

    if candidate_type == "data_url":
        content = decode_data_url(value)
        if len(content) > MAX_CANDIDATE_IMAGE_BYTES:
            raise ValueError(f"解码失败: data_url 体积超过限制({len(content)} bytes)")
        return content
    if candidate_type == "base64":
        content = base64.b64decode(normalize_base64(value))
        if len(content) > MAX_CANDIDATE_IMAGE_BYTES:
            raise ValueError(f"解码失败: base64 体积超过限制({len(content)} bytes)")
        return content

    if candidate_type == "url":
        if value.lower().startswith("data:image/"):
            content = decode_data_url(value)
            if len(content) > MAX_CANDIDATE_IMAGE_BYTES:
                raise ValueError(f"解码失败: data_url 体积超过限制({len(content)} bytes)")
            return content
        last_error = ""
        for attempt in range(max_retries + 1):
            if cancel_event and cancel_event.is_set():
                raise ValueError("任务被紧急停止")
            try:
                current_url = value
                for _ in range(MAX_HTTP_REDIRECTS + 1):
                    validate_network_url(
                        current_url,
                        purpose="下载URL",
                        allow_private=allow_private_download,
                        allowlist=DOWNLOAD_URL_HOST_ALLOWLIST,
                    )
                    with requests.get(current_url, timeout=timeout_seconds, stream=True, allow_redirects=False) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location", "").strip()
                            if not location:
                                raise ValueError("下载失败: 重定向缺少 Location")
                            current_url = urljoin(current_url, location)
                            continue

                        if response.status_code >= 400:
                            if attempt < max_retries and should_retry(response.status_code):
                                time.sleep(retry_interval)
                                break
                            raise ValueError(f"下载失败: HTTP {response.status_code}")

                        content_length = response.headers.get("Content-Length")
                        if content_length and str(content_length).isdigit() and int(content_length) > MAX_CANDIDATE_IMAGE_BYTES:
                            raise ValueError(f"下载失败: 文件过大({content_length} bytes)")

                        chunks: List[bytes] = []
                        total = 0
                        for chunk in response.iter_content(chunk_size=65536):
                            if cancel_event and cancel_event.is_set():
                                raise ValueError("任务被紧急停止")
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > MAX_CANDIDATE_IMAGE_BYTES:
                                raise ValueError("下载失败: 文件过大(超过限制)")
                            chunks.append(chunk)
                        return b"".join(chunks)
                else:
                    raise ValueError(f"下载失败: 重定向次数超过上限({MAX_HTTP_REDIRECTS})")
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(retry_interval)
        if last_error.startswith("下载失败"):
            raise ValueError(last_error)
        raise ValueError(f"下载失败: {last_error}")

    raise ValueError("不支持的图片候选类型")


def unlink_paths(paths: List[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            continue


def save_content_as_image(cache_dir: Path, task_id: str, content: bytes, source_type: str) -> Dict[str, Any]:
    image_format, width, height = read_image_info(content)
    ext = image_format if image_format != "jpeg" else "jpg"
    content_hash = hashlib.sha256(content).hexdigest()
    filename = f"{task_id}_{uuid.uuid4().hex}.{ext}"
    target_path = cache_dir / filename
    staged_path = cache_dir / f".{filename}.{uuid.uuid4().hex}.part"
    staged_path.write_bytes(content)
    return {
        "id": uuid.uuid4().hex,
        "file_path": str(target_path),
        "_staged_file_path": str(staged_path),
        "width": width,
        "height": height,
        "source_type": source_type,
        "hash": content_hash,
        "selected": True,
        "deleted": False,
        "created_at": now_ts(),
    }


def commit_staged_image_files(images: List[Dict[str, Any]]) -> List[Path]:
    committed: List[Path] = []
    try:
        for image in images:
            staged_raw = (image.get("_staged_file_path") or "").strip()
            if not staged_raw:
                continue
            staged_path = Path(staged_raw)
            final_path = Path(str(image.get("file_path", "") or ""))
            if not staged_path.exists():
                raise ValueError(f"暂存文件不存在: {staged_path}")
            os.replace(staged_path, final_path)
            committed.append(final_path)
            image.pop("_staged_file_path", None)
    except Exception:
        # 回滚已提交与残余暂存，避免部分成功
        unlink_paths(committed)
        remaining_staged: List[Path] = []
        for image in images:
            staged_raw = (image.get("_staged_file_path") or "").strip()
            if staged_raw:
                remaining_staged.append(Path(staged_raw))
        unlink_paths(remaining_staged)
        raise
    return committed


def call_api_and_extract_images(
    task: Dict[str, Any],
    config: Dict[str, Any],
    cache_dir: Path,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    if cancel_event and cancel_event.is_set():
        return {
            "ok": False,
            "http_status": 0,
            "error": "任务被紧急停止",
            "debug_log": compose_debug_log(["任务开始前已收到停止信号"]),
        }

    try:
        validate_network_url(
            config.get("api_url", ""),
            purpose="API URL",
            allow_private=is_private_api_url_unlocked(config),
            allowlist=API_URL_HOST_ALLOWLIST,
        )
    except Exception as exc:
        return {
            "ok": False,
            "http_status": 0,
            "error": f"API地址校验失败: {exc}",
            "debug_log": compose_debug_log([f"API URL: {config.get('api_url', '')}", f"校验错误: {exc}"]),
        }

    source_path = Path(task["source_path"])
    try:
        source_size = source_path.stat().st_size
    except Exception:
        source_size = 0
    if source_size > MAX_SOURCE_IMAGE_BYTES:
        return {
            "ok": False,
            "http_status": 0,
            "error": f"源图片过大({source_size} bytes)，超过限制 {MAX_SOURCE_IMAGE_BYTES} bytes",
            "debug_log": compose_debug_log([f"源图片大小: {source_size} bytes", f"限制: {MAX_SOURCE_IMAGE_BYTES} bytes"]),
        }
    source_bytes = source_path.read_bytes()
    source_b64 = base64.b64encode(source_bytes).decode("utf-8")
    payload = build_payload(config, source_b64)

    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"

    response_json = None
    sse_text = ""
    http_status = 0
    last_error = ""
    debug_parts: List[str] = []
    response_headers: Dict[str, str] = {}

    for attempt in range(config["max_retries"] + 1):
        if cancel_event and cancel_event.is_set():
            return {
                "ok": False,
                "http_status": http_status,
                "error": "任务被紧急停止",
                "debug_log": compose_debug_log(debug_parts + ["请求前检测到停止信号"]),
            }
        try:
            with requests.post(
                config["api_url"],
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=config["timeout_seconds"],
                stream=True,
            ) as response:
                http_status = response.status_code
                response_headers = {k: v for k, v in response.headers.items()}
                debug_parts.append(f"HTTP状态: {http_status}")
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    if attempt < config["max_retries"] and should_retry(response.status_code):
                        time.sleep(config["retry_interval"])
                        continue
                    return {
                        "ok": False,
                        "http_status": http_status,
                        "error": last_error,
                        "debug_log": compose_debug_log(debug_parts + [f"失败响应片段:\n{response.text[:3000]}"]),
                    }

                content_type = (response.headers.get("Content-Type") or "").lower()
                is_sse = "text/event-stream" in content_type
                debug_parts.append(f"Content-Type: {content_type or 'unknown'}")
                if is_sse:
                    parsed_sse = parse_sse_from_response(response, cancel_event=cancel_event)
                    if parsed_sse.get("cancelled"):
                        return {
                            "ok": False,
                            "http_status": http_status,
                            "error": "任务被紧急停止",
                            "debug_log": compose_debug_log(
                                debug_parts + ["SSE处理中收到停止信号", f"SSE数据流:\n{parsed_sse.get('raw_text', '')}"]
                            ),
                        }
                    if parsed_sse.get("ok"):
                        sse_text = parsed_sse.get("text", "")
                        response_json = build_sse_summary(parsed_sse)
                        debug_parts.append(
                            f"SSE解析: event_count={parsed_sse.get('event_count', 0)}, parse_errors={parsed_sse.get('parse_errors', 0)}, done={parsed_sse.get('done', False)}"
                        )
                        debug_parts.append(f"SSE数据流:\n{parsed_sse.get('raw_text', '')}")
                    else:
                        raw_body = (parsed_sse.get("raw_text") or "").strip()
                        if not raw_body:
                            raise ValueError("SSE 响应为空")
                        try:
                            response_json = json.loads(raw_body)
                        except Exception as exc:
                            raise ValueError(f"SSE 解析失败: {exc}") from exc
                else:
                    raw_text = response.text
                    try:
                        response_json = json.loads(raw_text)
                    except Exception:
                        parsed_sse = parse_sse_from_text(raw_text)
                        if parsed_sse.get("ok"):
                            sse_text = parsed_sse.get("text", "")
                            response_json = build_sse_summary(parsed_sse)
                            debug_parts.append(
                                f"SSE文本解析: event_count={parsed_sse.get('event_count', 0)}, parse_errors={parsed_sse.get('parse_errors', 0)}, done={parsed_sse.get('done', False)}"
                            )
                            debug_parts.append(f"SSE数据流:\n{(parsed_sse.get('raw_text', '') or '')[:MAX_SSE_RAW_LOG_CHARS]}")
                        else:
                            raise
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < config["max_retries"]:
                time.sleep(config["retry_interval"])

    if response_json is None:
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"请求失败: {last_error}",
            "debug_log": compose_debug_log(debug_parts + [f"异常: {last_error}"]),
        }

    candidates: List[Dict[str, str]] = []
    collect_image_candidates(response_json, candidates)
    if sse_text:
        useful_text = extract_after_think_tag(sse_text)
        for candidate in parse_markdown_image_urls(useful_text):
            candidates.append(build_url_candidate(candidate, "sse_markdown"))
        for candidate in parse_plain_image_urls(useful_text):
            candidates.append(build_url_candidate(candidate, "sse_text"))
    deduped: List[Dict[str, str]] = []
    seen_keys = set()
    for candidate in candidates:
        key = f"{candidate.get('type','')}::{candidate.get('value','')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(candidate)
    debug_parts.append(f"候选图片数量: 原始={len(candidates)}, 去重后={len(deduped)}")

    if not deduped:
        return {
            "ok": False,
            "http_status": http_status,
            "error": "接口返回中未解析到图片",
            "raw_json": json.dumps(response_json, ensure_ascii=False),
            "debug_log": compose_debug_log(
                debug_parts
                + [
                    f"响应头:\n{json.dumps(response_headers, ensure_ascii=False, indent=2)}",
                    f"响应JSON:\n{json.dumps(response_json, ensure_ascii=False, indent=2)[:300000]}",
                ]
            ),
        }

    images: List[Dict[str, Any]] = []
    staged_paths: List[Path] = []
    created_paths: List[Path] = []
    non_fatal_errors = []
    allow_private_download = is_private_download_url_unlocked(config)
    for candidate in deduped:
        if cancel_event and cancel_event.is_set():
            unlink_paths(staged_paths + created_paths)
            return {
                "ok": False,
                "http_status": http_status,
                "error": "任务被紧急停止",
                "raw_json": json.dumps(response_json, ensure_ascii=False),
                "debug_log": compose_debug_log(debug_parts + ["下载候选图片时收到停止信号，已清理暂存/落盘文件"]),
            }
        try:
            content = download_candidate_content(
                candidate=candidate,
                timeout_seconds=config["timeout_seconds"],
                max_retries=config["max_retries"],
                retry_interval=config["retry_interval"],
                cancel_event=cancel_event,
                allow_private_download=allow_private_download,
            )
            if cancel_event and cancel_event.is_set():
                raise ValueError("任务被紧急停止")
            image_result = save_content_as_image(cache_dir, task["id"], content, candidate.get("type", "unknown"))
            images.append(image_result)
            staged_raw = (image_result.get("_staged_file_path") or "").strip()
            if staged_raw:
                staged_paths.append(Path(staged_raw))
        except Exception as exc:
            if "任务被紧急停止" in str(exc):
                unlink_paths(staged_paths + created_paths)
                return {
                    "ok": False,
                    "http_status": http_status,
                    "error": "任务被紧急停止",
                    "raw_json": json.dumps(response_json, ensure_ascii=False),
                    "debug_log": compose_debug_log(debug_parts + ["下载过程中收到停止信号，已清理暂存/落盘文件"]),
                }
            non_fatal_errors.append(str(exc))

    if not images:
        unlink_paths(staged_paths)
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"解析到候选图片但全部下载/解码失败: {' | '.join(non_fatal_errors[:2])}",
            "raw_json": json.dumps(response_json, ensure_ascii=False),
            "debug_log": compose_debug_log(
                debug_parts
                + [
                    f"候选下载错误:\n{json.dumps(non_fatal_errors[:100], ensure_ascii=False, indent=2)}",
                    f"响应JSON:\n{json.dumps(response_json, ensure_ascii=False, indent=2)[:300000]}",
                ]
            ),
        }

    try:
        created_paths = commit_staged_image_files(images)
    except Exception as exc:
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"结果提交失败: {exc}",
            "raw_json": json.dumps(response_json, ensure_ascii=False),
            "debug_log": compose_debug_log(debug_parts + [f"两阶段提交失败: {exc}"]),
        }

    if cancel_event and cancel_event.is_set():
        unlink_paths(created_paths)
        return {
            "ok": False,
            "http_status": http_status,
            "error": "任务被紧急停止",
            "raw_json": json.dumps(response_json, ensure_ascii=False),
            "debug_log": compose_debug_log(debug_parts + ["提交后检测到停止信号，已回滚本次落盘文件"]),
        }

    debug_parts.append(f"下载成功图片数: {len(images)}")
    if non_fatal_errors:
        debug_parts.append(f"非致命下载错误:\n{json.dumps(non_fatal_errors[:100], ensure_ascii=False, indent=2)}")

    return {
        "ok": True,
        "http_status": http_status,
        "images": images,
        "raw_json": json.dumps(response_json, ensure_ascii=False),
        "error": "",
        "warning": " | ".join(non_fatal_errors[:2]) if non_fatal_errors else "",
        "debug_log": compose_debug_log(
            debug_parts
            + [
                f"响应头:\n{json.dumps(response_headers, ensure_ascii=False, indent=2)}",
                f"响应JSON:\n{json.dumps(response_json, ensure_ascii=False, indent=2)[:300000]}",
            ]
        ),
    }


async def process_single_task(
    task_id: str,
    config: Dict[str, Any],
    run_token: int,
    cancel_event: Optional[threading.Event],
    preserve: bool = False,
) -> None:
    with STATE.lock:
        if run_token != STATE.run_token or STATE.stop_requested:
            return
        task = STATE.tasks.get(task_id)
        if not task:
            return
        task["status"] = "processing"
        task["attempts"] = int(task.get("attempts", 0)) + 1
        task["updated_at"] = now_ts()
        task["last_error"] = ""
        task["last_warning"] = ""
    STATE.save_state()

    try:
        result = await asyncio.to_thread(call_api_and_extract_images, task, config, STATE.cache_dir, cancel_event)
    except Exception as exc:
        with STATE.lock:
            if run_token != STATE.run_token:
                return
            current = STATE.tasks.get(task_id)
            if current:
                current["status"] = "failed"
                current["last_error"] = f"运行异常: {exc}"
                current["updated_at"] = now_ts()
        STATE.save_state()
        return

    should_return_after_save = False
    should_save_state = False
    with STATE.lock:
        if run_token != STATE.run_token:
            return
        current = STATE.tasks.get(task_id)
        if not current:
            return
        if STATE.stop_requested:
            for image_item in result.get("images", []):
                raw_path = (image_item.get("file_path") or "").strip()
                if not raw_path:
                    continue
                file_path = Path(raw_path)
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    continue
            if current.get("status") in {"pending", "processing"}:
                current["status"] = "stopped"
            current["last_error"] = current.get("last_error") or "任务被紧急停止"
            current["updated_at"] = now_ts()
            should_save_state = True
            should_return_after_save = True
        else:
            current["http_status"] = result.get("http_status", 0)
            current["raw_json"] = result.get("raw_json", "")
            debug_log = result.get("debug_log", "")
            if debug_log:
                attempt_no = int(current.get("attempts", 0))
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts()))
                header = f"[尝试#{attempt_no} {stamp}]"
                old_debug = current.get("debug_log", "")
                merged_debug = f"{old_debug}\n\n{header}\n{debug_log}".strip() if old_debug else f"{header}\n{debug_log}"
                current["debug_log"] = merged_debug[-MAX_TASK_DEBUG_LOG_CHARS:]

            if not result.get("ok"):
                current["status"] = "failed"
                current["last_error"] = result.get("error", "未知错误")
                current["last_warning"] = ""
                current["updated_at"] = now_ts()
                should_save_state = True
                should_return_after_save = True
            else:
                old_results = current.get("results", []) if preserve else []
                old_hashes = {item.get("hash") for item in old_results}
                new_results = []
                for item in result.get("images", []):
                    if item.get("hash") in old_hashes:
                        continue
                    new_results.append(item)
                    old_hashes.add(item.get("hash"))

                current["results"] = old_results + new_results
                current["status"] = "done" if new_results else "partial_success"
                current["last_error"] = result.get("error", "")
                current["last_warning"] = result.get("warning", "")
                current["updated_at"] = now_ts()
                should_save_state = True
    if should_save_state:
        STATE.save_state()
    if should_return_after_save:
        return


async def run_batch(task_ids: List[str], config: Dict[str, Any], run_token: int, cancel_event: threading.Event) -> None:
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency_limit", 1))))

    async def worker(task_id: str) -> None:
        async with semaphore:
            with STATE.lock:
                if run_token != STATE.run_token or STATE.stop_requested or cancel_event.is_set():
                    task = STATE.tasks.get(task_id)
                    if task and task.get("status") in {"pending", "processing"}:
                        task["status"] = "stopped"
                        task["last_error"] = task.get("last_error") or "任务被紧急停止"
                        task["updated_at"] = now_ts()
                    return
            await process_single_task(task_id, config, run_token, cancel_event, preserve=False)

    try:
        await asyncio.gather(*(worker(task_id) for task_id in task_ids))
    except asyncio.CancelledError:
        pass
    finally:
        finalize_run(run_token, asyncio.current_task())
        STATE.save_state()
        await asyncio.to_thread(cleanup_orphan_cache_files, STATE, f"run_finalize:{run_token}", True)


async def run_failed_retry_batch(
    task_ids: List[str], config: Dict[str, Any], run_token: int, cancel_event: threading.Event
) -> None:
    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency_limit", 1))))

    async def worker(task_id: str) -> None:
        async with semaphore:
            with STATE.lock:
                if run_token != STATE.run_token or STATE.stop_requested or cancel_event.is_set():
                    task = STATE.tasks.get(task_id)
                    if task and task.get("status") in {"pending", "processing"}:
                        task["status"] = "stopped"
                        task["last_error"] = task.get("last_error") or "任务被紧急停止"
                        task["updated_at"] = now_ts()
                    return
            await process_single_task(task_id, config, run_token, cancel_event, preserve=True)

    try:
        await asyncio.gather(*(worker(task_id) for task_id in task_ids))
    except asyncio.CancelledError:
        pass
    finally:
        finalize_run(run_token, asyncio.current_task())
        STATE.save_state()
        await asyncio.to_thread(cleanup_orphan_cache_files, STATE, f"retry_failed_finalize:{run_token}", True)


def get_task_or_404(task_id: str) -> Dict[str, Any]:
    with STATE.lock:
        task = STATE.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        return task


def reserve_single_retry_slot(task_id: str) -> int:
    with STATE.lock:
        if task_id in STATE.active_single_retry_task_ids:
            raise HTTPException(status_code=409, detail="该任务正在重试中，请勿重复提交")
        active_count = len(STATE.active_single_retry_task_ids)
        if active_count >= SINGLE_RETRY_MAX_CONCURRENCY:
            raise HTTPException(status_code=409, detail=f"单任务重试并发已达上限({SINGLE_RETRY_MAX_CONCURRENCY})，请稍后再试")
        STATE.active_single_retry_task_ids.add(task_id)
        return len(STATE.active_single_retry_task_ids)


def release_single_retry_slot(task_id: str) -> int:
    with STATE.lock:
        STATE.active_single_retry_task_ids.discard(task_id)
        return len(STATE.active_single_retry_task_ids)


def get_result_or_404(task_id: str, result_id: str) -> Dict[str, Any]:
    task = get_task_or_404(task_id)
    for result in task.get("results", []):
        if result.get("id") == result_id:
            return result
    raise HTTPException(status_code=404, detail="结果不存在")


def is_loopback_client(host: str) -> bool:
    normalized = (host or "").strip().lower()
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except Exception:
        return False


def enforce_debug_api_access(request: Request) -> None:
    if ALLOW_REMOTE_DEBUG_LOG_API:
        return
    client_host = ""
    if request.client:
        client_host = request.client.host or ""
    if not is_loopback_client(client_host):
        raise HTTPException(
            status_code=403,
            detail="调试日志接口默认仅允许本机访问；如需远程访问请设置 ALLOW_REMOTE_DEBUG_LOG_API=1",
        )


def apply_filename_rule(
    template: str,
    src_path: Path,
    task_id: str,
    result_id: str,
    result_index: int,
    global_index: int,
    replace_from: str,
    replace_to: str,
) -> str:
    src_stem = src_path.stem
    if replace_from:
        src_stem = src_stem.replace(replace_from, replace_to)
    return template.format(
        src_stem=src_stem,
        src_name=src_path.name,
        src_ext=src_path.suffix.lstrip("."),
        task_id=task_id,
        result_id=result_id,
        result_index=result_index,
        global_index=global_index,
    )


def unique_target_path(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("无法生成唯一文件名")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    index_path = TEMPLATE_DIR / "index.html"
    if not index_path.exists():
        return "<h3>前端文件不存在，请确认 templates/index.html 已创建。</h3>"
    return index_path.read_text(encoding="utf-8")


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return STATE.snapshot()


@app.get("/api/tasks/{task_id}/debug")
async def get_task_debug(task_id: str, request: Request) -> Dict[str, Any]:
    enforce_debug_api_access(request)
    task = get_task_or_404(task_id)
    debug_log = str(task.get("debug_log", "") or "")
    has_sse = "SSE数据流" in debug_log
    return {
        "ok": True,
        "task_id": task_id,
        "has_debug_log": bool(debug_log),
        "has_sse": has_sse,
        "debug_log": debug_log,
        "updated_at": task.get("updated_at", 0),
    }


@app.post("/api/tasks/start")
async def start_batch(request: StartBatchRequest) -> Dict[str, Any]:
    config = request.config.sanitize()
    input_path = request.input_path.strip()
    if not input_path:
        raise HTTPException(status_code=400, detail="请输入图片路径")

    try:
        images = discover_images(input_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with STATE.lock:
        if STATE.running:
            raise HTTPException(status_code=409, detail="当前已有批处理运行中，请等待完成")
        config = merge_api_key_if_empty(config, STATE.current_config)

        task_ids = []
        for image_path in images:
            task_id = uuid.uuid4().hex
            STATE.tasks[task_id] = {
                "id": task_id,
                "source_path": str(image_path),
                "source_name": image_path.name,
                "status": "pending",
                "attempts": 0,
                "http_status": 0,
                "last_error": "",
                "raw_json": "",
                "results": [],
                "created_at": now_ts(),
                "updated_at": now_ts(),
            }
            STATE.task_order.append(task_id)
            task_ids.append(task_id)

        STATE.running = True
        STATE.stop_requested = False
        STATE.run_token += 1
        run_token = STATE.run_token
        STATE.current_config = config
        cancel_event = threading.Event()
    STATE.save_state()

    running_task = asyncio.create_task(run_batch(task_ids, config, run_token, cancel_event))
    register_active_run(running_task, cancel_event)
    return {"ok": True, "task_count": len(task_ids)}


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str) -> Dict[str, Any]:
    task = get_task_or_404(task_id)
    reserve_single_retry_slot(task_id)
    try:
        with STATE.lock:
            if task.get("status") == "processing":
                raise HTTPException(status_code=409, detail="任务当前正在处理中，请稍后重试")
            task["status"] = "pending"
            task["last_error"] = ""
            task["updated_at"] = now_ts()
            run_token = STATE.run_token
            config = dict(STATE.current_config)
            cancel_event = ACTIVE_RUN_CANCEL_EVENT if ACTIVE_RUN_CANCEL_EVENT is not None else threading.Event()
        STATE.save_state()
    except Exception:
        release_single_retry_slot(task_id)
        raise

    async def retry_and_release() -> None:
        try:
            await process_single_task(task_id, config, run_token, cancel_event, preserve=True)
        except asyncio.CancelledError:
            pass
        finally:
            release_single_retry_slot(task_id)
            STATE.save_state()
            await asyncio.to_thread(cleanup_orphan_cache_files, STATE, f"single_retry_finalize:{run_token}", True)

    try:
        asyncio.create_task(retry_and_release())
    except Exception:
        release_single_retry_slot(task_id)
        raise
    with STATE.lock:
        current_running = len(STATE.active_single_retry_task_ids)
    return {"ok": True, "task_id": task_id, "single_retry_running": current_running}


@app.post("/api/tasks/{task_id}/retry_clear")
async def retry_task_clear(task_id: str) -> Dict[str, Any]:
    task = get_task_or_404(task_id)
    reserve_single_retry_slot(task_id)
    old_result_paths: List[Path] = []

    try:
        with STATE.lock:
            if task.get("status") == "processing":
                raise HTTPException(status_code=409, detail="任务当前正在处理中，请稍后重试")

            for result in task.get("results", []):
                file_path = Path(result.get("file_path", ""))
                if file_path.exists() and file_path.is_file():
                    old_result_paths.append(file_path)

            task["results"] = []
            task["status"] = "pending"
            task["last_error"] = ""
            task["last_warning"] = ""
            task["raw_json"] = ""
            task["http_status"] = 0
            task["debug_log"] = ""
            task["updated_at"] = now_ts()
            run_token = STATE.run_token
            config = dict(STATE.current_config)
            cancel_event = ACTIVE_RUN_CANCEL_EVENT if ACTIVE_RUN_CANCEL_EVENT is not None else threading.Event()
    except Exception:
        release_single_retry_slot(task_id)
        raise

    removed_count = 0
    for file_path in old_result_paths:
        try:
            file_path.unlink(missing_ok=True)
            removed_count += 1
        except Exception:
            continue

    try:
        STATE.save_state()
    except Exception:
        release_single_retry_slot(task_id)
        raise

    async def retry_and_release() -> None:
        try:
            await process_single_task(task_id, config, run_token, cancel_event, preserve=False)
        except asyncio.CancelledError:
            pass
        finally:
            release_single_retry_slot(task_id)
            STATE.save_state()
            await asyncio.to_thread(cleanup_orphan_cache_files, STATE, f"single_retry_clear_finalize:{run_token}", True)

    try:
        asyncio.create_task(retry_and_release())
    except Exception:
        release_single_retry_slot(task_id)
        raise
    with STATE.lock:
        current_running = len(STATE.active_single_retry_task_ids)
    return {
        "ok": True,
        "task_id": task_id,
        "removed_old_results": removed_count,
        "single_retry_running": current_running,
    }


@app.post("/api/tasks/retry_failed")
async def retry_failed_tasks() -> Dict[str, Any]:
    with STATE.lock:
        if STATE.running:
            raise HTTPException(status_code=409, detail="请等待当前批处理结束后再重试失败任务")

        task_ids = []
        for task_id in STATE.task_order:
            task = STATE.tasks.get(task_id)
            if not task:
                continue
            if task.get("status") == "failed":
                task["status"] = "pending"
                task["last_error"] = ""
                task["updated_at"] = now_ts()
                task_ids.append(task_id)

        if not task_ids:
            return {"ok": True, "retried_count": 0}

        STATE.running = True
        STATE.stop_requested = False
        STATE.run_token += 1
        run_token = STATE.run_token
        config = dict(STATE.current_config)
        cancel_event = threading.Event()
    STATE.save_state()

    running_task = asyncio.create_task(run_failed_retry_batch(task_ids, config, run_token, cancel_event))
    register_active_run(running_task, cancel_event)
    return {"ok": True, "retried_count": len(task_ids)}


@app.post("/api/tasks/stop")
async def emergency_stop() -> Dict[str, Any]:
    global ACTIVE_RUN_TASK, ACTIVE_RUN_CANCEL_EVENT
    with STATE.lock:
        single_retry_running = len(STATE.active_single_retry_task_ids)
        if not STATE.running and single_retry_running <= 0:
            return {"ok": True, "stopped": False, "message": "当前没有运行中的任务", "affected": 0}

        affected = 0
        for task in STATE.tasks.values():
            if task.get("status") in {"pending", "processing"}:
                task["status"] = "stopped"
                task["last_error"] = "任务被紧急停止"
                task["updated_at"] = now_ts()
                affected += 1

        STATE.stop_requested = True
        STATE.running = False
        STATE.run_token += 1
        running_task = ACTIVE_RUN_TASK
        cancel_event = ACTIVE_RUN_CANCEL_EVENT

    if cancel_event:
        cancel_event.set()
    if running_task and not running_task.done():
        running_task.cancel()
    STATE.save_state()
    cleanup_summary = await asyncio.to_thread(cleanup_orphan_cache_files, STATE, "emergency_stop", True)
    return {
        "ok": True,
        "stopped": True,
        "affected": affected,
        "single_retry_running": single_retry_running,
        "cleanup": cleanup_summary,
    }


@app.post("/api/tasks/{task_id}/results/{result_id}/toggle")
async def toggle_result(task_id: str, result_id: str) -> Dict[str, Any]:
    selected = False
    should_save_state = False
    with STATE.lock:
        task = STATE.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        for result in task.get("results", []):
            if result.get("id") == result_id:
                result["selected"] = not bool(result.get("selected", True))
                task["updated_at"] = now_ts()
                selected = bool(result["selected"])
                should_save_state = True
                break
    if should_save_state:
        STATE.save_state()
        return {"ok": True, "selected": selected}
    raise HTTPException(status_code=404, detail="结果不存在")


@app.delete("/api/tasks/{task_id}/results/{result_id}")
async def delete_result(task_id: str, result_id: str) -> Dict[str, Any]:
    should_save_state = False
    with STATE.lock:
        task = STATE.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        for result in task.get("results", []):
            if result.get("id") == result_id:
                result["deleted"] = True
                result["selected"] = False
                task["updated_at"] = now_ts()
                should_save_state = True
                break
    if should_save_state:
        STATE.save_state()
        return {"ok": True}
    raise HTTPException(status_code=404, detail="结果不存在")


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, Any]:
    removed_files = 0
    with STATE.lock:
        if STATE.running:
            raise HTTPException(status_code=409, detail="运行中任务不可删除")
        if STATE.active_single_retry_task_ids:
            raise HTTPException(status_code=409, detail="存在单任务重试运行中，暂不可删除任务")
        task = STATE.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        cache_paths: List[Path] = []
        for result in task.get("results", []):
            file_path = Path(result.get("file_path", ""))
            if file_path.exists() and file_path.is_file():
                cache_paths.append(file_path)

        STATE.tasks.pop(task_id, None)
        STATE.task_order = [tid for tid in STATE.task_order if tid != task_id]

    for file_path in cache_paths:
        try:
            file_path.unlink(missing_ok=True)
            removed_files += 1
        except Exception:
            continue

    STATE.save_state()
    cleanup_summary = await asyncio.to_thread(cleanup_orphan_cache_files, STATE, f"delete_task:{task_id}", True)
    return {"ok": True, "task_id": task_id, "removed_cache_files": removed_files, "cleanup": cleanup_summary}


@app.post("/api/tasks/{task_id}/results/{result_id}/save")
async def save_single_result(task_id: str, result_id: str, request: SaveSingleRequest) -> Dict[str, Any]:
    raw_output_dir = request.output_dir.strip()
    if not raw_output_dir:
        raise HTTPException(status_code=400, detail="输出目录不能为空")
    output_dir = Path(raw_output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    with STATE.lock:
        task = STATE.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        source_path = Path(task.get("source_path", ""))
        results = task.get("results", [])
        result_item: Dict[str, Any] = {}
        result_index = 0
        for index, result in enumerate(results, 1):
            if result.get("id") == result_id:
                result_item = dict(result)
                result_index = index
                break

    if not result_item:
        raise HTTPException(status_code=404, detail="结果不存在")
    if result_item.get("deleted", False):
        raise HTTPException(status_code=400, detail="该结果已被删除，无法保存")

    image_path = Path(result_item.get("file_path", ""))
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    try:
        file_stem = apply_filename_rule(
            template=request.filename_template,
            src_path=source_path,
            task_id=task_id,
            result_id=result_item.get("id", ""),
            result_index=result_index,
            global_index=1,
            replace_from=request.replace_from,
            replace_to=request.replace_to,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"文件名模板解析失败: {exc}") from exc

    safe_stem = safe_filename_component(file_stem, fallback=f"{source_path.stem}_{result_index}")
    target_path = (output_dir / f"{safe_stem}{image_path.suffix}").resolve()
    if output_dir not in target_path.parents and target_path != output_dir:
        raise HTTPException(status_code=400, detail="保存路径非法")
    target_path = unique_target_path(target_path)
    target_path.write_bytes(image_path.read_bytes())

    txt_path = ""
    if request.create_txt:
        try:
            txt_content = request.txt_template.format(
                src_stem=source_path.stem,
                src_name=source_path.name,
                src_ext=source_path.suffix.lstrip("."),
                task_id=task_id,
                result_id=result_item.get("id", ""),
                result_index=result_index,
                global_index=1,
                image_name=target_path.name,
                image_path=str(target_path),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"TXT 模板解析失败: {exc}") from exc
        txt_file = target_path.with_suffix(".txt")
        txt_file.write_text(txt_content, encoding="utf-8")
        txt_path = str(txt_file)

    return {
        "ok": True,
        "saved_image": str(target_path),
        "saved_txt": txt_path,
    }


@app.post("/api/tasks/save_all")
async def save_all(request: SaveAllRequest) -> Dict[str, Any]:
    raw_output_dir = request.output_dir.strip()
    if not raw_output_dir:
        raise HTTPException(status_code=400, detail="输出目录不能为空")
    output_dir = Path(raw_output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    saved_count = 0
    txt_count = 0
    errors = []

    with STATE.lock:
        task_ids = list(STATE.task_order)
        tasks_copy = {task_id: dict(STATE.tasks[task_id]) for task_id in task_ids if task_id in STATE.tasks}

    for task_id in task_ids:
        task = tasks_copy.get(task_id)
        if not task:
            continue
        source_path = Path(task.get("source_path", ""))
        results = task.get("results", [])
        eligible_results: List[Tuple[int, Dict[str, Any]]] = []
        for index, result in enumerate(results, 1):
            if result.get("deleted", False):
                continue
            if request.only_selected and not result.get("selected", True):
                continue
            eligible_results.append((index, result))

        if request.only_first_if_multiple and len(eligible_results) >= 1:
            eligible_results = eligible_results[:1]

        for index, result in eligible_results:

            image_path = Path(result.get("file_path", ""))
            if not image_path.exists():
                errors.append(f"{task.get('source_name','')} -> 结果文件不存在")
                continue

            try:
                filename_template = request.filename_template
                if request.only_first_if_multiple and len(eligible_results) >= 1:
                    filename_template = "{src_stem}"
                file_stem = apply_filename_rule(
                    template=filename_template,
                    src_path=source_path,
                    task_id=task_id,
                    result_id=result.get("id", ""),
                    result_index=index,
                    global_index=saved_count + 1,
                    replace_from=request.replace_from,
                    replace_to=request.replace_to,
                )
            except Exception as exc:
                errors.append(f"{source_path.name}: 文件名模板解析失败 ({exc})")
                continue

            safe_stem = safe_filename_component(file_stem, fallback=f"{source_path.stem}_{index}")
            target_path = (output_dir / f"{safe_stem}{image_path.suffix}").resolve()
            if output_dir not in target_path.parents and target_path != output_dir:
                errors.append(f"{source_path.name}: 保存路径非法")
                continue
            target_path = unique_target_path(target_path)
            target_path.write_bytes(image_path.read_bytes())
            saved_count += 1

            if request.create_txt:
                try:
                    txt_content = request.txt_template.format(
                        src_stem=source_path.stem,
                        src_name=source_path.name,
                        src_ext=source_path.suffix.lstrip("."),
                        task_id=task_id,
                        result_id=result.get("id", ""),
                        result_index=index,
                        global_index=saved_count,
                        image_name=target_path.name,
                        image_path=str(target_path),
                    )
                    txt_path = target_path.with_suffix(".txt")
                    txt_path.write_text(txt_content, encoding="utf-8")
                    txt_count += 1
                except Exception as exc:
                    errors.append(f"{target_path.name}: TXT 模板解析失败 ({exc})")

    return {
        "ok": True,
        "saved_count": saved_count,
        "txt_count": txt_count,
        "error_count": len(errors),
        "errors": errors[:20],
    }


@app.post("/api/tasks/clear")
async def clear_tasks() -> Dict[str, Any]:
    removed_files = 0
    with STATE.lock:
        if STATE.running:
            raise HTTPException(status_code=409, detail="运行中任务不可清空")
        if STATE.active_single_retry_task_ids:
            raise HTTPException(status_code=409, detail="存在单任务重试运行中，暂不可清空任务")

        cache_paths = []
        for task in STATE.tasks.values():
            for result in task.get("results", []):
                file_path = Path(result.get("file_path", ""))
                if file_path.exists() and file_path.is_file():
                    cache_paths.append(file_path)

        STATE.tasks = {}
        STATE.task_order = []

    for file_path in cache_paths:
        try:
            file_path.unlink(missing_ok=True)
            removed_files += 1
        except Exception:
            continue

    STATE.save_state()
    cleanup_summary = await asyncio.to_thread(cleanup_orphan_cache_files, STATE, "clear_tasks", True)
    return {"ok": True, "removed_cache_files": removed_files, "cleanup": cleanup_summary}


@app.get("/api/tasks/{task_id}/source")
async def task_source_image(task_id: str) -> FileResponse:
    task = get_task_or_404(task_id)
    source_path = Path(task.get("source_path", ""))
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="原图不存在")
    return FileResponse(path=str(source_path))


@app.get("/api/tasks/{task_id}/results/{result_id}/image")
async def task_result_image(task_id: str, result_id: str) -> FileResponse:
    result = get_result_or_404(task_id, result_id)
    image_path = Path(result.get("file_path", ""))
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    return FileResponse(path=str(image_path))


@app.post("/api/config/current")
async def save_current_config(request: SaveCurrentConfigRequest) -> Dict[str, Any]:
    config = request.config.sanitize()
    with STATE.lock:
        config = merge_api_key_if_empty(config, STATE.current_config)
        STATE.current_config = config
    STATE.save_state()
    return {"ok": True}


@app.post("/api/presets/save")
async def save_preset(request: SavePresetRequest) -> Dict[str, Any]:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="预设名称不能为空")
    config = request.config.sanitize()
    with STATE.lock:
        config = merge_api_key_if_empty(config, STATE.presets.get(name, STATE.current_config))
        STATE.presets[name] = config
    STATE.save_presets()
    return {"ok": True, "name": name}


@app.post("/api/presets/apply")
async def apply_preset(request: ApplyPresetRequest) -> Dict[str, Any]:
    name = request.name.strip()
    with STATE.lock:
        config = STATE.presets.get(name)
        if not config:
            raise HTTPException(status_code=404, detail="预设不存在")
        STATE.current_config = dict(config)
    STATE.save_state()
    return {"ok": True, "config": sanitize_config_for_client(config)}


@app.delete("/api/presets/{name}")
async def delete_preset(name: str) -> Dict[str, Any]:
    with STATE.lock:
        if name not in STATE.presets:
            raise HTTPException(status_code=404, detail="预设不存在")
        STATE.presets.pop(name, None)
    STATE.save_presets()
    return {"ok": True}


@app.get("/api/presets/export")
async def export_presets() -> StreamingResponse:
    include_secrets = os.getenv("EXPORT_INCLUDE_SECRETS", "0") == "1"
    with STATE.lock:
        data = {
            "current_config": STATE.current_config if include_secrets else sanitize_config_for_client(STATE.current_config),
            "presets": STATE.presets if include_secrets else {name: sanitize_config_for_client(cfg) for name, cfg in STATE.presets.items()},
        }
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"api_presets_{now_ts()}.json"
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/presets/import")
async def import_presets(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw = await file.read(MAX_PRESET_IMPORT_BYTES + 1)
    if len(raw) > MAX_PRESET_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail=f"导入文件过大，限制 {MAX_PRESET_IMPORT_BYTES} bytes")
    try:
        content = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"配置文件解析失败: {exc}") from exc

    imported = 0
    current_config = None
    presets = content.get("presets", {})
    if not isinstance(presets, dict):
        presets = {}

    with STATE.lock:
        for name, cfg in presets.items():
            if not isinstance(name, str) or not isinstance(cfg, dict):
                continue
            try:
                STATE.presets[name] = ApiConfig(**cfg).sanitize()
                imported += 1
            except Exception:
                continue

        cfg = content.get("current_config")
        if isinstance(cfg, dict):
            try:
                current_config = ApiConfig(**cfg).sanitize()
                STATE.current_config = current_config
            except Exception:
                current_config = None

    STATE.save_presets()
    STATE.save_state()
    return {
        "ok": True,
        "imported_presets": imported,
        "applied_current_config": bool(current_config),
    }


def run_app() -> int:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", DEFAULT_PORT))
    status_ui = (os.getenv("STATUS_UI", "") or "").strip().lower()
    if not status_ui:
        status_ui = "none" if os.getenv("STATUS_WINDOW", "1") == "0" else "cli"

    if not ensure_port_ready_or_exit(host, port):
        return 1

    if status_ui in {"cli", "terminal", "tui"}:
        return run_with_cli_dashboard(host, port)
    if status_ui in {"none", "off", "0"}:
        uvicorn.run(app, host=host, port=port)
        return 0

    print(f"未知 STATUS_UI={status_ui}，自动回退为 cli 界面。")
    return run_with_cli_dashboard(host, port)

if __name__ == "__main__":
    raise SystemExit(run_app())
