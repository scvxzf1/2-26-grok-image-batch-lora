import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = str(Path(__file__).resolve().parents[1])
ORIGINAL_APP_DATA_DIR = os.environ.get("APP_DATA_DIR")
APP_DATA_TMPDIR = tempfile.TemporaryDirectory(prefix="image-batch-test-data-")
os.environ["APP_DATA_DIR"] = APP_DATA_TMPDIR.name

repo_root_inserted = False
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    repo_root_inserted = True

import app

if repo_root_inserted:
    try:
        sys.path.remove(REPO_ROOT)
    except ValueError:
        pass

if ORIGINAL_APP_DATA_DIR is None:
    os.environ.pop("APP_DATA_DIR", None)
else:
    os.environ["APP_DATA_DIR"] = ORIGINAL_APP_DATA_DIR


def make_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (6, 6), (12, 34, 56)).save(buf, format="PNG")
    return buf.getvalue()


def test_two_phase_commit_success(tmp_path: Path) -> None:
    content = make_png_bytes()
    info = app.save_content_as_image(tmp_path, "task1", content, "data_url")
    staged = Path(info["_staged_file_path"])
    final = Path(info["file_path"])

    assert staged.exists()
    assert not final.exists()

    committed = app.commit_staged_image_files([info])
    assert len(committed) == 1
    assert final.exists()
    assert not staged.exists()
    assert "_staged_file_path" not in info


def test_two_phase_commit_failure_rollback(tmp_path: Path) -> None:
    content = make_png_bytes()
    info = app.save_content_as_image(tmp_path, "task2", content, "data_url")
    staged = Path(info["_staged_file_path"])
    info["file_path"] = str(tmp_path / "missing_dir" / "target.png")

    with pytest.raises(FileNotFoundError):
        app.commit_staged_image_files([info])

    assert not staged.exists()


def test_cleanup_orphan_cache_files_remove_unreferenced() -> None:
    with tempfile.TemporaryDirectory(prefix="image-batch-cleanup-") as td:
        data_dir = Path(td)
        cache_dir = data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ref_file = cache_dir / "ref.png"
        orphan_file = cache_dir / "orphan.png"
        ref_file.write_bytes(b"ref")
        orphan_file.write_bytes(b"orphan")

        state_payload = {
            "tasks": {
                "task_a": {
                    "results": [{"file_path": str(ref_file)}],
                    "status": "done",
                }
            },
            "task_order": ["task_a"],
            "current_config": {},
        }
        (data_dir / "state.json").write_text(json.dumps(state_payload), encoding="utf-8")
        (data_dir / "presets.json").write_text("{}", encoding="utf-8")

        state = app.AppState(data_dir)
        summary = app.cleanup_orphan_cache_files(state, reason="test", remove_all_unreferenced=True)

        assert summary["removed"] >= 1
        assert ref_file.exists()
        assert not orphan_file.exists()


def test_download_data_url_and_cancel() -> None:
    content = app.download_candidate_content(
        {"type": "url", "value": "data:image/png;base64,QUJDRA==", "key": "x"},
        timeout_seconds=3,
        max_retries=0,
        retry_interval=0,
    )
    assert content == b"ABCD"

    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(ValueError, match="任务被紧急停止"):
        app.download_candidate_content(
            {"type": "url", "value": "data:image/png;base64,QUJDRA==", "key": "x"},
            timeout_seconds=3,
            max_retries=0,
            retry_interval=0,
            cancel_event=cancel_event,
        )


def test_validate_network_url_rules() -> None:
    with pytest.raises(ValueError):
        app.validate_network_url("ftp://example.com/file.png", "下载URL", False, set())

    with pytest.raises(ValueError):
        app.validate_network_url("http://127.0.0.1/file.png", "下载URL", False, set())

    app.validate_network_url("https://example.com/file.png", "下载URL", True, {"example.com"})

    with pytest.raises(ValueError):
        app.validate_network_url("https://bad.example.com/file.png", "下载URL", True, {"example.org"})


def test_is_private_api_url_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "ALLOW_PRIVATE_API_URL", False)
    assert app.is_private_api_url_unlocked({}) is False
    assert app.is_private_api_url_unlocked({"unlock_local_api_url": True}) is True
    assert app.is_private_api_url_unlocked({"unlock_local_api_url": "true"}) is True
    assert app.is_private_api_url_unlocked({"unlock_local_api_url": "0"}) is False

    monkeypatch.setattr(app, "ALLOW_PRIVATE_API_URL", True)
    assert app.is_private_api_url_unlocked({"unlock_local_api_url": False}) is True


def test_is_private_download_url_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "ALLOW_PRIVATE_DOWNLOAD_URL", False)
    assert app.is_private_download_url_unlocked({}) is False
    assert app.is_private_download_url_unlocked({"unlock_local_download_url": True}) is True
    assert app.is_private_download_url_unlocked({"unlock_local_download_url": "true"}) is True
    assert app.is_private_download_url_unlocked({"unlock_local_download_url": "0"}) is False

    monkeypatch.setattr(app, "ALLOW_PRIVATE_DOWNLOAD_URL", True)
    assert app.is_private_download_url_unlocked({"unlock_local_download_url": False}) is True


def test_api_config_sanitize_contains_unlock_local_switches() -> None:
    sanitized = app.ApiConfig(unlock_local_api_url=True, unlock_local_download_url=True).sanitize()
    assert sanitized["unlock_local_api_url"] is True
    assert sanitized["unlock_local_download_url"] is True


def test_call_api_short_circuit_on_cancel() -> None:
    cancel_event = threading.Event()
    cancel_event.set()
    with tempfile.TemporaryDirectory(prefix="image-batch-cache-") as cache_dir:
        result = app.call_api_and_extract_images(
            task={"id": "t1", "source_path": "/tmp/does-not-exist.png"},
            config={
                "api_url": "https://example.com/v1/chat/completions",
                "api_key": "",
                "api_mode": "auto",
                "token_field": "auto",
                "model_name": "x",
                "temperature": 0.7,
                "system_prompt": "s",
                "user_prompt": "u",
                "max_tokens": 10,
                "timeout_seconds": 1,
                "max_retries": 0,
                "retry_interval": 0,
                "concurrency_limit": 1,
            },
            cache_dir=Path(cache_dir),
            cancel_event=cancel_event,
        )
    assert result["ok"] is False
    assert "任务被紧急停止" in result["error"]
