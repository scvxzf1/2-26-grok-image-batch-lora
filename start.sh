#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-12607}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://127.0.0.1:${PORT},http://localhost:${PORT}}"
EXPORT_INCLUDE_SECRETS="${EXPORT_INCLUDE_SECRETS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
APP_DATA_DIR="${APP_DATA_DIR:-${HOME}/.image-batch-console}"

export HOST PORT ALLOWED_ORIGINS EXPORT_INCLUDE_SECRETS APP_DATA_DIR

if [[ -n "${PORT_CONFLICT_PROMPT:-}" ]]; then
  echo "提示：PORT_CONFLICT_PROMPT 已废弃，端口冲突将仅输出诊断信息并退出。"
fi

if ! "${PYTHON_BIN}" app.py; then
  echo "启动失败，请根据上方端口冲突诊断信息处理后重试。"
  exit 1
fi
