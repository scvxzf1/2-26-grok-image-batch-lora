#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
APP_NAME="${APP_NAME:-image-batch-console}"

if ! "${PYTHON_BIN}" -c "import PyInstaller" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install pyinstaller
fi

"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name "${APP_NAME}" \
  --add-data "templates:templates" \
  --add-data "static:static" \
  app.py

echo "构建完成：dist/${APP_NAME}"
