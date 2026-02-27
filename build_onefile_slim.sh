#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
APP_NAME="${APP_NAME:-image-batch-console-slim}"

if ! "${PYTHON_BIN}" -c "import PyInstaller" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install pyinstaller
fi

"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --strip \
  --name "${APP_NAME}" \
  --add-data "templates:templates" \
  --add-data "static:static" \
  --exclude-module tkinter \
  --exclude-module _tkinter \
  --exclude-module PIL.ImageTk \
  app.py

echo "瘦身构建完成：dist/${APP_NAME}"
