#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
APP_NAME="${APP_NAME:-image-batch-console}"
PYINSTALLER_VERSION="${PYINSTALLER_VERSION:-6.19.0}"
BUILD_REQUIREMENTS_FILE="${BUILD_REQUIREMENTS_FILE:-requirements-build.txt}"

if ! "${PYTHON_BIN}" - "${PYINSTALLER_VERSION}" <<'PY'
import importlib
import sys

required = sys.argv[1]

try:
    mod = importlib.import_module("PyInstaller")
except Exception:
    raise SystemExit(1)

installed = getattr(mod, "__version__", "")
if installed != required:
    raise SystemExit(1)
PY
then
  echo "错误：未检测到 pyinstaller==${PYINSTALLER_VERSION}" >&2
  if [[ -f "${BUILD_REQUIREMENTS_FILE}" ]]; then
    echo "请先执行：${PYTHON_BIN} -m pip install -r ${BUILD_REQUIREMENTS_FILE}" >&2
  else
    echo "请先执行：${PYTHON_BIN} -m pip install \"pyinstaller==${PYINSTALLER_VERSION}\"" >&2
    echo "（可选）创建 ${BUILD_REQUIREMENTS_FILE} 并写入固定版本构建依赖。" >&2
  fi
  exit 1
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
