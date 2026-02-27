#!/usr/bin/env bash
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
  echo "请先安装 python3"
  read -p "按回车退出..."
  exit 1
fi

[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export PYTHON_BIN="python"

# 先在后台注册：3秒后自动打开浏览器
(sleep 3 && echo "正在打开浏览器..." && open "http://127.0.0.1:12607") &

./start.sh || {
  echo -e "\n提示: 若要在开发时修改代码后自动重启，请在虚拟环境中执行: uvicorn app:app --reload --port 12607"
  read -p "按回车退出..."
}
