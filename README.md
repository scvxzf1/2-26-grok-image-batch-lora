# 图片批处理调度台
<img width="1408" height="1399" alt="image" src="https://github.com/user-attachments/assets/82390f03-f236-43d7-a904-93cd596b97d1" />

## 项目说明

- 本项目专门面向 `grok-image-edit` 模型设计，用于我个人生成编辑模型数据集的批处理流程。
- 本仓库主要用于自用留档（“水个库”），不提供、也不讨论 `grok-image-edit` API 获取渠道。
- 本项目代码与文档主要由 AI 辅助编写与整理。

## 功能概览

- 输入路径（目录或单图）后，批量读取图片并并发调用 API。
- 从 API 返回结果中自动提取全部图片（URL、data URL、base64）。
- 支持原图 + 返回图预览，手动勾选、删除结果。
- 支持在任务列表中对单个结果执行“单次保存”（复用 3) 手动保存规则）。
- 支持单任务重试，并保留此前已成功的结果图。
- 支持失败任务一键批量重试（保留历史结果），并展示总进度条。
- 支持紧急停止按钮，立即停止当前批处理并将未完成任务标记为“已停止”。
- 支持批量保存结果，支持文件名模板、`start -> end` 自定义替换、可选生成 `.txt`。
- 支持“多图仅保存首张”开关：开启后每个任务仅保存首张，并按源名保存（如 `001_end`，单张任务同样生效）。
- 支持 API 配置持久化、多预设保存、配置导入导出。
- 支持 SSE 流式返回解析（兼容 `data:` 事件流与“多段 JSON + [DONE]”格式），自动从流式内容中提取图片链接。
- 支持任务级“调试日志”（默认折叠），展开可查看对应 SSE 数据流与解析信息。

## 启动方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动服务（默认端口 `12607`）：

```bash
python app.py
```

启动时行为：

- 默认会打开“CLI 状态界面”（终端面板），实时显示后端日志与运行状态。
- 若端口被占用，不会自动关闭任何进程；会输出绑定错误、占用 PID 探测结果与处理建议。

3. 打开浏览器：

```text
http://127.0.0.1:12607
```

## 脚本启动

仓库内提供一键启动脚本 `./start.sh`（默认使用 `python app.py` 启动）。

1. 赋予执行权限（首次执行时）：

```bash
chmod +x ./start.sh
```

2. 默认启动：

```bash
./start.sh
```

3. 通过环境变量覆盖默认值（示例）：

```bash
HOST=0.0.0.0 PORT=18080 ./start.sh
```

可用环境变量：

- `HOST`：监听地址，默认 `127.0.0.1`
- `PORT`：监听端口，默认 `12607`
- `ALLOWED_ORIGINS`：CORS 白名单，逗号分隔
- `EXPORT_INCLUDE_SECRETS`：导出配置是否包含密钥（`1`/`0`）
- `PYTHON_BIN`：Python 命令，默认 `python`
- `APP_DATA_DIR`：运行数据目录（`state.json` / `presets.json` / `cache`），`start.sh` 默认 `~/.image-batch-console`
- `STATUS_UI`：状态界面模式（`cli`/`none`，默认 `cli`）
- `STATUS_WINDOW`：旧开关兼容（`0` 等价于 `STATUS_UI=none`）

## GitHub 发布前（免手动清理）

仓库已内置以下防护：

- `.gitignore` 忽略 `web_data/state.json`、`web_data/presets.json`、`web_data/cache/`
- 运行数据默认写入 `APP_DATA_DIR=~/.image-batch-console`（不落到仓库）
- 提供安全模板：`web_data/state.example.json`、`web_data/presets.example.json`
- 提交前密钥扫描：`.pre-commit-config.yaml` + `.gitleaks.toml`

推荐初始化步骤：

```bash
mkdir -p ~/.image-batch-console
cp -n web_data/state.example.json ~/.image-batch-console/state.json
cp -n web_data/presets.example.json ~/.image-batch-console/presets.json
```

启用提交前扫描：

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

若这两个文件此前已被 Git 跟踪，可执行：

```bash
git rm --cached web_data/state.json web_data/presets.json
```

若你历史上已经提交过敏感文件（即使后来删除），建议立即轮换密钥，并清理历史记录后再公开仓库。

## Linux 单文件打包

仓库内提供打包脚本 `./build_onefile.sh`，可生成单个 Linux 可执行文件。

1. 执行打包：

```bash
./build_onefile.sh
```

瘦身版打包（推荐发布）：

```bash
./build_onefile_slim.sh
```

2. 打包产物：

```text
dist/image-batch-console
dist/image-batch-console-slim
```

3. 直接运行产物：

```bash
./dist/image-batch-console
./dist/image-batch-console-slim
```

说明：

- `image-batch-console-slim` 会排除 `tkinter/tcl/tk` 相关内容，体积更小。
- 瘦身版默认使用命令行交互（端口冲突会输出详细诊断并退出，不做端口清理）。

可选环境变量：

- `APP_NAME`：产物文件名，默认 `image-batch-console`
- `PYTHON_BIN`：构建使用的 Python 命令，默认 `python`
- `HOST` / `PORT` / `ALLOWED_ORIGINS` / `EXPORT_INCLUDE_SECRETS` / `APP_DATA_DIR` / `STATUS_UI`：运行时环境变量，与源码启动一致

## 安全与网络说明

- 默认仅监听本机：`HOST=127.0.0.1`（可通过环境变量覆盖）。
- 默认 CORS 仅允许：
  - `http://127.0.0.1:12607`
  - `http://localhost:12607`
- 可通过 `ALLOWED_ORIGINS` 自定义（逗号分隔）。
- 导出配置默认包含密钥；若不希望包含，启动前设置：`EXPORT_INCLUDE_SECRETS=0`。

## 文件名模板变量

可在保存规则中使用以下变量：

- `{src_stem}`：原图文件名（无后缀）
- `{src_name}`：原图文件名（含后缀）
- `{src_ext}`：原图后缀（无点）
- `{task_id}`：任务 ID
- `{result_id}`：结果 ID
- `{result_index}`：当前任务内第几个结果（从 1 开始）
- `{global_index}`：全局保存序号（从 1 开始）
- `{image_name}`：保存后的图片文件名（仅 TXT 模板可用）
- `{image_path}`：保存后的图片路径（仅 TXT 模板可用）

示例：

- 文件名模板：`{src_stem}_{result_index}`
- 替换规则：`start -> end`
- TXT 模板：`{src_name}`
