const state = {
  latest: null,
  timer: null,
  stateLoadSeq: 0,
  stateLoadingCount: 0,
  tasksFingerprint: null,
  debugOpenTaskIds: new Set(),
  debugLogCache: {},
  debugLoadingTaskIds: new Set(),
};

const SAVE_RULE_PRESETS = {
  aitookit_edit: {
    label: "AITookit编辑（start→end）",
    hint: "示例：输入 001_start 输出 001_end（不生成同名txt）。",
    values: {
      filename_template: "{src_stem}",
      replace_from: "start",
      replace_to: "end",
      create_txt: false,
      txt_template: "{src_name}",
      only_selected: true,
      only_first_if_multiple: false,
    },
  },
  keep_source_name: {
    label: "源名直出（保留后缀）",
    hint: "输出文件名保持源名（不做替换），适合快速整理。",
    values: {
      filename_template: "{src_stem}",
      replace_from: "",
      replace_to: "",
      create_txt: false,
      txt_template: "{src_name}",
      only_selected: true,
      only_first_if_multiple: false,
    },
  },
  source_with_txt: {
    label: "源名 + 元信息TXT",
    hint: "输出源名，同时生成 TXT（包含源名与任务标识）。",
    values: {
      filename_template: "{src_stem}",
      replace_from: "",
      replace_to: "",
      create_txt: true,
      txt_template: "{src_name}\\naitookit编辑\\n任务:{task_id}",
      only_selected: true,
      only_first_if_multiple: false,
    },
  },
  first_only_clean: {
    label: "仅首图（简洁命名）",
    hint: "多图任务仅保存首图，命名使用源名，适合出图挑选。",
    values: {
      filename_template: "{src_stem}",
      replace_from: "",
      replace_to: "",
      create_txt: false,
      txt_template: "{src_name}",
      only_selected: true,
      only_first_if_multiple: true,
    },
  },
  indexed_archive: {
    label: "归档编号（带序号）",
    hint: "按 {src_stem}_{result_index} 输出，适合保留多结果。",
    values: {
      filename_template: "{src_stem}_{result_index}",
      replace_from: "",
      replace_to: "",
      create_txt: false,
      txt_template: "{src_name}",
      only_selected: true,
      only_first_if_multiple: false,
    },
  },
};

function $(id) {
  return document.getElementById(id);
}

function toast(text, timeout = 2200) {
  const el = $("toast");
  el.textContent = text;
  el.style.display = "block";
  clearTimeout(el._timer);
  el._timer = setTimeout(() => {
    el.style.display = "none";
  }, timeout);
}

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const isJson = res.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await res.json() : {};
  if (!res.ok) {
    throw new Error(payload.detail || payload.message || `请求失败(${res.status})`);
  }
  return payload;
}

async function runWithButtonLock(button, loadingText, task) {
  if (!button) {
    return task();
  }
  if (button.disabled) {
    return null;
  }
  const originalText = button.dataset.originalText || button.textContent;
  button.dataset.originalText = originalText;
  button.disabled = true;
  button.textContent = loadingText;
  try {
    return await task();
  } finally {
    button.disabled = false;
    button.textContent = button.dataset.originalText || originalText;
  }
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const input = document.createElement("textarea");
  input.value = text;
  input.style.position = "fixed";
  input.style.opacity = "0";
  document.body.appendChild(input);
  input.focus();
  input.select();
  document.execCommand("copy");
  document.body.removeChild(input);
}

async function ensureDebugLog(taskId, preElement = null) {
  const safeTaskId = sanitizeTaskId(taskId);
  if (!safeTaskId) {
    return null;
  }
  if (state.debugLogCache[safeTaskId]) {
    return state.debugLogCache[safeTaskId];
  }
  if (state.debugLoadingTaskIds.has(safeTaskId)) {
    return null;
  }
  if (preElement) {
    preElement.textContent = "日志加载中...";
  }
  state.debugLoadingTaskIds.add(safeTaskId);
  try {
    const payload = await api(`/api/tasks/${encodeURIComponent(safeTaskId)}/debug`);
    const logText = payload.has_debug_log ? payload.debug_log : "暂无调试日志（任务可能尚未执行完成）。";
    state.debugLogCache[safeTaskId] = logText;
    return logText;
  } finally {
    state.debugLoadingTaskIds.delete(safeTaskId);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const SAFE_ENTITY_ID_RE = /^[A-Za-z0-9_-]{1,128}$/;
const DANGEROUS_ID_VALUES = new Set(["__proto__", "constructor", "prototype"]);

function sanitizeEntityId(value) {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return null;
  }
  if (!SAFE_ENTITY_ID_RE.test(normalized)) {
    return null;
  }
  if (DANGEROUS_ID_VALUES.has(normalized)) {
    return null;
  }
  return normalized;
}

function sanitizeTaskId(value) {
  return sanitizeEntityId(value);
}

function sanitizeResultId(value) {
  return sanitizeEntityId(value);
}

function parseTaskResultPair(value) {
  const parts = String(value ?? "").split("::");
  if (parts.length !== 2) {
    return null;
  }
  const taskId = sanitizeTaskId(parts[0]);
  const resultId = sanitizeResultId(parts[1]);
  if (!taskId || !resultId) {
    return null;
  }
  return { taskId, resultId };
}

function sanitizeImageUrl(value) {
  const raw = String(value ?? "").trim();
  if (!raw) {
    return "";
  }
  try {
    const parsed = new URL(raw, window.location.origin);
    const protocol = parsed.protocol.toLowerCase();
    if (protocol === "http:" || protocol === "https:" || protocol === "blob:") {
      return parsed.href;
    }
    const allowDataImage = /^data:image\/(?:png|jpeg|jpg|webp|gif|bmp);base64,[a-z0-9+/=\s]+$/i.test(raw);
    if (protocol === "data:" && allowDataImage) {
      return raw;
    }
    return "";
  } catch (error) {
    return "";
  }
}

function renderSafePreviewImage(imageUrl, altText) {
  const safeUrl = sanitizeImageUrl(imageUrl);
  if (!safeUrl) {
    return `<div class="task-meta">图片地址无效</div>`;
  }
  return `<img src="${escapeHtml(safeUrl)}" alt="${escapeHtml(altText)}" loading="lazy" />`;
}

function safeInlineText(value, fallback = "-") {
  const text = String(value ?? "").trim();
  return escapeHtml(text || fallback);
}

function safeStatusClass(status) {
  const allow = new Set(["pending", "processing", "done", "partial_success", "failed", "stopped"]);
  return allow.has(status) ? status : "pending";
}

function isConfigEditing() {
  const active = document.activeElement;
  if (!active) {
    return false;
  }
  const configCard = document.querySelector(".card");
  if (!configCard) {
    return false;
  }
  return configCard.contains(active) && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName);
}

function collectConfigFromUI() {
  return {
    system_prompt: $("system_prompt").value,
    user_prompt: $("user_prompt").value,
    api_key: $("api_key").value,
    api_url: $("api_url").value,
    max_tokens: Number($("max_tokens").value || 4096),
    temperature: Number($("temperature").value || 0.7),
    model_name: $("model_name").value,
    concurrency_limit: Number($("concurrency_limit").value || 3),
    max_retries: Number($("max_retries").value || 2),
    retry_interval: Number($("retry_interval").value || 1.5),
    timeout_seconds: Number($("timeout_seconds").value || 120),
    api_mode: $("api_mode").value,
    token_field: $("token_field").value,
    unlock_local_api_url: $("unlock_local_api_url").value === "true",
    unlock_local_download_url: $("unlock_local_download_url").value === "true",
  };
}

function insertTextAtCursor(inputEl, text) {
  if (!inputEl) {
    return;
  }
  inputEl.focus();
  const start = inputEl.selectionStart ?? inputEl.value.length;
  const end = inputEl.selectionEnd ?? inputEl.value.length;
  const before = inputEl.value.slice(0, start);
  const after = inputEl.value.slice(end);
  inputEl.value = `${before}${text}${after}`;
  const nextPos = start + text.length;
  inputEl.setSelectionRange(nextPos, nextPos);
}

function fillConfigToUI(config) {
  $("system_prompt").value = config.system_prompt ?? "";
  $("user_prompt").value = config.user_prompt ?? "";
  $("api_key").value = config.api_key ?? "";
  $("api_url").value = config.api_url ?? "";
  $("max_tokens").value = config.max_tokens ?? 4096;
  $("temperature").value = config.temperature ?? 0.7;
  $("model_name").value = config.model_name ?? "gpt-4o";
  $("concurrency_limit").value = config.concurrency_limit ?? 3;
  $("max_retries").value = config.max_retries ?? 2;
  $("retry_interval").value = config.retry_interval ?? 1.5;
  $("timeout_seconds").value = config.timeout_seconds ?? 120;
  $("api_mode").value = config.api_mode ?? "auto";
  $("token_field").value = config.token_field ?? "auto";
  $("unlock_local_api_url").value = config.unlock_local_api_url ? "true" : "false";
  $("unlock_local_download_url").value = config.unlock_local_download_url ? "true" : "false";
}

function fillSaveRulePresetSelect() {
  const select = $("save_rule_preset");
  if (!select) {
    return;
  }
  select.innerHTML = "";
  const defaultOpt = document.createElement("option");
  defaultOpt.value = "";
  defaultOpt.textContent = "选择常用规则预设";
  select.appendChild(defaultOpt);
  Object.entries(SAVE_RULE_PRESETS).forEach(([presetId, preset]) => {
    const option = document.createElement("option");
    option.value = presetId;
    option.textContent = preset.label;
    select.appendChild(option);
  });
}

function updateSaveRulePresetHint() {
  const hintEl = $("save_rule_preset_hint");
  const select = $("save_rule_preset");
  if (!hintEl || !select) {
    return;
  }
  const preset = SAVE_RULE_PRESETS[select.value];
  if (!preset) {
    hintEl.textContent = "选择预设后可一键填充保存规则参数。";
    return;
  }
  hintEl.textContent = preset.hint;
}

function applySaveRulePreset(presetId, options = {}) {
  const showToast = options.showToast !== false;
  const preset = SAVE_RULE_PRESETS[presetId];
  if (!preset) {
    toast("预设不存在");
    return false;
  }
  const values = preset.values || {};
  $("filename_template").value = values.filename_template ?? "{src_stem}_{result_index}";
  $("replace_from").value = values.replace_from ?? "";
  $("replace_to").value = values.replace_to ?? "";
  $("create_txt").value = values.create_txt ? "true" : "false";
  $("txt_template").value = values.txt_template ?? "{src_name}";
  $("only_selected").value = values.only_selected === false ? "false" : "true";
  $("only_first_if_multiple").value = values.only_first_if_multiple ? "true" : "false";

  const select = $("save_rule_preset");
  if (select) {
    select.value = presetId;
  }
  updateSaveRulePresetHint();
  if (showToast) {
    toast(`已应用规则预设：${preset.label}`);
  }
  return true;
}

function renderPresets(presets) {
  const select = $("preset_select");
  const oldValue = select.value;
  select.innerHTML = "";
  const defaultOpt = document.createElement("option");
  defaultOpt.value = "";
  defaultOpt.textContent = "选择预设";
  select.appendChild(defaultOpt);

  Object.keys(presets).sort().forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });

  if ([...select.options].some((x) => x.value === oldValue)) {
    select.value = oldValue;
  }
}

function statusLabel(status) {
  const map = {
    pending: "等待中",
    processing: "处理中",
    done: "完成",
    partial_success: "部分成功",
    failed: "失败",
    stopped: "已停止",
  };
  return map[status] || status;
}

function captureDebugOpenStateFromDOM() {
  const opened = document.querySelectorAll("#task_list details.debug-details[open]");
  opened.forEach((node) => {
    const taskId = sanitizeTaskId(node.dataset.debugTaskId);
    if (taskId) {
      state.debugOpenTaskIds.add(taskId);
    }
  });
}

function cleanupDebugState(tasks) {
  const validTaskIds = new Set((tasks || []).map((task) => sanitizeTaskId(task.id)).filter(Boolean));
  for (const taskId of [...state.debugOpenTaskIds]) {
    if (!validTaskIds.has(taskId)) {
      state.debugOpenTaskIds.delete(taskId);
    }
  }
  for (const taskId of Object.keys(state.debugLogCache)) {
    if (!validTaskIds.has(taskId)) {
      delete state.debugLogCache[taskId];
    }
  }
  for (const taskId of [...state.debugLoadingTaskIds]) {
    if (!validTaskIds.has(taskId)) {
      state.debugLoadingTaskIds.delete(taskId);
    }
  }
}

function buildTasksFingerprint(tasks) {
  const normalizedTasks = Array.isArray(tasks) ? tasks : [];
  return JSON.stringify(normalizedTasks.map((task) => ({
    id: task?.id ?? "",
    status: task?.status ?? "",
    source_name: task?.source_name ?? "",
    source_path: task?.source_path ?? "",
    source_image_url: task?.source_image_url ?? "",
    last_error: task?.last_error ?? "",
    last_warning: task?.last_warning ?? "",
    attempts: task?.attempts ?? "",
    http_status: task?.http_status ?? "",
    results: (Array.isArray(task?.results) ? task.results : []).map((result) => ({
      id: result?.id ?? "",
      deleted: Boolean(result?.deleted),
      selected: Boolean(result?.selected),
      source_type: result?.source_type ?? "",
      image_url: result?.image_url ?? "",
      width: result?.width ?? "",
      height: result?.height ?? "",
    })),
  })));
}

function renderTasks(tasks) {
  const list = $("task_list");
  cleanupDebugState(tasks);
  list.innerHTML = "";
  $("task_stats").textContent = `任务：${tasks.length}`;

  tasks.forEach((task) => {
    const safeTaskId = sanitizeTaskId(task.id);
    if (!safeTaskId) {
      console.warn("[XSS-BLOCK] 跳过非法 task id:", task.id);
      return;
    }
    const taskItem = document.createElement("div");
    taskItem.className = "task-item";
    const sourceImageHtml = renderSafePreviewImage(task.source_image_url, "原图预览");

    const resultHtml = (Array.isArray(task.results) ? task.results : [])
      .filter((x) => !x.deleted)
      .map((result) => {
        const safeResultId = sanitizeResultId(result.id);
        if (!safeResultId) {
          console.warn("[XSS-BLOCK] 跳过非法 result id:", result.id);
          return `<div class="task-meta">结果 ID 无效，已拦截</div>`;
        }
        const checked = result.selected ? "checked" : "";
        const pairId = `${safeTaskId}::${safeResultId}`;
        const resultSourceType = escapeHtml(result.source_type || "");
        const resultImageHtml = renderSafePreviewImage(result.image_url, "返回图预览");
        const widthText = safeInlineText(result.width);
        const heightText = safeInlineText(result.height);
        return `
          <div class="result-row">
            <div class="pair-preview">
              <div>
                <div class="img-label">原图</div>
                ${sourceImageHtml}
              </div>
              <div>
                <div class="img-label">返回图</div>
                ${resultImageHtml}
              </div>
              <div class="row-actions">
                <label><input type="checkbox" data-toggle-id="${escapeHtml(pairId)}" ${checked} /> 选中保存</label>
                <button data-save-one-id="${escapeHtml(pairId)}" class="primary">单次保存</button>
                <button data-delete-id="${escapeHtml(pairId)}" class="danger">删除此结果</button>
                <span class="task-meta">${widthText}x${heightText} | ${resultSourceType}</span>
              </div>
            </div>
          </div>
        `;
      })
      .join("");

    const safeSourceName = escapeHtml(task.source_name || "");
    const safeSourcePath = escapeHtml(task.source_path || "");
    const safeLastError = escapeHtml(task.last_error || "");
    const safeLastWarning = escapeHtml(task.last_warning || "");
    const statusClass = safeStatusClass(task.status);
    const statusText = escapeHtml(statusLabel(task.status));
    const safeAttempts = safeInlineText(task.attempts, "0");
    const safeHttpStatus = safeInlineText(task.http_status, "-");
    const isDebugOpen = state.debugOpenTaskIds.has(safeTaskId);
    const isDebugLoading = state.debugLoadingTaskIds.has(safeTaskId);
    const cachedDebugLog = state.debugLogCache[safeTaskId] || "";
    const debugText = cachedDebugLog || (isDebugLoading ? "日志加载中..." : "点击展开后加载日志...");
    const debugLoaded = cachedDebugLog ? "1" : "0";

    taskItem.innerHTML = `
      <div class="task-head">
        <div>
          <strong>${safeSourceName}</strong>
          <div class="task-meta">${safeSourcePath}</div>
          <div class="task-meta">尝试次数: ${safeAttempts} | HTTP: ${safeHttpStatus}</div>
          ${safeLastError ? `<div class="task-error">${safeLastError}</div>` : ""}
          ${safeLastWarning ? `<div><span class="warning-tag" title="${safeLastWarning}">⚠ ${safeLastWarning}</span></div>` : ""}
        </div>
        <div class="task-head-actions">
          <span class="status ${statusClass}">${statusText}</span>
          <button data-retry-clear-id="${escapeHtml(safeTaskId)}">重试（清除已有结果）</button>
          <button data-retry-id="${escapeHtml(safeTaskId)}">重试（保留已有结果）</button>
          <button data-delete-task-id="${escapeHtml(safeTaskId)}" class="danger">删除任务</button>
        </div>
      </div>
      <div class="result-list">${resultHtml || `<div class="task-meta">暂无返回图片</div>`}</div>
      <details class="debug-details" data-debug-task-id="${escapeHtml(safeTaskId)}" data-loaded="${debugLoaded}" ${isDebugOpen ? "open" : ""}>
        <summary>调试日志（默认折叠，点击展开）</summary>
        <div class="debug-tools">
          <button class="copy-log-btn" data-copy-debug-id="${escapeHtml(safeTaskId)}">复制日志</button>
        </div>
        <pre class="debug-log"></pre>
      </details>
    `;
    const pre = taskItem.querySelector(".debug-log");
    if (pre) {
      pre.textContent = debugText;
    }
    list.appendChild(taskItem);
  });
}

function renderProgress(progress) {
  const total = Number(progress?.total || 0);
  const finished = Number(progress?.finished || 0);
  const percent = Number(progress?.percent || 0);
  const pending = Number(progress?.pending || 0);
  const processing = Number(progress?.processing || 0);
  const done = Number(progress?.done || 0);
  const partialSuccess = Number(progress?.partial_success || 0);
  const failed = Number(progress?.failed || 0);
  const stopped = Number(progress?.stopped || 0);

  $("progress_text").textContent = `总进度：${finished}/${total}（${percent.toFixed(1)}%） | 处理中：${processing} | 失败：${failed} | 已停止：${stopped}`;
  $("progress_bar").style.width = `${Math.max(0, Math.min(100, percent))}%`;
  if ($("progress_pending")) $("progress_pending").textContent = `等待中：${pending}`;
  if ($("progress_processing")) $("progress_processing").textContent = `处理中：${processing}`;
  if ($("progress_done")) $("progress_done").textContent = `完成：${done}`;
  if ($("progress_partial_success")) $("progress_partial_success").textContent = `部分成功：${partialSuccess}`;
  if ($("progress_failed")) $("progress_failed").textContent = `失败：${failed}`;
  if ($("progress_stopped")) $("progress_stopped").textContent = `已停止：${stopped}`;
}

async function loadState(options = {}) {
  const syncConfig = Boolean(options.syncConfig);
  const skipIfLoading = Boolean(options.skipIfLoading);
  if (skipIfLoading && state.stateLoadingCount > 0) {
    return;
  }
  const requestSeq = ++state.stateLoadSeq;
  state.stateLoadingCount += 1;
  try {
    captureDebugOpenStateFromDOM();
    const payload = await api("/api/state");
    if (requestSeq !== state.stateLoadSeq) {
      return;
    }
    state.latest = payload;
    if (syncConfig && !isConfigEditing()) {
      fillConfigToUI(payload.current_config || {});
    }
    renderPresets(payload.presets || {});
    renderProgress(payload.progress || {});
    const tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    const tasksFingerprint = buildTasksFingerprint(tasks);
    if (tasksFingerprint !== state.tasksFingerprint) {
      renderTasks(tasks);
      state.tasksFingerprint = tasksFingerprint;
    }
    const runStatusEl = $("run_status");
    if (payload.running) {
      runStatusEl.textContent = "状态：批处理运行中";
      runStatusEl.dataset.state = "running";
    } else if (Number(payload.single_retry_running || 0) > 0) {
      runStatusEl.textContent = `状态：单任务重试中（${Number(payload.single_retry_running || 0)}）`;
      runStatusEl.dataset.state = "retry-running";
    } else if (payload.stop_requested) {
      runStatusEl.textContent = "状态：已紧急停止";
      runStatusEl.dataset.state = "stopped";
    } else {
      runStatusEl.textContent = "状态：待机";
      runStatusEl.dataset.state = "idle";
    }
  } catch (err) {
    if (requestSeq === state.stateLoadSeq) {
      toast(`刷新失败: ${err.message}`);
    }
  } finally {
    state.stateLoadingCount = Math.max(0, state.stateLoadingCount - 1);
  }
}

function bindEvents() {
  fillSaveRulePresetSelect();
  updateSaveRulePresetHint();
  $("btn_refresh").addEventListener("click", loadState);
  $("txt_var_guide")?.addEventListener("click", (event) => {
    const btn = event.target?.closest?.("button[data-insert-value]");
    if (!btn) {
      return;
    }
    const value = btn.dataset.insertValue || "";
    insertTextAtCursor($("txt_template"), value);
  });
  $("save_rule_preset")?.addEventListener("change", updateSaveRulePresetHint);
  $("btn_apply_save_rule_preset")?.addEventListener("click", () => {
    const presetId = $("save_rule_preset")?.value || "";
    if (!presetId) {
      toast("请先选择规则预设");
      return;
    }
    applySaveRulePreset(presetId);
  });

  $("btn_start").addEventListener("click", async (event) => {
    await runWithButtonLock(event.currentTarget, "启动中...", async () => {
      try {
        const inputPath = $("input_path").value.trim();
        if (!inputPath) {
          toast("请先输入路径");
          return;
        }
        await api("/api/tasks/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_path: inputPath,
            config: collectConfigFromUI(),
          }),
        });
        toast("批处理已启动");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    });
  });

  $("btn_retry_failed").addEventListener("click", async (event) => {
    await runWithButtonLock(event.currentTarget, "重试中...", async () => {
      try {
        const payload = await api("/api/tasks/retry_failed", { method: "POST" });
        if (payload.retried_count > 0) {
          toast(`已重试失败任务：${payload.retried_count} 个`);
        } else {
          toast("当前没有可重试的失败任务");
        }
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    });
  });

  $("btn_stop").addEventListener("click", async (event) => {
    await runWithButtonLock(event.currentTarget, "停止中...", async () => {
      try {
        const payload = await api("/api/tasks/stop", { method: "POST" });
        if (payload.stopped) {
          toast(`已紧急停止，影响任务 ${payload.affected} 个`);
        } else {
          toast(payload.message || "当前没有运行中的任务");
        }
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    });
  });

  $("btn_clear").addEventListener("click", async (event) => {
    await runWithButtonLock(event.currentTarget, "清空中...", async () => {
      try {
        if (!window.confirm("确认清空全部任务吗？该操作不可撤销。")) {
          return;
        }
        await api("/api/tasks/clear", { method: "POST" });
        toast("已清空任务");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    });
  });

  $("btn_save_config").addEventListener("click", async () => {
    try {
      await api("/api/config/current", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: collectConfigFromUI() }),
      });
      toast("当前配置已保存");
      await loadState({ syncConfig: true });
    } catch (err) {
      toast(err.message);
    }
  });

  $("btn_save_preset").addEventListener("click", async () => {
    try {
      const name = $("preset_name").value.trim();
      if (!name) {
        toast("请输入预设名称");
        return;
      }
      await api("/api/presets/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          config: collectConfigFromUI(),
        }),
      });
      toast("预设已保存");
      $("preset_name").value = "";
      await loadState();
    } catch (err) {
      toast(err.message);
    }
  });

  $("btn_apply_preset").addEventListener("click", async () => {
    try {
      const name = $("preset_select").value;
      if (!name) {
        toast("请先选择预设");
        return;
      }
      const payload = await api("/api/presets/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      fillConfigToUI(payload.config || {});
      toast(`已应用预设: ${name}`);
      await loadState({ syncConfig: true });
    } catch (err) {
      toast(err.message);
    }
  });

  $("btn_delete_preset").addEventListener("click", async () => {
    try {
      const name = $("preset_select").value;
      if (!name) {
        toast("请先选择预设");
        return;
      }
      if (!window.confirm(`确认删除预设“${name}”吗？该操作不可撤销。`)) {
        return;
      }
      await api(`/api/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast(`已删除预设: ${name}`);
      await loadState();
    } catch (err) {
      toast(err.message);
    }
  });

  $("btn_export").addEventListener("click", () => {
    const exportWindow = window.open("/api/presets/export", "_blank", "noopener,noreferrer");
    if (exportWindow) {
      exportWindow.opener = null;
    }
  });

  $("import_file").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const formData = new FormData();
      formData.append("file", file);
      const payload = await api("/api/presets/import", {
        method: "POST",
        body: formData,
      });
      toast(`导入完成：预设 ${payload.imported_presets} 个`);
      event.target.value = "";
      await loadState({ syncConfig: true });
    } catch (err) {
      toast(err.message);
    }
  });

  $("btn_save_all").addEventListener("click", async (event) => {
    await runWithButtonLock(event.currentTarget, "保存中...", async () => {
      try {
        const outputDir = $("output_dir").value.trim();
        if (!outputDir) {
          toast("请填写输出目录");
          return;
        }
        const payload = await api("/api/tasks/save_all", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            output_dir: outputDir,
            filename_template: $("filename_template").value || "{src_stem}_{result_index}",
            replace_from: $("replace_from").value || "",
            replace_to: $("replace_to").value || "",
            create_txt: $("create_txt").value === "true",
            txt_template: $("txt_template").value || "{src_name}",
            only_selected: $("only_selected").value === "true",
            only_first_if_multiple: $("only_first_if_multiple").value === "true",
          }),
        });
        toast(`保存完成：图片 ${payload.saved_count} 张，TXT ${payload.txt_count} 个`);
        if (payload.error_count > 0) {
          console.warn("保存错误:", payload.errors);
        }
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    });
  });

  $("task_list").addEventListener("click", async (event) => {
    const retryId = event.target?.dataset?.retryId;
    const retryClearId = event.target?.dataset?.retryClearId;
    const deleteTaskId = event.target?.dataset?.deleteTaskId;
    const saveOneId = event.target?.dataset?.saveOneId;
    const copyDebugId = event.target?.dataset?.copyDebugId;
    const deleteId = event.target?.dataset?.deleteId;
    if (retryId) {
      const taskId = sanitizeTaskId(retryId);
      if (!taskId) {
        toast("任务 ID 无效，已拦截");
        return;
      }
      try {
        await api(`/api/tasks/${encodeURIComponent(taskId)}/retry`, { method: "POST" });
        toast("已重试任务（保留旧结果）");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (retryClearId) {
      const taskId = sanitizeTaskId(retryClearId);
      if (!taskId) {
        toast("任务 ID 无效，已拦截");
        return;
      }
      try {
        const payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/retry_clear`, { method: "POST" });
        toast(`已重试任务（已清除旧结果 ${payload.removed_old_results || 0} 个）`);
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (deleteTaskId) {
      const taskId = sanitizeTaskId(deleteTaskId);
      if (!taskId) {
        toast("任务 ID 无效，已拦截");
        return;
      }
      try {
        if (!window.confirm("确认删除该任务吗？该操作不可撤销。")) {
          return;
        }
        await api(`/api/tasks/${encodeURIComponent(taskId)}`, { method: "DELETE" });
        toast("已删除任务");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (saveOneId) {
      const pair = parseTaskResultPair(saveOneId);
      if (!pair) {
        toast("任务/结果 ID 无效，已拦截");
        return;
      }
      const { taskId, resultId } = pair;
      try {
        const outputDir = $("output_dir").value.trim();
        if (!outputDir) {
          toast("请先在 3) 中填写输出目录");
          return;
        }
        const payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/results/${encodeURIComponent(resultId)}/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            output_dir: outputDir,
            filename_template: $("filename_template").value || "{src_stem}_{result_index}",
            replace_from: $("replace_from").value || "",
            replace_to: $("replace_to").value || "",
            create_txt: $("create_txt").value === "true",
            txt_template: $("txt_template").value || "{src_name}",
          }),
        });
        const imageName = (payload.saved_image || "").split("/").pop() || "图片";
        toast(`单次保存完成：${imageName}`);
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (deleteId) {
      const pair = parseTaskResultPair(deleteId);
      if (!pair) {
        toast("任务/结果 ID 无效，已拦截");
        return;
      }
      const { taskId, resultId } = pair;
      try {
        if (!window.confirm("确认删除该结果吗？该操作不可撤销。")) {
          return;
        }
        await api(`/api/tasks/${encodeURIComponent(taskId)}/results/${encodeURIComponent(resultId)}`, { method: "DELETE" });
        toast("已删除该结果");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (copyDebugId) {
      const taskId = sanitizeTaskId(copyDebugId);
      if (!taskId) {
        toast("任务 ID 无效，已拦截");
        return;
      }
      const details = event.target.closest(".debug-details");
      const pre = details?.querySelector(".debug-log");
      try {
        const logText = await ensureDebugLog(taskId, pre);
        if (!logText) {
          toast("日志加载中，请稍后重试复制");
          return;
        }
        if (pre) {
          pre.textContent = logText;
        }
        await copyTextToClipboard(logText);
        toast("调试日志已复制");
      } catch (err) {
        toast(`复制失败: ${err.message}`);
      }
    }
  });

  $("task_list").addEventListener("change", async (event) => {
    const toggleId = event.target?.dataset?.toggleId;
    if (!toggleId) {
      return;
    }
    const pair = parseTaskResultPair(toggleId);
    if (!pair) {
      toast("任务/结果 ID 无效，已拦截");
      return;
    }
    const { taskId, resultId } = pair;
    try {
      await api(`/api/tasks/${encodeURIComponent(taskId)}/results/${encodeURIComponent(resultId)}/toggle`, { method: "POST" });
      await loadState();
    } catch (err) {
      toast(err.message);
    }
  });

  $("task_list").addEventListener("toggle", async (event) => {
    const details = event.target;
    if (!(details instanceof HTMLDetailsElement)) {
      return;
    }
    if (!details.classList.contains("debug-details")) {
      return;
    }
    const taskId = sanitizeTaskId(details.dataset.debugTaskId);
    if (!taskId) {
      return;
    }
    const pre = details.querySelector(".debug-log");
    if (!pre) {
      return;
    }

    if (!details.open) {
      state.debugOpenTaskIds.delete(taskId);
      return;
    }
    state.debugOpenTaskIds.add(taskId);

    try {
      const logText = await ensureDebugLog(taskId, pre);
      if (logText) {
        pre.textContent = logText;
      } else {
        pre.textContent = "日志加载中...";
      }
      details.dataset.loaded = "1";
    } catch (err) {
      pre.textContent = `日志读取失败: ${err.message}`;
    }
  }, true);
}

async function init() {
  bindEvents();
  await loadState({ syncConfig: true });
  state.timer = setInterval(() => loadState({ skipIfLoading: true }), 2500);
}

init();
