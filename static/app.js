const state = {
  latest: null,
  timer: null,
  debugOpenTaskIds: new Set(),
  debugLogCache: {},
  debugLoadingTaskIds: new Set(),
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
  if (state.debugLogCache[taskId]) {
    return state.debugLogCache[taskId];
  }
  if (state.debugLoadingTaskIds.has(taskId)) {
    return null;
  }
  if (preElement) {
    preElement.textContent = "日志加载中...";
  }
  state.debugLoadingTaskIds.add(taskId);
  try {
    const payload = await api(`/api/tasks/${taskId}/debug`);
    const logText = payload.has_debug_log ? payload.debug_log : "暂无调试日志（任务可能尚未执行完成）。";
    state.debugLogCache[taskId] = logText;
    return logText;
  } finally {
    state.debugLoadingTaskIds.delete(taskId);
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
    const taskId = node.dataset.debugTaskId;
    if (taskId) {
      state.debugOpenTaskIds.add(taskId);
    }
  });
}

function cleanupDebugState(tasks) {
  const validTaskIds = new Set((tasks || []).map((task) => task.id));
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

function renderTasks(tasks) {
  const list = $("task_list");
  cleanupDebugState(tasks);
  list.innerHTML = "";
  $("task_stats").textContent = `任务：${tasks.length}`;

  tasks.forEach((task) => {
    const taskItem = document.createElement("div");
    taskItem.className = "task-item";

    const resultHtml = task.results
      .filter((x) => !x.deleted)
      .map((result) => {
        const checked = result.selected ? "checked" : "";
        const resultSourceType = escapeHtml(result.source_type || "");
        return `
          <div class="result-row">
            <div class="pair-preview">
              <div>
                <div class="img-label">原图</div>
                <img src="${task.source_image_url}" alt="原图预览" loading="lazy" />
              </div>
              <div>
                <div class="img-label">返回图</div>
                <img src="${result.image_url}" alt="返回图预览" loading="lazy" />
              </div>
              <div class="row-actions">
                <label><input type="checkbox" data-toggle-id="${task.id}::${result.id}" ${checked} /> 选中保存</label>
                <button data-save-one-id="${task.id}::${result.id}" class="primary">单次保存</button>
                <button data-delete-id="${task.id}::${result.id}" class="danger">删除此结果</button>
                <span class="task-meta">${result.width}x${result.height} | ${resultSourceType}</span>
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
    const isDebugOpen = state.debugOpenTaskIds.has(task.id);
    const isDebugLoading = state.debugLoadingTaskIds.has(task.id);
    const cachedDebugLog = state.debugLogCache[task.id] || "";
    const debugText = cachedDebugLog || (isDebugLoading ? "日志加载中..." : "点击展开后加载日志...");
    const debugLoaded = cachedDebugLog ? "1" : "0";

    taskItem.innerHTML = `
      <div class="task-head">
        <div>
          <strong>${safeSourceName}</strong>
          <div class="task-meta">${safeSourcePath}</div>
          <div class="task-meta">尝试次数: ${task.attempts} | HTTP: ${task.http_status || "-"}</div>
          ${safeLastError ? `<div class="task-error">${safeLastError}</div>` : ""}
          ${safeLastWarning ? `<div><span class="warning-tag" title="${safeLastWarning}">⚠ ${safeLastWarning}</span></div>` : ""}
        </div>
        <div>
          <span class="status ${statusClass}">${statusText}</span>
          <button data-retry-clear-id="${task.id}">重试（清除已有结果）</button>
          <button data-retry-id="${task.id}">重试（保留已有结果）</button>
        </div>
      </div>
      <div class="result-list">${resultHtml || `<div class="task-meta">暂无返回图片</div>`}</div>
      <details class="debug-details" data-debug-task-id="${task.id}" data-loaded="${debugLoaded}" ${isDebugOpen ? "open" : ""}>
        <summary>调试日志（默认折叠，点击展开）</summary>
        <div class="debug-tools">
          <button class="copy-log-btn" data-copy-debug-id="${task.id}">复制日志</button>
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
  try {
    captureDebugOpenStateFromDOM();
    const payload = await api("/api/state");
    state.latest = payload;
    if (syncConfig && !isConfigEditing()) {
      fillConfigToUI(payload.current_config || {});
    }
    renderPresets(payload.presets || {});
    renderProgress(payload.progress || {});
    renderTasks(payload.tasks || []);
    const runStatusEl = $("run_status");
    if (payload.running) {
      runStatusEl.textContent = "状态：批处理运行中";
      runStatusEl.dataset.state = "running";
    } else if (payload.stop_requested) {
      runStatusEl.textContent = "状态：已紧急停止";
      runStatusEl.dataset.state = "stopped";
    } else {
      runStatusEl.textContent = "状态：待机";
      runStatusEl.dataset.state = "idle";
    }
  } catch (err) {
    toast(`刷新失败: ${err.message}`);
  }
}

function bindEvents() {
  $("btn_refresh").addEventListener("click", loadState);
  $("txt_var_guide")?.addEventListener("click", (event) => {
    const btn = event.target?.closest?.("button[data-insert-value]");
    if (!btn) {
      return;
    }
    const value = btn.dataset.insertValue || "";
    insertTextAtCursor($("txt_template"), value);
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
      await api(`/api/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast(`已删除预设: ${name}`);
      await loadState();
    } catch (err) {
      toast(err.message);
    }
  });

  $("btn_export").addEventListener("click", () => {
    window.open("/api/presets/export", "_blank");
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
    const saveOneId = event.target?.dataset?.saveOneId;
    const copyDebugId = event.target?.dataset?.copyDebugId;
    const deleteId = event.target?.dataset?.deleteId;
    if (retryId) {
      try {
        await api(`/api/tasks/${retryId}/retry`, { method: "POST" });
        toast("已重试任务（保留旧结果）");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (retryClearId) {
      try {
        const payload = await api(`/api/tasks/${retryClearId}/retry_clear`, { method: "POST" });
        toast(`已重试任务（已清除旧结果 ${payload.removed_old_results || 0} 个）`);
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (saveOneId) {
      const [taskId, resultId] = saveOneId.split("::");
      try {
        const outputDir = $("output_dir").value.trim();
        if (!outputDir) {
          toast("请先在 3) 中填写输出目录");
          return;
        }
        const payload = await api(`/api/tasks/${taskId}/results/${resultId}/save`, {
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
      const [taskId, resultId] = deleteId.split("::");
      try {
        await api(`/api/tasks/${taskId}/results/${resultId}`, { method: "DELETE" });
        toast("已删除该结果");
        await loadState();
      } catch (err) {
        toast(err.message);
      }
    }
    if (copyDebugId) {
      const details = event.target.closest(".debug-details");
      const pre = details?.querySelector(".debug-log");
      try {
        const logText = await ensureDebugLog(copyDebugId, pre);
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
    const [taskId, resultId] = toggleId.split("::");
    try {
      await api(`/api/tasks/${taskId}/results/${resultId}/toggle`, { method: "POST" });
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
    const taskId = details.dataset.debugTaskId;
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
  state.timer = setInterval(loadState, 2500);
}

init();
