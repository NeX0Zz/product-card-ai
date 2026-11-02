// Карточка товара: интерфейс без сборки и без внешних зависимостей.
// Все данные от пользователя и модели попадают на страницу только через textContent и DOM API.

const SVG_NS = "http://www.w3.org/2000/svg";
const REQUEST_TIMEOUT_MS = 300_000;

/* ---------- DOM helpers ---------- */

const $ = (selector, root = document) => root.querySelector(selector);

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child != null && child !== false) el.append(child);
  }
  return el;
}

const ICONS = {
  check: [["circle", { cx: 8, cy: 8, r: 6.5 }], ["path", { d: "M5 8.2l2.1 2.1L11 6" }]],
  question: [["circle", { cx: 8, cy: 8, r: 6.5 }], ["path", { d: "M6.3 6.4a1.8 1.8 0 1 1 2.6 1.6c-.6.3-.9.7-.9 1.3M8 11.4v.1" }]],
  warning: [["path", { d: "M8 2.2l6.2 10.9H1.8z" }], ["path", { d: "M8 6.4v3M8 11.3v.1" }]],
  stop: [["circle", { cx: 8, cy: 8, r: 6.5 }], ["path", { d: "M5.2 8h5.6" }]],
  cross: [["circle", { cx: 8, cy: 8, r: 6.5 }], ["path", { d: "M5.8 5.8l4.4 4.4M10.2 5.8l-4.4 4.4" }]],
  info: [["circle", { cx: 8, cy: 8, r: 6.5 }], ["path", { d: "M8 7.2v4M8 4.7v.1" }]],
  copy: [["path", { d: "M5.5 5.5h7.5v8H5.5z" }], ["path", { d: "M3 10.5V2.5h7.5" }]],
  download: [["path", { d: "M8 2v8M4.6 6.8L8 10.2l3.4-3.4M3 13.5h10" }]],
  refresh: [["path", { d: "M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3" }]],
  plus: [["path", { d: "M8 3v10M3 8h10" }]],
  close: [["path", { d: "M4 4l8 8M12 4l-8 8" }]],
  chevron: [["path", { d: "M4 6l4 4 4-4" }]],
};

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  for (const [tag, attrs] of ICONS[name]) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    svg.append(node);
  }
  return svg;
}

const fmt = (n) => Number(n).toLocaleString("ru-RU");
const len = (text) => Array.from(text).length;
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- Elements and state ---------- */

const els = {
  banner: $("#llm-banner"),
  bannerText: $("#llm-banner-text"),
  examples: $("#examples-list"),
  tabForm: $("#tab-form"),
  tabJson: $("#tab-json"),
  panelForm: $("#panel-form"),
  panelJson: $("#panel-json"),
  form: $("#product-form"),
  name: $("#f-name"),
  manufacturer: $("#f-manufacturer"),
  description: $("#f-description"),
  specsList: $("#specs-list"),
  specsError: $("#err-specs"),
  addSpec: $("#add-spec"),
  json: $("#json-input"),
  jsonError: $("#json-error"),
  submit: $("#submit"),
  submitError: $("#submit-error"),
  resultTitle: $("#result-title"),
  empty: $("#empty"),
  progress: $("#progress"),
  elapsed: $("#elapsed"),
  result: $("#result"),
  live: $("#sr-live"),
};

const state = {
  mode: "form",
  busy: false,
  hasResult: false,
  rowSeq: 0,
  timer: null,
  jsonDebounce: null,
};

const FIELD_ERRORS = {
  name: [els.name, $("#err-name")],
  manufacturer: [els.manufacturer, $("#err-manufacturer")],
  description: [els.description, $("#err-description")],
};

function announce(message) {
  els.live.textContent = "";
  window.setTimeout(() => { els.live.textContent = message; }, 40);
}

/* ---------- Clipboard and download ---------- */

async function writeClipboard(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the legacy path */ }
  const area = h("textarea", { "aria-hidden": "true", tabindex: "-1", readonly: true });
  area.value = text;
  area.className = "sr-only";
  document.body.append(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  area.remove();
  return ok;
}

function copyButton({ getText, label, text = "Копировать", small = true, iconOnly = false }) {
  const btn = h(
    "button",
    {
      type: "button",
      class: `btn copy ${small ? "btn--sm" : ""} ${iconOnly ? "btn--icon" : ""}`,
      "aria-label": `${text} ${label}`.trim(),
    },
    h("span", { class: "copy__label" }, icon("copy"), iconOnly ? null : h("span", { text })),
    h("span", { class: "copy__done", "aria-hidden": "true" }, icon("check"), iconOnly ? null : h("span", { text: "Скопировано" })),
  );
  let reset = 0;
  btn.addEventListener("click", async () => {
    const ok = await writeClipboard(getText());
    if (!ok) {
      announce("Не удалось скопировать. Выделите текст и скопируйте вручную.");
      return;
    }
    btn.classList.add("is-copied");
    announce(`Скопировано: ${label}`);
    window.clearTimeout(reset);
    reset = window.setTimeout(() => btn.classList.remove("is-copied"), 1800);
  });
  return btn;
}

const TRANSLIT = {
  а: "a", б: "b", в: "v", г: "g", д: "d", е: "e", ё: "e", ж: "zh", з: "z", и: "i", й: "y", к: "k", л: "l", м: "m",
  н: "n", о: "o", п: "p", р: "r", с: "s", т: "t", у: "u", ф: "f", х: "h", ц: "ts", ч: "ch", ш: "sh", щ: "sch",
  ъ: "", ы: "y", ь: "", э: "e", ю: "yu", я: "ya",
};

function fileSlug(name) {
  const slug = Array.from((name || "").toLowerCase())
    .map((ch) => TRANSLIT[ch] ?? ch)
    .join("")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return slug || "kartochka-tovara";
}

function downloadText(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------- Field errors ---------- */

function setError(input, errEl, message) {
  if (input) input.setAttribute("aria-invalid", "true");
  errEl.textContent = message;
  errEl.hidden = false;
}

function clearError(input, errEl) {
  if (input) input.removeAttribute("aria-invalid");
  errEl.textContent = "";
  errEl.hidden = true;
}

function clearAllErrors() {
  for (const [input, errEl] of Object.values(FIELD_ERRORS)) clearError(input, errEl);
  clearError(null, els.specsError);
  for (const row of els.specsList.children) {
    clearError($(".spec-key", row), $(".field-error", row));
    $(".spec-val", row).removeAttribute("aria-invalid");
  }
  clearJsonError();
  clearSubmitError();
}

function clearJsonError() {
  els.jsonError.hidden = true;
  els.jsonError.replaceChildren();
  els.json.removeAttribute("aria-invalid");
}

function showAlert(container, title, items = [], kind = "error") {
  container.className = `alert${container === els.jsonError ? " alert--inline" : ""}${kind === "info" ? " alert--info" : ""}`;
  container.replaceChildren(
    icon(kind === "info" ? "info" : "cross"),
    h("div", {}, h("strong", { text: title }), items.length ? h("ul", {}, items.map((t) => h("li", { text: t }))) : null),
  );
  container.hidden = false;
}

function clearSubmitError() {
  els.submitError.hidden = true;
  els.submitError.replaceChildren();
}

/* ---------- Characteristics rows ---------- */

function addSpecRow(key = "", value = "", { focus = false } = {}) {
  const id = ++state.rowSeq;
  const keyInput = h("input", {
    type: "text", class: "spec-key", id: `sk${id}`, maxlength: 200, placeholder: "Параметр, например «класс точности»",
    "aria-describedby": `se${id}`,
  });
  const valInput = h("input", {
    type: "text", class: "spec-val", id: `sv${id}`, maxlength: 1000, placeholder: "Значение, например «1,5»",
    "aria-describedby": `se${id}`,
  });
  keyInput.value = key;
  valInput.value = value;
  const remove = h(
    "button",
    { type: "button", class: "btn btn--ghost btn--icon", "aria-label": "Удалить характеристику", title: "Удалить характеристику" },
    icon("close"),
  );
  const row = h(
    "div",
    { class: "spec-row", role: "group", "aria-label": "Характеристика" },
    h("label", { class: "sr-only", for: keyInput.id, text: "Название характеристики" }),
    keyInput,
    h("label", { class: "sr-only", for: valInput.id, text: "Значение характеристики" }),
    valInput,
    remove,
    h("p", { class: "field-error", id: `se${id}`, hidden: true }),
  );
  keyInput.addEventListener("input", () => {
    remove.setAttribute("aria-label", keyInput.value.trim() ? `Удалить характеристику «${keyInput.value.trim()}»` : "Удалить характеристику");
  });
  remove.addEventListener("click", () => {
    const next = row.nextElementSibling || row.previousElementSibling;
    row.remove();
    if (!els.specsList.children.length) addSpecRow();
    (next ? $(".spec-key", next) : $(".spec-key", els.specsList.firstElementChild))?.focus();
  });
  for (const input of [keyInput, valInput]) {
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.ctrlKey || event.metaKey || event.isComposing) return;
      event.preventDefault();
      if (input === keyInput) valInput.focus();
      else if (row.nextElementSibling) $(".spec-key", row.nextElementSibling).focus();
      else addSpecRow("", "", { focus: true });
    });
  }
  els.specsList.append(row);
  if (key) keyInput.dispatchEvent(new Event("input"));
  if (focus) keyInput.focus();
  return row;
}

function setSpecRows(specs) {
  els.specsList.replaceChildren();
  const entries = Object.entries(specs);
  if (!entries.length) addSpecRow();
  for (const [k, v] of entries) addSpecRow(k, v);
}

/* ---------- Data conversion ---------- */

const KNOWN_FIELDS = ["name", "manufacturer", "specs", "description"];

function scalarToText(value) {
  if (value == null) return "";
  if (typeof value === "boolean") return value ? "да" : "нет";
  if (typeof value === "string" || typeof value === "number") return String(value);
  return null;
}

const isPlainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

// Returns {ok: true, model} when the JSON fits the form, otherwise {ok: false, reason}.
function toFormModel(data) {
  if (!isPlainObject(data)) return { ok: false, reason: "Ожидается JSON-объект с данными одного товара." };
  const extra = Object.keys(data).filter((k) => !KNOWN_FIELDS.includes(k));
  if (extra.length) {
    return {
      ok: false,
      reason: `В JSON есть поля, которых нет в форме: ${extra.join(", ")}. Удалите их или продолжайте работу во вкладке JSON.`,
    };
  }
  const model = { name: "", manufacturer: "", description: "", specs: {} };
  for (const field of ["name", "manufacturer", "description"]) {
    const text = scalarToText(data[field]);
    if (text === null) return { ok: false, reason: `Поле «${field}» должно быть текстом, а не списком или объектом.` };
    model[field] = text;
  }
  if (data.specs != null) {
    if (!isPlainObject(data.specs)) return { ok: false, reason: "Поле «specs» должно быть объектом «параметр: значение»." };
    for (const [k, v] of Object.entries(data.specs)) {
      const text = scalarToText(v);
      if (text === null) return { ok: false, reason: `Значение характеристики «${k}» должно быть текстом или числом.` };
      model.specs[k] = text;
    }
  }
  return { ok: true, model };
}

function fillForm(model) {
  els.name.value = model.name;
  els.manufacturer.value = model.manufacturer;
  els.description.value = model.description;
  setSpecRows(model.specs);
}

// Reads the form. Returns {data, problems: [{row?, field, message}]}.
function collectForm() {
  const problems = [];
  const specs = {};
  const seen = new Set();
  for (const row of els.specsList.children) {
    const key = $(".spec-key", row).value.trim();
    const value = $(".spec-val", row).value.trim();
    if (!key && !value) continue;
    if (!key) {
      problems.push({ row, input: $(".spec-key", row), message: "Укажите название параметра или очистите строку." });
    } else if (seen.has(key)) {
      problems.push({ row, input: $(".spec-key", row), message: "Такой параметр уже есть. Оставьте одну строку." });
    } else {
      seen.add(key);
      specs[key] = value;
    }
  }
  const data = {
    name: els.name.value.trim(),
    manufacturer: els.manufacturer.value.trim(),
    specs,
    description: els.description.value.trim(),
  };
  return { data, problems };
}

function showFormProblems(problems) {
  for (const { row, input, message } of problems) {
    setError(input, $(".field-error", row), message);
  }
  problems[0]?.input.focus();
}

// Position of a JSON syntax error as {line, column}, or null if the browser gives none.
function jsonErrorPosition(text, error) {
  const message = String(error?.message ?? "");
  const lc = message.match(/line (\d+) column (\d+)/i);
  if (lc) return { line: Number(lc[1]), column: Number(lc[2]) };
  const pos = message.match(/position (\d+)/i);
  if (!pos) return null;
  const before = text.slice(0, Number(pos[1]));
  const lines = before.split("\n");
  return { line: lines.length, column: lines.at(-1).length + 1 };
}

// Parses the JSON area. On failure shows the message and returns {error: true}.
function parseJsonInput({ announceEmpty = true } = {}) {
  const text = els.json.value;
  clearJsonError();
  if (!text.trim()) {
    if (announceEmpty) {
      els.json.setAttribute("aria-invalid", "true");
      showAlert(els.jsonError, "Поле пустое. Вставьте JSON с данными товара или заполните форму.");
    }
    return { error: true };
  }
  try {
    return { data: JSON.parse(text) };
  } catch (error) {
    const pos = jsonErrorPosition(text, error);
    els.json.setAttribute("aria-invalid", "true");
    showAlert(
      els.jsonError,
      pos ? `Ошибка в JSON: строка ${pos.line}, столбец ${pos.column}.` : "Ошибка в JSON.",
      ["Проверьте запятые, кавычки и скобки. Введённый текст сохранён."],
    );
    return { error: true };
  }
}

/* ---------- Modes (tabs) ---------- */

function showMode(mode) {
  state.mode = mode;
  const isForm = mode === "form";
  els.tabForm.setAttribute("aria-selected", String(isForm));
  els.tabJson.setAttribute("aria-selected", String(!isForm));
  els.tabForm.tabIndex = isForm ? 0 : -1;
  els.tabJson.tabIndex = isForm ? -1 : 0;
  els.panelForm.hidden = !isForm;
  els.panelJson.hidden = isForm;
}

// Tries to switch the input mode. Returns false and explains if the data cannot be carried over.
function switchMode(target) {
  if (target === state.mode) return true;
  clearSubmitError();
  if (target === "json") {
    const { data, problems } = collectForm();
    if (problems.length) {
      showFormProblems(problems);
      announce("Исправьте отмеченные строки, чтобы перейти в JSON.");
      return false;
    }
    els.json.value = JSON.stringify(data, null, 2);
    clearJsonError();
    showMode("json");
    return true;
  }
  const parsed = parseJsonInput();
  if (parsed.error) return false;
  const model = toFormModel(parsed.data);
  if (!model.ok) {
    els.json.setAttribute("aria-invalid", "true");
    showAlert(els.jsonError, "Эти данные нельзя показать в форме.", [model.reason], "info");
    return false;
  }
  clearAllErrors();
  fillForm(model.model);
  showMode("form");
  return true;
}

function initTabs() {
  const tabs = [els.tabForm, els.tabJson];
  const modeOf = (tab) => (tab === els.tabForm ? "form" : "json");
  for (const tab of tabs) {
    tab.addEventListener("click", () => { switchMode(modeOf(tab)); });
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      else if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      else if (event.key === "Home") next = tabs[0];
      else if (event.key === "End") next = tabs.at(-1);
      if (next) {
        event.preventDefault();
        for (const t of tabs) t.tabIndex = t === next ? 0 : -1;
        next.focus();
      }
    });
  }
  els.json.addEventListener("input", () => {
    window.clearTimeout(state.jsonDebounce);
    state.jsonDebounce = window.setTimeout(() => {
      if (els.json.value.trim()) parseJsonInput({ announceEmpty: false });
      else clearJsonError();
    }, 500);
  });
}

// Puts product data into both representations.
function applyProduct(data) {
  clearAllErrors();
  els.json.value = JSON.stringify(data, null, 2);
  const model = toFormModel(data);
  if (model.ok) {
    fillForm(model.model);
    return;
  }
  fillForm({ name: "", manufacturer: "", description: "", specs: {} });
  showMode("json");
  showAlert(els.jsonError, "Эти данные нельзя показать в форме.", [model.reason], "info");
}

/* ---------- Health and examples ---------- */

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!health.llm_configured) {
      els.bannerText.replaceChildren(
        "Ключ модели не задан, поэтому карточки пока не формируются. Добавьте ",
        h("code", { text: "OPENROUTER_API_KEY" }),
        " в файл ",
        h("code", { text: ".env" }),
        " и перезапустите сервис.",
      );
      els.banner.hidden = false;
    }
  } catch {
    els.bannerText.textContent = "Не удалось связаться с сервисом. Проверьте, что он запущен, и обновите страницу.";
    els.banner.hidden = false;
  }
}

const EXAMPLE_LABELS = {
  manometer_dm02: "Манометр DM 02",
  pressure_transmitter_demo: "Демо: неполные данные",
};

async function loadExamples() {
  try {
    const response = await fetch("/api/examples");
    const { examples } = await response.json();
    for (const example of examples) {
      const label = EXAMPLE_LABELS[example.id] || example.data?.name || example.id;
      els.examples.append(
        h("button", { type: "button", class: "chip", onclick: () => applyProduct(example.data), text: label }),
        " ",
      );
    }
  } catch {
    /* Examples are optional: the form works without them. */
  }
}

/* ---------- Generation ---------- */

function setBusy(busy, fresh = false) {
  state.busy = busy;
  els.submit.setAttribute("aria-disabled", String(busy));
  els.submit.textContent = busy ? "Формируем…" : "Сформировать карточку";
  for (const btn of document.querySelectorAll("[data-regen]")) btn.setAttribute("aria-disabled", String(busy));
  els.progress.hidden = !busy;
  els.result.classList.toggle("is-stale", busy);
  els.result.toggleAttribute("inert", busy);
  els.result.setAttribute("aria-busy", String(busy));
  els.empty.hidden = busy || state.hasResult;
  window.clearInterval(state.timer);
  if (busy) {
    const started = Date.now();
    els.elapsed.textContent = "0";
    state.timer = window.setInterval(() => {
      els.elapsed.textContent = String(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    announce(fresh ? "Формируем карточку заново." : "Формируем карточку.");
    if (isStacked()) scrollToResult();
  }
}

const isStacked = () => window.matchMedia("(max-width: 999px)").matches;

function scrollToResult() {
  els.resultTitle.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
}

const HTTP_FALLBACKS = {
  413: "Данные слишком большие. Сократите описание.",
  415: "Сервис не принял формат запроса.",
  429: "Слишком много запросов. Подождите минуту и повторите.",
};

function mapDetails(details) {
  const unmapped = [];
  for (const { field, message } of details) {
    if (field in FIELD_ERRORS) {
      const [input, errEl] = FIELD_ERRORS[field];
      if (state.mode === "form") setError(input, errEl, message);
      else unmapped.push(`${fieldLabel(field)}: ${message}`);
    } else if (field === "specs" || field.startsWith("specs.")) {
      const key = field.slice(6);
      const row = key && [...els.specsList.children].find((r) => $(".spec-key", r).value.trim().slice(0, 60) === key);
      if (state.mode === "form" && row) setError($(".spec-key", row), $(".field-error", row), message);
      else if (state.mode === "form" && field === "specs") setError(null, els.specsError, message);
      else unmapped.push(`${fieldLabel(field)}: ${message}`);
    } else {
      unmapped.push(`${fieldLabel(field)}: ${message}`);
    }
  }
  return unmapped;
}

function showRequestError(status, body, fallback) {
  const error = body?.error;
  const message = error?.message || HTTP_FALLBACKS[status] || fallback
    || (status >= 500 ? "Сервис временно недоступен. Повторите чуть позже." : "Не удалось сформировать карточку.");
  const details = Array.isArray(error?.details) ? error.details : [];
  const unmapped = mapDetails(details);
  const mappedCount = details.length - unmapped.length;
  showAlert(els.submitError, message, unmapped);
  if (mappedCount && state.mode === "form") {
    const first = document.querySelector('#product-form [aria-invalid="true"]');
    first?.focus();
  }
}

async function submit({ fresh = false } = {}) {
  if (state.busy) return;
  clearAllErrors();

  let payload;
  if (state.mode === "form") {
    const { data, problems } = collectForm();
    if (problems.length) {
      showFormProblems(problems);
      return;
    }
    payload = data;
  } else {
    const parsed = parseJsonInput();
    if (parsed.error) {
      els.json.focus();
      return;
    }
    payload = parsed.data;
  }

  setBusy(true, fresh);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let outcome = null;
  try {
    const response = await fetch(`/api/generate${fresh ? "?fresh=true" : ""}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    let body = null;
    try { body = await response.json(); } catch { body = null; }
    outcome = response.ok && body?.card ? { body } : { status: response.status, body };
  } catch (error) {
    outcome = {
      network: true,
      timeout: error?.name === "AbortError",
    };
  } finally {
    window.clearTimeout(timeout);
  }
  setBusy(false);

  if (outcome.body && !outcome.status) {
    renderResult(outcome.body, payload);
    return;
  }
  if (outcome.network) {
    showAlert(
      els.submitError,
      outcome.timeout ? "Сервис слишком долго не отвечает." : "Нет связи с сервером.",
      [outcome.timeout ? "Повторите запрос чуть позже." : "Проверьте, что сервис запущен, и повторите попытку. Введённые данные сохранены."],
    );
    return;
  }
  showRequestError(outcome.status, outcome.body);
}

/* ---------- Result rendering ---------- */

const FIELD_LABELS = {
  name: "Название",
  manufacturer: "Производитель",
  description: "Описание",
  specs: "Характеристики",
  seo_title: "SEO-заголовок",
  seo_description: "SEO-описание",
  search_queries: "Поисковые запросы",
  characteristics: "Характеристики",
  "commercial.suitable_for": "Кому подходит",
  "commercial.solves": "Какие задачи решает",
};

function fieldLabel(field) {
  if (FIELD_LABELS[field]) return FIELD_LABELS[field];
  const query = field.match(/^search_queries\[(\d+)\]$/);
  if (query) return `Поисковый запрос ${Number(query[1]) + 1}`;
  const spec = field.match(/^(?:specs|characteristics)\.(.+)$/);
  if (spec) return `Характеристика «${spec[1]}»`;
  return field || "Карточка";
}

const STATUS = {
  ready: {
    icon: "check",
    title: "Можно проверять и публиковать",
    text: "Автоматические проверки пройдены. Перед публикацией всё равно прочитайте текст.",
  },
  needs_clarification: {
    icon: "question",
    title: "Есть что уточнить",
    text: "Карточка собрана, но часть данных не подтверждена. Уточните их у поставщика и сформируйте карточку заново.",
  },
  needs_review: {
    icon: "warning",
    title: "Есть ошибки: не публикуйте без правки",
    text: "Автоматическая проверка нашла проблемы, которые не удалось устранить. Они перечислены ниже.",
  },
  insufficient_data: {
    icon: "stop",
    title: "Недостаточно данных",
    text: "По этим данным карточку не собрать. Добавьте название и характеристики и повторите.",
  },
};

function renderStatus(card) {
  const meta = STATUS[card.status] || { icon: "info", title: "Результат получен", text: "" };
  return h(
    "div",
    { class: `status status--${card.status}` },
    icon(meta.icon),
    h(
      "div",
      {},
      h("p", { class: "status__title", text: meta.title }),
      meta.text ? h("p", { class: "status__text", text: meta.text }) : null,
      card.product_name ? h("p", { class: "status__product" }, "Товар: ", h("b", { text: card.product_name })) : null,
    ),
  );
}

function section(title, { tools = null, count = null, first = false, labelId = null } = {}, ...content) {
  return h(
    "section",
    { class: `section${first ? " section--first" : ""}`, "aria-labelledby": labelId },
    h(
      "div",
      { class: "section__head" },
      h("h3", { class: "section__title", id: labelId }, title, count ? h("span", { class: "badge", text: String(count) }) : null),
      tools,
    ),
    content,
  );
}

function renderAttention(card) {
  // Items of the same kind for the same field are merged into one row.
  const groups = new Map();
  const add = (cls, iconName, kind, field, message, token) => {
    const key = `${cls}|${field}`;
    if (!groups.has(key)) groups.set(key, { cls, icon: iconName, kind, field, entries: [] });
    groups.get(key).entries.push({ message, token });
  };
  const issues = card.issues || [];
  for (const i of issues.filter((x) => x.severity === "error")) add("error", "cross", "Ошибка", i.field, i.message, i.token);
  for (const i of issues.filter((x) => x.severity !== "error")) add("warning", "warning", "Предупреждение", i.field, i.message, i.token);
  for (const m of card.missing_fields || []) add("missing", "stop", "Не хватает данных", m.field, m.message);
  for (const c of card.clarifications || []) add("clarify", "question", "Нужно уточнить", c.field, c.question);
  const items = [...groups.values()];
  if (!items.length) return null;
  const total = items.reduce((n, g) => n + g.entries.length, 0);

  const list = h(
    "ul",
    { class: "attention__list" },
    items.map((item) =>
      h(
        "li",
        { class: `att att--${item.cls}` },
        h("span", { class: "att__kind" }, icon(item.icon), item.kind),
        h(
          "div",
          {},
          h("p", { class: "att__field", text: fieldLabel(item.field) }),
          item.entries.map((e) =>
            h(
              "div",
              { class: "att__entry" },
              h(
                "p",
                { class: "att__msg" },
                e.message,
                e.token ? h("span", { class: "att__token" }, " Фрагмент: ", h("code", { text: e.token })) : null,
              ),
            ),
          ),
        ),
      ),
    ),
  );

  const questions = (card.clarifications || []).map((c, i) => `${i + 1}. ${c.question}`);
  const foot = questions.length
    ? h(
        "div",
        { class: "attention__foot" },
        copyButton({
          label: "вопросы поставщику",
          text: "Скопировать",
          getText: () => `Уточните, пожалуйста, по товару «${card.product_name || "без названия"}»:\n${questions.join("\n")}\n`,
        }),
      )
    : null;

  const box = h("div", { class: "attention" }, list, foot);
  return section("Требует внимания", { count: total, first: true, labelId: "sec-attention" }, box);
}

// Appends text to `el`, wrapping each known token in <mark>. No HTML parsing is involved.
function appendMarked(el, text, tokens) {
  const needles = [...new Set(tokens)].filter(Boolean).sort((a, b) => b.length - a.length);
  const lower = text.toLowerCase();
  // Case-insensitive search is safe only when lowercasing keeps every index in place.
  const haystack = lower.length === text.length ? lower : text;
  const lowNeedles = needles.map((t) => (haystack === lower ? t.toLowerCase() : t));
  let pos = 0;
  while (pos < text.length) {
    let best = -1;
    let bestLen = 0;
    lowNeedles.forEach((token, i) => {
      const at = haystack.indexOf(token, pos);
      if (at !== -1 && (best === -1 || at < best)) {
        best = at;
        bestLen = needles[i].length;
      }
    });
    if (best === -1) break;
    if (best > pos) el.append(text.slice(pos, best));
    el.append(h("mark", { text: text.slice(best, best + bestLen) }));
    pos = best + bestLen;
  }
  if (pos < text.length) el.append(text.slice(pos));
}

function gauge({ value, low, high, scaleMax, ok }) {
  const el = h("div", { class: `gauge ${ok ? "gauge--ok" : "gauge--bad"}`, "aria-hidden": "true" }, h("span", { class: "gauge__band" }), h("span", { class: "gauge__fill" }));
  const pct = (n) => String(Math.min(100, (n / scaleMax) * 100).toFixed(2));
  el.style.setProperty("--v", pct(value));
  el.style.setProperty("--lo", pct(low));
  el.style.setProperty("--hi", pct(high));
  return el;
}

function counterFor({ count, low, high, kind }) {
  const ok = count >= low && count <= high;
  let stateText = "в норме";
  if (count === 0) stateText = "пусто";
  else if (count < low) stateText = `короче нормы на ${fmt(low - count)}`;
  else if (count > high) stateText = `длиннее нормы на ${fmt(count - high)}`;
  const valueText = kind === "title" ? `${fmt(count)} / ${fmt(high)}` : fmt(count);
  const target = kind === "title" ? null : `цель ${fmt(low)}–${fmt(high)} знаков`;
  return {
    ok,
    node: h(
      "p",
      { class: `counter ${ok ? "counter--ok" : "counter--bad"}` },
      h("span", { class: "counter__value", text: valueText }),
      h("span", { class: "sr-only", text: kind === "title" ? " знаков из допустимых" : " знаков" }),
      target ? h("span", { text: target }) : null,
      h("span", { class: "counter__state", text: ok ? `✓ ${stateText}` : `! ${stateText}` }),
    ),
  };
}

function renderSeo(card, tokens) {
  const blocks = [];
  let anyMark = false;

  const make = ({ title, text, kind, low, high, scaleMax, labelId }) => {
    const count = len(text);
    const { ok, node } = counterFor({ count, low, high, kind });
    const body = h("p", { class: `seo-text${kind === "title" ? " seo-text--title" : ""}`, id: labelId });
    appendMarked(body, text, tokens);
    if (body.querySelector("mark")) anyMark = true;
    return h(
      "div",
      { class: "text-block" },
      h(
        "div",
        { class: "text-block__label" },
        h("h4", { class: "text-block__name", text: title }),
        h("div", { class: "text-block__tools" }, node, copyButton({ label: title, getText: () => text })),
      ),
      body,
      gauge({ value: count, low, high, scaleMax: Math.max(scaleMax, count), ok }),
    );
  };

  if (card.seo_title) {
    blocks.push(make({ title: "SEO-заголовок", text: card.seo_title, kind: "title", low: 1, high: 70, scaleMax: 80, labelId: "seo-title-text" }));
  }
  if (card.seo_description) {
    blocks.push(make({ title: "SEO-описание", text: card.seo_description, kind: "description", low: 800, high: 1200, scaleMax: 1400, labelId: "seo-desc-text" }));
  }
  if (!blocks.length) return null;
  if (anyMark) {
    blocks.push(h("p", { class: "mark-legend" }, h("mark", { text: "Выделенный" }), " фрагмент не подтверждён данными. Проверьте или замените его."));
  }
  return section("Тексты для карточки", { labelId: "sec-seo" }, blocks);
}

function renderCharacteristics(card) {
  const rows = card.characteristics || [];
  if (!rows.length) return null;
  let quoteSeq = 0;

  const body = rows.map((c) => {
    const noData = c.value.trim() === "данных нет" || c.source === "none";
    const valueNode = noData
      ? h("span", { class: "nodata", text: c.value || "данных нет" })
      : h("span", { class: "spec-value", text: c.value });
    const cell = h("td", {}, valueNode);

    if (c.source === "description") {
      if (c.quote) {
        const quoteId = `quote-${++quoteSeq}`;
        const quote = h("p", { class: "quote", id: quoteId, hidden: true }, h("b", { text: "Фрагмент описания" }), `«${c.quote}»`);
        const tag = h(
          "button",
          {
            type: "button", class: "tag", "aria-expanded": "false", "aria-controls": quoteId,
            title: `Фрагмент описания: «${c.quote}»`,
          },
          "из описания",
          icon("chevron"),
        );
        tag.addEventListener("click", () => {
          const open = tag.getAttribute("aria-expanded") === "true";
          tag.setAttribute("aria-expanded", String(!open));
          quote.hidden = open;
        });
        cell.append(tag, quote);
      } else {
        cell.append(h("span", { class: "tag", text: "из описания" }));
      }
    }
    return h("tr", {}, h("th", { scope: "row", text: c.name }), cell);
  });

  const table = h(
    "table",
    { class: "spec-table" },
    h("caption", { class: "sr-only", text: "Характеристики товара" }),
    h("thead", {}, h("tr", {}, h("th", { scope: "col", text: "Параметр" }), h("th", { scope: "col", text: "Значение" }))),
    h("tbody", {}, body),
  );
  const copyAll = copyButton({ label: "характеристики", getText: () => rows.map((c) => `${c.name}: ${c.value}`).join("\n") });
  return section("Характеристики", { labelId: "sec-chars", tools: copyAll }, h("div", { class: "table-wrap" }, table));
}

function renderQueries(card) {
  const queries = card.search_queries || [];
  if (!queries.length) return null;
  const list = h(
    "ul",
    { class: "queries" },
    queries.map((q) =>
      h("li", { class: "query" }, h("span", { class: "query__text", text: q }), copyButton({ label: `запрос «${q}»`, getText: () => q })),
    ),
  );
  const all = copyButton({ label: "все запросы", text: "Копировать все", getText: () => queries.join("\n") });
  return section("Поисковые запросы", { labelId: "sec-queries", tools: all }, list);
}

function renderCommercial(card) {
  const { suitable_for: suitable = [], solves = [] } = card.commercial || {};
  if (!suitable.length && !solves.length) return null;
  const column = (title, items) =>
    items.length ? h("div", {}, h("h4", { text: title }), h("ul", { class: "bullets" }, items.map((t) => h("li", { text: t })))) : null;
  return section(
    "Для менеджера по продажам",
    { labelId: "sec-commercial" },
    h("div", { class: "cols" }, column("Кому подходит", suitable), column("Какие задачи решает", solves)),
  );
}

function renderExport(card, exports, payload) {
  const { meta, ...withoutMeta } = card;
  const json = JSON.stringify(withoutMeta, null, 2);
  const markdown = exports?.markdown ?? "";
  const text = exports?.text ?? "";

  const download = h(
    "button",
    { type: "button", class: "btn btn--sm" },
    icon("download"),
    "Скачать .md",
  );
  download.addEventListener("click", () => {
    downloadText(`${fileSlug(card.product_name)}.md`, markdown, "text/markdown;charset=utf-8");
    announce("Файл .md сохранён.");
  });

  const regen = h("button", { type: "button", class: "btn btn--sm", "data-regen": "" }, icon("refresh"), "Сгенерировать заново");
  regen.addEventListener("click", () => {
    if (state.busy) return;
    submit({ fresh: true });
  });

  return section(
    "Экспорт",
    { labelId: "sec-export" },
    h(
      "div",
      { class: "export" },
      copyButton({ label: "карточку в JSON", text: "Копировать JSON", getText: () => json }),
      copyButton({ label: "карточку в Markdown", text: "Копировать Markdown", getText: () => markdown }),
      copyButton({ label: "карточку как текст", text: "Копировать текст", getText: () => text }),
      download,
    ),
    h(
      "div",
      { class: "export__again" },
      regen,
      h("p", { class: "hint", text: card.meta?.cached ? "Этот ответ взят из кэша. Новый запрос обратится к модели заново." : "Новый запрос обратится к модели без кэша." }),
    ),
  );
}

function renderMeta(meta) {
  if (!meta) return null;
  const parts = [];
  if (meta.models_used?.length) parts.push(`Модель: ${meta.models_used.join(", ")}`);
  if (meta.attempts) parts.push(`Попыток: ${meta.attempts}`);
  const tokens = (meta.prompt_tokens || 0) + (meta.completion_tokens || 0);
  if (tokens) parts.push(`Токены: ${fmt(tokens)}`);
  if (typeof meta.cost_usd === "number") {
    parts.push(`Стоимость: $${meta.cost_usd.toLocaleString("ru-RU", { maximumFractionDigits: 6 })}`);
  }
  if (meta.duration_ms) {
    const sec = meta.duration_ms / 1000;
    parts.push(`${sec.toLocaleString("ru-RU", { maximumFractionDigits: sec < 10 ? 1 : 0 })} с`);
  }
  if (meta.cached) parts.push("из кэша");
  if (meta.prompt_version) parts.push(`промпт ${meta.prompt_version}`);
  if (!parts.length) return null;
  return h("p", { class: "meta" }, parts.map((p) => h("span", { text: p })));
}

function renderResult({ card, exports }, payload) {
  const tokens = (card.issues || []).map((i) => i.token).filter(Boolean);
  const parts = [
    renderStatus(card),
    renderAttention(card),
    renderSeo(card, tokens),
    renderCharacteristics(card),
    renderQueries(card),
    renderCommercial(card),
    renderExport(card, exports, payload),
    renderMeta(card.meta),
  ];
  state.hasResult = true;
  els.result.replaceChildren(...parts.filter(Boolean));
  els.result.hidden = false;
  els.empty.hidden = true;
  // Restart the entrance animation.
  els.result.style.animation = "none";
  void els.result.offsetWidth;
  els.result.style.removeProperty("animation");

  const status = STATUS[card.status]?.title ?? "результат получен";
  announce(`Карточка сформирована. ${status}.`);
  els.resultTitle.focus({ preventScroll: true });
  if (isStacked()) scrollToResult();
}

/* ---------- Start ---------- */

function init() {
  els.kbd = $("#kbd-mod");
  if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) els.kbd.textContent = "⌘";

  addSpecRow();
  initTabs();

  els.addSpec.addEventListener("click", () => addSpecRow("", "", { focus: true }));
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  els.submit.addEventListener("click", (event) => {
    if (state.busy) event.preventDefault();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      submit();
    }
  });
  for (const [field, [input, errEl]] of Object.entries(FIELD_ERRORS)) {
    input.addEventListener("input", () => clearError(input, errEl));
    void field;
  }
  els.addSpec.addEventListener("click", () => clearError(null, els.specsError));

  loadHealth();
  loadExamples();
}

init();
