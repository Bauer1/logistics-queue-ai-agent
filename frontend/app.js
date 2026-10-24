let currentFacilityId = "FAC00012";
let currentScheduleData = [];
let currentTableFilter = "ALL";
let hourlyChartInstance = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadOverview();
  await runOptimization();
  await runMLPrediction();
  appendAgentGreeting();
});

async function loadOverview() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();

    const kpis = data.network_kpis;
    document.getElementById("network-telemetry-pill").innerHTML = `
      <span class="status-dot"></span>
      <span>Сеть: <b>${kpis.total_facilities}</b> складов • <b>${kpis.total_delivery_events.toLocaleString("ru-RU")}</b> событий • Ср. простой: <b>${kpis.avg_network_detention_min} мин</b></span>
    `;

    if (kpis.ml_model_metrics && kpis.ml_model_metrics.mae_minutes) {
      document.getElementById("ml-metrics-badge").textContent =
        `Обучено на ${kpis.ml_model_metrics.training_samples.toLocaleString("ru-RU")} событиях • MAE: ${kpis.ml_model_metrics.mae_minutes} мин`;
    }

    // Populate facility dropdown
    const sel = document.getElementById("facility-select");
    sel.innerHTML = "";
    data.facilities.forEach((f) => {
      const opt = document.createElement("option");
      opt.value = f.facility_id;
      opt.textContent = `${f.facility_id} — ${f.facility_name} (${f.city}, ${f.state}) | Режим: ${f.operating_hours} | Доков: ${f.dock_doors} | Простой: ${f.avg_detention} мин`;
      if (f.facility_id === currentFacilityId) opt.selected = true;
      sel.appendChild(opt);
    });

    // Populate top bottleneck preset chips
    const presetContainer = document.getElementById("preset-bottlenecks");
    presetContainer.innerHTML = `<span class="preset-label">🔥 Топ узких мест сети:</span>`;
    data.facilities.slice(0, 6).forEach((f) => {
      const chip = document.createElement("button");
      chip.className = `preset-chip ${f.facility_id === currentFacilityId ? "active" : ""}`;
      chip.dataset.fid = f.facility_id;
      chip.textContent = `${f.facility_id} ${f.city} (${f.operating_hours})`;
      chip.onclick = () => {
        document.getElementById("facility-select").value = f.facility_id;
        runOptimization();
        runMLPrediction();
      };
      presetContainer.appendChild(chip);
    });
  } catch (err) {
    console.error("Failed to load overview:", err);
  }
}

async function runOptimization() {
  const facSelect = document.getElementById("facility-select");
  if (facSelect && facSelect.value) {
    currentFacilityId = facSelect.value;
  }

  // Highlight active preset chip
  document.querySelectorAll(".preset-chip").forEach((c) => {
    c.classList.toggle("active", c.dataset.fid === currentFacilityId);
  });

  const surgeExtra = parseInt(document.getElementById("surge-select").value || "0", 10);
  const baysOverride = parseInt(document.getElementById("bays-select").value || "0", 10);
  const allowReroute = document.getElementById("chk-city-reroute").checked;

  try {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        facility_id: currentFacilityId,
        window_days: 3,
        active_bays_override: baysOverride > 0 ? baysOverride : null,
        allow_city_reroute: allowReroute,
        surge_extra_trucks: surgeExtra,
      }),
    });
    const data = await res.json();
    renderOptimizationResult(data);
  } catch (err) {
    console.error("Optimization error:", err);
  }
}

function renderOptimizationResult(data) {
  const fac = data.facility;
  const k = data.kpis;
  currentScheduleData = data.schedule || [];

  // KPIs
  document.getElementById("kpi-avg-after").textContent = `${k.avg_detention_after_min} мин`;
  document.getElementById("kpi-avg-before").textContent = `${k.avg_detention_before_min} мин`;
  document.getElementById("kpi-reduction-pct").textContent = `-${k.detention_reduction_pct}%`;
  document.getElementById("kpi-total-saved").textContent =
    `Сэкономлено: ${k.total_detention_saved_min} мин (${(k.total_detention_saved_min / 60).toFixed(1)} ч) на ${k.total_trucks} рейсах`;

  document.getElementById("kpi-ooh-caught").textContent = `${k.out_of_hours_arrivals_caught} фур`;
  document.getElementById("kpi-fac-hours").textContent = `Режим: ${fac.operating_hours}`;
  document.getElementById("kpi-reroute-sub").textContent =
    `Перенаправлено в открытые хабы г. ${fac.city}: ${k.rerouted_to_city_hubs} фур • Перенесено слотов: ${k.slot_shifted_trucks}`;

  document.getElementById("kpi-fast-track").textContent = `${k.fast_tracked_trucks} фур`;
  document.getElementById("kpi-reefer-saved").textContent = `❄️ ${k.reefer_loads_protected} реф.`;

  document.getElementById("kpi-cost-saved").textContent = `$${k.estimated_cost_saved_usd.toLocaleString("en-US")}`;
  document.getElementById("kpi-fuel-saved").textContent = `${k.fuel_saved_gallons} гал. дизеля`;
  document.getElementById("kpi-co2-saved").textContent =
    `Снижение выбросов: -${k.co2_saved_kg} кг CO₂ • Устранение холостого хода`;

  document.getElementById("chart-hours-hint").textContent =
    `${fac.facility_name} (${fac.city}) • Рабочие часы: ${fac.operating_hours}`;
  document.getElementById("active-bays-badge").textContent =
    `Активных бригад/доков: ${fac.active_bays} (всего ворот: ${fac.dock_doors})`;

  renderHourlyChart(data.hourly_profile, fac);
  renderDockBaysBoard(data.schedule);
  renderScheduleTable();
}

function renderHourlyChart(hourlyProfile, fac) {
  const ctx = document.getElementById("hourlyChart").getContext("2d");
  const labels = hourlyProfile.map((h) => h.label);
  const beforeData = hourlyProfile.map((h) => h.avg_detention_before);
  const afterData = hourlyProfile.map((h) => h.avg_detention_after);

  if (hourlyChartInstance) {
    hourlyChartInstance.destroy();
  }

  hourlyChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Простой ДО AI (мин)",
          data: beforeData,
          backgroundColor: "rgba(244, 63, 94, 0.55)",
          borderColor: "#f43f5e",
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: "Простой ПОСЛЕ AI (мин)",
          data: afterData,
          backgroundColor: "rgba(16, 185, 129, 0.75)",
          borderColor: "#10b981",
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#cbd5e1", font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            afterLabel: (context) => {
              const hObj = hourlyProfile[context.dataIndex];
              return hObj.is_open ? "✅ Склад открыт" : "⛔ Вне рабочих часов склада";
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { size: 10 } },
          grid: { color: "rgba(30, 41, 59, 0.5)" },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Минуты простоя", color: "#94a3b8", font: { size: 11 } },
          ticks: { color: "#94a3b8", font: { size: 10 } },
          grid: { color: "rgba(30, 41, 59, 0.6)" },
        },
      },
    },
  });
}

function renderDockBaysBoard(schedule) {
  const board = document.getElementById("dock-bays-board");
  board.innerHTML = "";

  const byDock = {};
  schedule.forEach((item) => {
    const d = item.assigned_dock || "Док #1";
    if (!byDock[d]) byDock[d] = [];
    byDock[d].push(item);
  });

  Object.keys(byDock)
    .sort()
    .forEach((dockName) => {
      const items = byDock[dockName];
      const row = document.createElement("div");
      row.className = "dock-bay-row";

      const isHub = dockName.includes("Хаб");
      const headerHtml = `
        <div class="dock-bay-header">
          <span>${isHub ? "🔄 " : "🚪 "}${dockName}</span>
          <span style="font-size:0.73rem; color:#94a3b8;">Обслужено: ${items.length} фур</span>
        </div>
      `;

      const slotsHtml = items
        .map((it) => {
          const timePart = it.optimized_start_time.split(" ")[1] || it.optimized_start_time;
          const endPart = it.optimized_end_time.split(" ")[1] || "";
          const isReefer = it.load_type === "Refrigerated";
          const cls = isHub ? "rerouted" : isReefer ? "reefer" : "";
          const icon = isReefer ? "❄️" : "📦";
          return `
            <div class="slot-pill ${cls}" title="${it.load_id} • ${it.customer_name} • Приоритет: ${it.priority_score} • Экономия: -${it.detention_saved_min} мин">
              <span>${icon}</span>
              <b>${timePart}–${endPart}</b>
              <span style="color:#94a3b8;">${it.load_id.slice(-4)}</span>
            </div>
          `;
        })
        .join("");

      row.innerHTML = `${headerHtml}<div class="dock-slots-track">${slotsHtml}</div>`;
      board.appendChild(row);
    });
}

function setTableFilter(filterType, btnEl) {
  currentTableFilter = filterType;
  document.querySelectorAll(".filter-tab").forEach((b) => b.classList.remove("active"));
  if (btnEl) btnEl.classList.add("active");
  renderScheduleTable();
}

function renderScheduleTable() {
  const tbody = document.getElementById("schedule-table-body");
  tbody.innerHTML = "";

  let filtered = currentScheduleData;
  if (currentTableFilter === "CRITICAL") {
    filtered = currentScheduleData.filter((x) => x.priority_tier === "CRITICAL" || x.priority_tier === "HIGH");
  } else if (currentTableFilter === "REEFER") {
    filtered = currentScheduleData.filter((x) => x.load_type === "Refrigerated");
  } else if (currentTableFilter === "OOH") {
    filtered = currentScheduleData.filter((x) => x.is_out_of_hours_before);
  } else if (currentTableFilter === "REROUTED") {
    filtered = currentScheduleData.filter((x) => x.action_code === "CITY_HUB_REROUTE");
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:#94a3b8; padding:22px;">Нет рейсов по выбранному фильтру</td></tr>`;
    return;
  }

  filtered.forEach((r) => {
    const tr = document.createElement("tr");
    const cargoIcon = r.load_type === "Refrigerated" ? "❄️ Рефрижератор" : "📦 Dry Van";
    const oohBadge = r.is_out_of_hours_before
      ? `<span class="kpi-badge badge-rose" style="margin-left:4px;" title="Прибытие вне часов работы склада">Вне часов</span>`
      : "";
    const surgeBadge = r.is_surge_injected
      ? `<span class="kpi-badge badge-amber" style="margin-left:4px;">Стресс-тест</span>`
      : "";

    let actionBadgeClass = "badge-blue";
    if (r.action_code === "CITY_HUB_REROUTE") actionBadgeClass = "badge-amber";
    else if (r.action_code === "PRIORITY_FAST_TRACK") actionBadgeClass = "badge-emerald";

    tr.innerHTML = `
      <td>
        <div style="font-weight:700; color:#f8fafc;">${r.load_id} ${surgeBadge}</div>
        <div style="font-size:0.73rem; color:#94a3b8;">${r.customer_name} • ${r.truck_info}</div>
      </td>
      <td>
        <div>${cargoIcon}</div>
        <div style="font-size:0.73rem; color:#94a3b8;">${r.booking_type} • ${(r.weight_lbs / 1000).toFixed(1)}k lbs (${r.pieces} пал.)</div>
      </td>
      <td>
        <span class="priority-pill prio-${r.priority_tier}">
          ${r.priority_score} • ${r.priority_label}
        </span>
      </td>
      <td>
        <div>${r.actual_arrival_time} ${oohBadge}</div>
        <div style="font-size:0.72rem; color:#64748b;">План: ${r.scheduled_time}</div>
      </td>
      <td>
        <div style="font-weight:700; color:#38bdf8;">${r.optimized_start_time} – ${r.optimized_end_time.split(" ")[1]}</div>
        <div style="font-size:0.72rem; color:#94a3b8;">Обработка: ${r.service_duration_min} мин</div>
      </td>
      <td>
        <div style="font-weight:600;">${r.assigned_dock}</div>
        <div style="font-size:0.72rem; color:#94a3b8;">${r.assigned_facility_name}</div>
      </td>
      <td>
        <span style="color:#94a3b8; text-decoration:line-through;">${r.detention_before_min} м</span>
        <span style="margin:0 4px;">→</span>
        <b style="color:#34d399;">${r.detention_after_min} м</b>
        <span class="kpi-badge badge-emerald" style="margin-left:5px;">-${r.detention_saved_min} м</span>
      </td>
      <td>
        <span class="kpi-badge ${actionBadgeClass}">${r.action_label}</span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function runMLPrediction() {
  const loadType = document.getElementById("ml-load-type").value;
  const bookingType = document.getElementById("ml-booking-type").value;
  const weightLbs = parseFloat(document.getElementById("ml-weight").value || "35000");
  const pieces = parseInt(document.getElementById("ml-pieces").value || "18", 10);
  const arrivalHour = parseInt(document.getElementById("ml-hour").value || "18", 10);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        facility_id: currentFacilityId,
        load_type: loadType,
        booking_type: bookingType,
        event_type: "Delivery",
        weight_lbs: weightLbs,
        pieces: pieces,
        arrival_hour: arrivalHour,
      }),
    });
    const d = await res.json();
    const box = document.getElementById("ml-prediction-result");
    const statusHtml = d.is_requested_slot_open
      ? `<span class="kpi-badge badge-emerald">Склад открыт (${d.operating_hours})</span>`
      : `<span class="kpi-badge badge-rose">⛔ Вне рабочих часов (${d.operating_hours}) — высокий риск простоя!</span>`;

    box.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
          <b>Прогноз ML для ${d.facility_id} (${d.facility_name}, ${d.city}):</b> ${statusHtml}
          <div style="margin-top:4px; color:#cbd5e1;">
            При прибытии в <b>${String(d.requested_hour).padStart(2, "0")}:00</b> ожидаемый простой составит <b style="color:#fb7185;">${d.predicted_detention_min} мин</b> (длительность разгрузки у дока: <b>${d.estimated_service_duration_min} мин</b>).
          </div>
        </div>
        <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35); padding:8px 14px; border-radius:8px;">
          🎯 <b>Рекомендуемый AI слот: <span style="color:#34d399;">${d.recommended_best_time}</span></b>
          <div style="font-size:0.76rem; color:#94a3b8;">Ожидание снизится до <b>${d.best_slot_detention_min} мин</b> (экономия <b>-${d.estimated_savings_min} мин</b>)</div>
        </div>
      </div>
    `;
  } catch (err) {
    console.error("ML prediction failed:", err);
  }
}

/* AI Agent Chat Functions */
function appendAgentGreeting() {
  const container = document.getElementById("chat-messages");
  container.innerHTML = "";
  appendChatBubble(
    "agent",
    `### 👋 Привет! Я AI-Агент регулирования очередей
Я подключен к вашему логистическому датасету (**50 складов**, **170 820 событий доставки**, **85 410 грузов**).

**Что я делаю в реальном времени:**
- Устраняю простои фур (\`detention_minutes\`), синхронизируя тайм-слоты с часами работы складов (\`8AM-5PM\`, \`7AM-7PM\`, \`6AM-10PM\`, \`24/7\`).
- Выделяю **Fast-Track** приоритет для рефрижераторов (\`Refrigerated\`) и контрактов \`Dedicated\`.
- При перегрузке терминала перенаправляю поток в открытые партнерские хабы того же города.

Выберите быстрый запрос выше или напишите команду, например: *«Оптимизируй склад в Шарлотт FAC00034»*!`
  );
}

function formatMarkdown(md) {
  let html = md
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/^#### (.*$)/gim, "<h4>$1</h4>")
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
    .replace(/\*(.*?)\*/g, "<i>$1</i>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n- (.*)/g, "<br/>• $1")
    .replace(/\n(\d+)\. (.*)/g, "<br/><b>$1.</b> $2")
    .replace(/\n\n/g, "<br/><br/>");
  return html;
}

function appendChatBubble(role, text, toolsExecuted = []) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;

  let toolsHtml = "";
  if (toolsExecuted && toolsExecuted.length > 0) {
    const pills = toolsExecuted
      .map((t) => `<div class="tool-pill">⚙️ <b>${t.tool}</b>: ${t.summary}</div>`)
      .join("");
    toolsHtml = `<div class="tool-execution-list">${pills}</div>`;
  }

  div.innerHTML = toolsHtml + (role === "agent" ? formatMarkdown(text) : text);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

async function sendQuickPrompt(text) {
  document.getElementById("chat-input").value = text;
  await sendChatMessage();
}

async function sendChatMessage() {
  const input = document.getElementById("chat-input");
  const msg = input.value.trim();
  if (!msg) return;

  input.value = "";
  appendChatBubble("user", msg);

  const surgeExtra = parseInt(document.getElementById("surge-select").value || "0", 10);

  try {
    const res = await fetch("/api/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        facility_id: currentFacilityId,
        surge_extra_trucks: surgeExtra,
      }),
    });
    const data = await res.json();

    document.getElementById("agent-mode-pill").textContent = `🤖 ${data.mode}`;
    appendChatBubble("agent", data.reply, data.tools_executed);

    // If the agent switched or optimized a specific facility, sync the dashboard view!
    if (data.facility_id && data.facility_id !== currentFacilityId) {
      currentFacilityId = data.facility_id;
      document.getElementById("facility-select").value = currentFacilityId;
      if (data.optimization_snapshot) {
        renderOptimizationResult(data.optimization_snapshot);
      }
      runMLPrediction();
    }
  } catch (err) {
    appendChatBubble("agent", "⚠️ Ошибка связи с сервером оптимизации.");
  }
}

/* Modal Config */
function openConfigModal() {
  document.getElementById("config-modal").classList.add("open");
}

function closeConfigModal() {
  document.getElementById("config-modal").classList.remove("open");
}

async function saveAgentConfig() {
  const provider = document.getElementById("cfg-provider").value;
  const apiKey = document.getElementById("cfg-apikey").value;
  const model = document.getElementById("cfg-model").value;

  const res = await fetch("/api/agent/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider,
      api_key: apiKey,
      model,
    }),
  });
  const data = await res.json();
  document.getElementById("agent-mode-pill").textContent =
    data.provider === "autonomous"
      ? "🤖 Режим: Автономное ML-ядро"
      : `🤖 Режим: ${data.provider.toUpperCase()} (${data.model})`;
  closeConfigModal();
}
