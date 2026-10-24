import json
import os
import re
import sys
from pathlib import Path
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data_loader import LogisticsDataStore
from backend.ml_predictor import DetentionMLPredictor
from backend.queue_optimizer import QueueOptimizerEngine


class HybridLogisticsAgent:
    """
    Hybrid AI Agent for Logistics Queue & Dock Regulation.
    Works autonomously out-of-the-box using deterministic tool execution + ML reasoning,
    and optionally connects to OpenAI, Google Gemini, or local Ollama LLMs.
    """

    def __init__(
        self,
        store: LogisticsDataStore,
        predictor: DetentionMLPredictor,
        optimizer: QueueOptimizerEngine,
    ):
        self.store = store
        self.predictor = predictor
        self.optimizer = optimizer

        self.llm_config: Dict[str, str] = {
            "provider": "autonomous",  # 'autonomous' | 'gemini' | 'openai' | 'ollama'
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            "model": "gemini-2.5-flash",
            "ollama_url": os.environ.get("OLLAMA_URL", ""),
        }
        if os.environ.get("GEMINI_API_KEY"):
            self.llm_config["provider"] = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            self.llm_config["provider"] = "openai"
            self.llm_config["model"] = "gpt-4o-mini"

    def update_config(
        self,
        provider: str,
        api_key: str = "",
        model: str = "",
        ollama_url: str = "",
    ) -> Dict[str, Any]:
        self.llm_config["provider"] = provider.lower().strip()
        if api_key is not None:
            self.llm_config["api_key"] = api_key.strip()
        if model:
            self.llm_config["model"] = model.strip()
        if ollama_url:
            self.llm_config["ollama_url"] = ollama_url.strip()
        return {
            "provider": self.llm_config["provider"],
            "model": self.llm_config["model"],
            "has_api_key": bool(self.llm_config["api_key"]),
            "ollama_url": self.llm_config["ollama_url"],
        }

    def _detect_facility_from_text(self, text: str, current_facility_id: Optional[str]) -> str:
        t_up = text.upper()
        # Direct FAC000XX match
        m = re.search(r"FAC\d{5}", t_up)
        if m and m.group(0) in self.store.facility_dict:
            return m.group(0)

        # City name match (English & Russian)
        city_aliases = {
            "ATLANTA": "FAC00012",
            "АТЛАНТ": "FAC00012",
            "PHILADELPHIA": "FAC00008",
            "ФИЛАДЕЛЬФ": "FAC00008",
            "CHARLOTTE": "FAC00034",
            "ШАРЛОТТ": "FAC00034",
            "INDIANAPOLIS": "FAC00005",
            "ИНДИАНАПОЛИС": "FAC00005",
            "CHICAGO": "FAC00004",
            "ЧИКАГО": "FAC00004",
            "HOUSTON": "FAC00001",
            "ХЬЮСТОН": "FAC00001",
            "DETROIT": "FAC00007",
            "ДЕТРОЙТ": "FAC00007",
            "NASHVILLE": "FAC00009",
            "НЭШВИЛЛ": "FAC00009",
            "MIAMI": "FAC00029",
            "МАЙАМИ": "FAC00029",
            "DALLAS": "FAC00018",
            "ДАЛЛАС": "FAC00018",
            "LOS ANGELES": "FAC00031",
            "ЛОС-АНДЖЕЛЕС": "FAC00031",
            "DENVER": "FAC00024",
            "ДЕНВЕР": "FAC00024",
            "PHOENIX": "FAC00015",
            "ФЕНИКС": "FAC00015",
            "LAS VEGAS": "FAC00021",
            "ЛАС-ВЕГАС": "FAC00021",
            "NEW YORK": "FAC00022",
            "НЬЮ-ЙОРК": "FAC00022",
            "PORTLAND": "FAC00023",
            "ПОРТЛЕНД": "FAC00023",
        }
        for alias, fid in city_aliases.items():
            if alias in t_up:
                return fid

        return current_facility_id or "FAC00012"

    def process_message(
        self,
        message: str,
        current_facility_id: Optional[str] = "FAC00012",
        surge_extra_trucks: int = 0,
    ) -> Dict[str, Any]:
        msg_lower = message.lower()
        target_fac_id = self._detect_facility_from_text(message, current_facility_id)
        tools_executed: List[Dict[str, Any]] = []

        # 1. Execute relevant tools to gather real dataset telemetry
        # Always pull facility optimization snapshot for target facility
        opt_result = self.optimizer.optimize_facility_queue(
            facility_id=target_fac_id,
            allow_city_reroute=True,
            surge_extra_trucks=surge_extra_trucks,
        )
        tools_executed.append(
            {
                "tool": "optimize_facility_queue",
                "args": {"facility_id": target_fac_id, "surge_extra_trucks": surge_extra_trucks},
                "summary": f"Расчет расписания доков {target_fac_id}: снижение простоя на {opt_result['kpis']['detention_reduction_pct']}%",
            }
        )

        # Check top network bottlenecks
        top_bottlenecks = sorted(
            self.store.facility_stats.values(),
            key=lambda x: (x["out_of_hours_rate"], x["avg_detention"]),
            reverse=True,
        )[:5]
        tools_executed.append(
            {
                "tool": "scan_network_bottlenecks",
                "args": {"top_k": 5},
                "summary": f"Проанализировано 50 терминалов и 170 820 событий доставки",
            }
        )

        # Run ML prediction for target facility
        ml_forecast = self.predictor.predict_detention_and_best_slot(
            facility_id=target_fac_id,
            load_type="Refrigerated",
            booking_type="Dedicated",
            arrival_hour=18,
        )
        tools_executed.append(
            {
                "tool": "predict_detention_and_best_slot",
                "args": {"facility_id": target_fac_id, "load_type": "Refrigerated"},
                "summary": f"ML-прогноз: лучший слот {ml_forecast['recommended_best_time']} (ожидание {ml_forecast['best_slot_detention_min']} мин)",
            }
        )

        # Build structured response in autonomous mode (or enrich with external LLM if configured)
        autonomous_reply = self._build_analytical_response(
            message=message,
            msg_lower=msg_lower,
            opt_result=opt_result,
            top_bottlenecks=top_bottlenecks,
            ml_forecast=ml_forecast,
        )

        provider = self.llm_config.get("provider", "autonomous")
        api_key = self.llm_config.get("api_key", "")

        if provider in ("gemini", "openai", "ollama") and (api_key or provider == "ollama"):
            llm_reply = self._call_external_llm(
                user_message=message,
                opt_result=opt_result,
                top_bottlenecks=top_bottlenecks,
                ml_forecast=ml_forecast,
            )
            if llm_reply:
                return {
                    "reply": llm_reply,
                    "mode": f"Гибридный ({provider.upper()} + ML Optimizer)",
                    "facility_id": target_fac_id,
                    "tools_executed": tools_executed,
                    "optimization_snapshot": opt_result,
                }

        return {
            "reply": autonomous_reply,
            "mode": "Автономное ML-ядро + Эвристический Оптимизатор",
            "facility_id": target_fac_id,
            "tools_executed": tools_executed,
            "optimization_snapshot": opt_result,
        }

    def _build_analytical_response(
        self,
        message: str,
        msg_lower: str,
        opt_result: Dict[str, Any],
        top_bottlenecks: List[Dict[str, Any]],
        ml_forecast: Dict[str, Any],
    ) -> str:
        fac = opt_result["facility"]
        kpis = opt_result["kpis"]
        fid = fac["facility_id"]
        fname = fac["facility_name"]
        city = f"{fac['city']}, {fac['state']}"

        # Intent 1: Network-wide bottlenecks / "где очереди" / "узкие места"
        if any(w in msg_lower for w in ["узк", "где", "сеть", "все", "рейтинг", "топ", "проблем", "bottleneck"]):
            lines = [
                f"### 📊 Диагностика узких мест логистической сети (50 складов, 170 820 рейсов)",
                "",
                "Анализ датасета выявил главную системную причину очередей: **несогласованность слотов прибытия с часами работы терминалов (`operating_hours`) и одновременные пики подачи фур**.",
                "",
                "**Топ-5 самых перегруженных терминалов сети:**",
            ]
            for idx, b in enumerate(top_bottlenecks, 1):
                lines.append(
                    f"{idx}. **{b['facility_id']} — {b['facility_name']}** ({b['city']}, {b['state']}) | "
                    f"Режим: `{b['operating_hours']}` | Доков: `{b['dock_doors']}` | "
                    f"Ср. простой: **{b['avg_detention']} мин** | Прибытие вне рабочих часов: **{b['out_of_hours_rate']}%**"
                )
            lines.extend(
                [
                    "",
                    f"💡 **Рекомендация агента для текущего терминала `{fid}` ({fname}):**",
                    f"- Применение алгоритма Time-Slot & Priority очереди снижает среднее время простоя с **{kpis['avg_detention_before_min']} мин** до **{kpis['avg_detention_after_min']} мин** (**-{kpis['detention_reduction_pct']}%**).",
                    f"- Экономия по выбранному окну составляет **${kpis['estimated_cost_saved_usd']}** и **{kpis['fuel_saved_gallons']} галлонов** дизельного топлива на холостом ходу.",
                ]
            )
            return "\n".join(lines)

        # Intent 2: Cold chain / Priority / Refrigerated ("рефрижератор", "приоритет", "скоропорт")
        if any(w in msg_lower for w in ["рефриж", "приорит", "скоропорт", "холод", "контракт", "sla", "правил"]):
            reefer_items = [
                s for s in opt_result["schedule"] if s["load_type"] == "Refrigerated"
            ][:4]
            lines = [
                f"### ❄️ Динамическая приоритизация грузов на терминале `{fid}` ({fname})",
                "",
                "Агент ранжирует каждую фуру в очереди по **индексу приоритета (0–99 баллов)**:",
                "- **Тип груза**: `Refrigerated` (рефрижераторы) получают **+34 балла** для защиты температурной цепи.",
                "- **Уровень контракта**: `Dedicated` (**+24**), `Contract` (**+16**), `Spot` (**+6**).",
                "- **Риск срыва SLA и накопленный простой**: до **+26 баллов** при задержке рейса.",
                "",
                f"**Результаты приоритизации на `{fid}` ({city}):**",
                f"- Переведено на ускоренную разгрузку (**Fast-Track**): **{kpis['fast_tracked_trucks']} фур**",
                f"- Защищено температурных грузов (`Refrigerated`): **{kpis['reefer_loads_protected']} рейсов**",
            ]
            if reefer_items:
                lines.append("\n**Примеры приоритетных рейсов в расписании:**")
                for item in reefer_items:
                    lines.append(
                        f"- `{item['load_id']}` ({item['customer_name']}) | Приоритет: **{item['priority_score']}** | "
                        f"Назначен: **{item['assigned_dock']}** (`{item['optimized_start_time']}`) | "
                        f"Простой: `{item['detention_before_min']} мин` → **`{item['detention_after_min']} мин`**"
                    )
            return "\n".join(lines)

        # Intent 3: ML prediction / best slot ("прогноз", "слот", "когда", "ml", "модель")
        if any(w in msg_lower for w in ["прогноз", "когда", "лучш", "слот", "ml", "модел", "предсказ"]):
            return "\n".join(
                [
                    f"### 🧠 ML-прогноз загрузки и выбор лучшего окна прибытия (`{fid}` — {fname})",
                    "",
                    f"Модель `HistGradientBoostingRegressor` (обучена на исторических данных `delivery_events.csv` + `loads.csv`) проанализировала профиль терминала в г. **{city}** (режим работы: `{fac['operating_hours']}`, доков: `{fac['dock_doors']}`):",
                    "",
                    f"- **Оптимальное окно подачи под разгрузку**: **`{ml_forecast['recommended_best_time']}`** (прогнозируемое ожидание всего **{ml_forecast['best_slot_detention_min']} мин**).",
                    f"- **Средняя длительность обработки у дока**: **{ml_forecast['estimated_service_duration_min']} мин** (с учетом веса и количества паллет).",
                    f"- При подаче фуры в вечерний пик / нерабочее окно (`18:00`) ожидание без AI-регулирования составило бы **{ml_forecast['predicted_detention_min']} мин** (потенциальная экономия: **{ml_forecast['estimated_savings_min']} мин** на рейс).",
                ]
            )

        # Default / Optimization report for the facility
        alt_hubs_str = (
            ", ".join(
                [
                    f"`{h['facility_id']}` ({h['facility_name']}, {h['operating_hours']})"
                    for h in fac["alternative_city_hubs"]
                ]
            )
            if fac["alternative_city_hubs"]
            else "нет соседних хабов в этом городе"
        )

        top_saved = sorted(
            opt_result["schedule"], key=lambda x: x["detention_saved_min"], reverse=True
        )[:3]

        lines = [
            f"### ⚡ План AI-регулирования очереди: `{fid}` — {fname} ({city})",
            "",
            f"**Параметры терминала:** Режим `{fac['operating_hours']}` | Всего доков: `{fac['dock_doors']}` (активных бригад/секций: `{fac['active_bays']}`) | Партнерские хабы в городе: {alt_hubs_str}.",
            "",
            "#### 1. Ключевые показатели оптимизации (До → После AI):",
            f"- **Среднее время простоя (`detention_minutes`)**: `{kpis['avg_detention_before_min']} мин` → **`{kpis['avg_detention_after_min']} мин`** (**сокращение на {kpis['detention_reduction_pct']}%**)",
            f"- **Суммарно сэкономлено времени**: **{kpis['total_detention_saved_min']} мин** ({round(kpis['total_detention_saved_min']/60.0, 1)} ч простоя фур)",
            f"- **Устранено прибытий в закрытые часы склада**: **{kpis['out_of_hours_arrivals_caught']} рейсов**",
            f"- **Балансировка в соседние открытые хабы города**: **{kpis['rerouted_to_city_hubs']} фур** | **Fast-Track приоритет**: **{kpis['fast_tracked_trucks']} фур**",
            f"- **Прямой финансовый эффект**: сохранено **${kpis['estimated_cost_saved_usd']}** штрафов за простой и **{kpis['fuel_saved_gallons']} гал.** топлива (снижение выбросов на **{kpis['co2_saved_kg']} кг CO₂**).",
            "",
            "#### 2. Конкретные перестановки в очереди (Топ-3 по эффекту):",
        ]
        for item in top_saved:
            lines.append(
                f"- 🚛 **{item['load_id']}** ({item['load_type']}, `{item['booking_type']}`, клиент *{item['customer_name']}*): "
                f"**{item['action_label']}** → назначен `{item['assigned_dock']}` на `{item['optimized_start_time']}`. "
                f"Простой снижен с `{item['detention_before_min']} мин` до **`{item['detention_after_min']} мин`** (-{item['detention_saved_min']} мин)."
            )

        return "\n".join(lines)

    def _call_external_llm(
        self,
        user_message: str,
        opt_result: Dict[str, Any],
        top_bottlenecks: List[Dict[str, Any]],
        ml_forecast: Dict[str, Any],
    ) -> Optional[str]:
        provider = self.llm_config["provider"]
        api_key = self.llm_config["api_key"]
        model = self.llm_config["model"]

        system_prompt = (
            "Ты — интеллектуальный AI-диспетчер по управлению очередями и расписанием доков (Time-Slot & Yard Management) в грузоперевозках. "
            "Отвечай четко, профессионально, на русском языке, используя реальные цифры из переданного контекста оптимизации."
        )
        context_payload = {
            "facility": opt_result["facility"],
            "optimization_kpis": opt_result["kpis"],
            "sample_optimized_trucks": opt_result["schedule"][:5],
            "ml_slot_forecast": {
                "recommended_best_time": ml_forecast["recommended_best_time"],
                "best_slot_detention_min": ml_forecast["best_slot_detention_min"],
                "estimated_service_duration_min": ml_forecast["estimated_service_duration_min"],
            },
            "top_network_bottlenecks": [
                {
                    "facility_id": b["facility_id"],
                    "facility_name": b["facility_name"],
                    "city": b["city"],
                    "operating_hours": b["operating_hours"],
                    "avg_detention": b["avg_detention"],
                    "out_of_hours_rate": b["out_of_hours_rate"],
                }
                for b in top_bottlenecks
            ],
        }
        full_prompt = (
            f"{system_prompt}\n\n"
            f"ДАННЫЕ ОПТИМИЗАТОРА И ML-МОДЕЛИ:\n{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n\n"
            f"ЗАПРОС ДИСПЕТЧЕРА: {user_message}"
        )

        try:
            if provider == "gemini":
                gemini_host = "generativelanguage.googleapis.com"
                gemini_model = model or "gemini-2.5-flash"
                url = f"https://{gemini_host}/v1beta/models/{gemini_model}:generateContent?key={api_key}"
                body = json.dumps(
                    {"contents": [{"parts": [{"text": full_prompt}]}]}
                ).encode("utf-8")
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return str(data["candidates"][0]["content"]["parts"][0]["text"])

            elif provider == "openai":
                openai_host = "api.openai.com"
                url = f"https://{openai_host}/v1/chat/completions"
                body = json.dumps(
                    {
                        "model": model or "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": full_prompt},
                        ],
                        "temperature": 0.3,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return str(data["choices"][0]["message"]["content"])

            elif provider == "ollama":
                ollama_endpoint = self.llm_config.get("ollama_url", "").strip()
                if not ollama_endpoint:
                    return None
                body = json.dumps(
                    {
                        "model": model or "llama3",
                        "prompt": full_prompt,
                        "stream": False,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    ollama_endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response")
        except Exception as e:
            print(f"[HybridAgent] External LLM call failed ({e}), falling back to autonomous engine.")
            return None
        return None
