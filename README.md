# 🚛 Logistics Queue & Dock AI-Agent (Control Tower)

Интеллектуальный AI-агент и веб-дашборд для регулирования очередей, оптимизации расписания доков (**Time-Slot Management**) и прогнозирования простоев (`detention_minutes`) в логистической цепочке на базе вашего датасета (`c:\Users\zhuma\Downloads\archive`).

## 🚀 Быстрый запуск

```powershell
python run.py
```
После запуска откройте в браузере: **http://127.0.0.1:8000**

## 🧠 Архитектура проекта

1. **[`backend/data_loader.py`](backend/data_loader.py)** — Загрузчик и индексатор реального датасета (`facilities.csv`, `delivery_events.csv`, `loads.csv`, `trips.csv`, `drivers.csv`, `trucks.csv`, `customers.csv`, `routes.csv` — 50 складов, 170 820 событий доставки, 85 410 грузов). Выявляет системные нарушения рабочих окон (`operating_hours`: `8AM-5PM`, `7AM-7PM`, `6AM-10PM`, `24/7`) и кластеры складов по городам.
2. **[`backend/ml_predictor.py`](backend/ml_predictor.py)** — ML-модель на базе `HistGradientBoostingRegressor` (`scikit-learn`), обученная на исторических рейсах. Прогнозирует длительность обработки фуры у дока (`service_duration_min`), ожидаемый простой (`detention_minutes`) для каждого часа суток и рекомендует оптимальное окно подачи под разгрузку.
3. **[`backend/queue_optimizer.py`](backend/queue_optimizer.py)** — Ядро динамической регуляции очереди и распределения доков:
   - **Индекс приоритета (0–99 баллов)**: учитывает тип груза (`Refrigerated` +34 балла для защиты температурной цепи), тип контракта (`Dedicated` +24, `Contract` +16, `Spot` +6), риск срыва SLA и вес/объем.
   - **Синхронизация с часами работы (`operating_hours`) и активными бригадами доков (`Dock #1..N`)**: устраняет простои у закрытых ворот и сглаживает пиковые подачи.
   - **Межскладская балансировка (Cross-Dock City Overflow)**: при приходе критического груза в нерабочие часы перенаправляет фуру в открытый партнерский хаб того же города (`24/7` или `6AM-10PM`).
4. **[`backend/ai_agent.py`](backend/ai_agent.py)** — Гибридный AI-диспетчер с вызовом инструментов (`optimize_facility_queue`, `scan_network_bottlenecks`, `predict_detention_and_best_slot`). Работает полностью автономно без ключей и поддерживает подключение внешних LLM (`Google Gemini`, `OpenAI`, `Ollama`).
5. **[`frontend/`](frontend/index.html)** — Интерактивный веб-интерфейс диспетчерской башни (Control Tower) с графиками До/После AI, картой загрузки доков, стресс-тестированием очередей, ML-калькулятором и чатом с AI-агентом.
