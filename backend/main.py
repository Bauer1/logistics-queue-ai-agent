from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data_loader import LogisticsDataStore, DEFAULT_DATASET_PATH
from backend.ml_predictor import DetentionMLPredictor
from backend.queue_optimizer import QueueOptimizerEngine
from backend.ai_agent import HybridLogisticsAgent

# Initialize Data Store, ML Predictor, Queue Optimizer, and Hybrid Agent
store = LogisticsDataStore(dataset_path=DEFAULT_DATASET_PATH)
predictor = DetentionMLPredictor(store=store)
optimizer = QueueOptimizerEngine(store=store, predictor=predictor)
agent = HybridLogisticsAgent(store=store, predictor=predictor, optimizer=optimizer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_all()
    predictor.train(sample_size=45000)
    yield


app = FastAPI(
    title="Logistics Queue & Dock AI-Agent Control Tower",
    description="AI-агент регулирования очередей, оптимизации расписания доков (Time-Slot Management) и прогнозирования простоев в логистической цепочке.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizeRequest(BaseModel):
    facility_id: str = "FAC00012"
    target_date: Optional[str] = None
    window_days: int = 3
    active_bays_override: Optional[int] = None
    allow_city_reroute: bool = True
    surge_extra_trucks: int = 0


class PredictRequest(BaseModel):
    facility_id: str = "FAC00012"
    load_type: str = "Refrigerated"
    booking_type: str = "Dedicated"
    event_type: str = "Delivery"
    weight_lbs: float = 34000
    pieces: int = 18
    arrival_hour: int = 18
    arrival_delay_min: float = 45.0


class ChatRequest(BaseModel):
    message: str
    facility_id: Optional[str] = "FAC00012"
    surge_extra_trucks: int = 0


class AgentConfigRequest(BaseModel):
    provider: str = "autonomous"
    api_key: Optional[str] = ""
    model: Optional[str] = "gemini-2.5-flash"
    ollama_url: Optional[str] = ""


@app.get("/api/overview")
def get_network_overview():
    fac_list = list(store.facility_stats.values())
    fac_list.sort(key=lambda x: (x["out_of_hours_rate"], x["avg_detention"]), reverse=True)

    total_events = int(len(store.events))
    avg_detention = round(float(store.events["detention_minutes"].mean()), 1)
    on_time_pct = round(float(store.events["on_time_flag"].mean()) * 100.0, 1)
    total_detention_hours = int(round(store.events["detention_minutes"].sum() / 60.0))

    return {
        "network_kpis": {
            "total_facilities": len(fac_list),
            "total_delivery_events": total_events,
            "total_loads": int(len(store.loads)),
            "avg_network_detention_min": avg_detention,
            "network_on_time_rate_pct": on_time_pct,
            "total_historical_detention_hours": total_detention_hours,
            "ml_model_metrics": predictor.metrics,
        },
        "facilities": fac_list,
        "top_congested_days": store.top_congested_days[:12],
        "agent_config": {
            "provider": agent.llm_config["provider"],
            "model": agent.llm_config["model"],
            "has_api_key": bool(agent.llm_config["api_key"]),
        },
    }


@app.post("/api/optimize")
def optimize_facility(req: OptimizeRequest):
    try:
        res = optimizer.optimize_facility_queue(
            facility_id=req.facility_id,
            target_date=req.target_date,
            window_days=req.window_days,
            active_bays_override=req.active_bays_override,
            allow_city_reroute=req.allow_city_reroute,
            surge_extra_trucks=req.surge_extra_trucks,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict")
def predict_slot(req: PredictRequest):
    return predictor.predict_detention_and_best_slot(
        facility_id=req.facility_id,
        load_type=req.load_type,
        booking_type=req.booking_type,
        event_type=req.event_type,
        weight_lbs=req.weight_lbs,
        pieces=req.pieces,
        arrival_hour=req.arrival_hour,
        arrival_delay_min=req.arrival_delay_min,
    )


@app.post("/api/agent/chat")
def chat_with_agent(req: ChatRequest):
    return agent.process_message(
        message=req.message,
        current_facility_id=req.facility_id,
        surge_extra_trucks=req.surge_extra_trucks,
    )


@app.post("/api/agent/config")
def update_agent_config(req: AgentConfigRequest):
    return agent.update_config(
        provider=req.provider,
        api_key=req.api_key or "",
        model=req.model or "",
        ollama_url=req.ollama_url or "",
    )


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Logistics Queue AI-Agent API running"}


@app.get("/api/guide.pdf")
def download_pdf_guide():
    pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Руководство_AI_Агент_Логистика.pdf",
    )
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="Руководство_AI_Агент_Логистика.pdf",
        )
    raise HTTPException(status_code=404, detail="PDF file not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


