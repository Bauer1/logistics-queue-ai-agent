import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Dict, Any, List
from backend.data_loader import LogisticsDataStore, parse_operating_hours


class DetentionMLPredictor:
    """
    Machine Learning model trained on the logistics dataset (delivery_events + loads + facilities)
    to predict expected detention minutes, dock handling duration, and optimal arrival time slots.
    """

    def __init__(self, store: LogisticsDataStore):
        self.store = store
        self.model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
        )
        self.is_trained = False
        self.feature_names = [
            "dock_doors",
            "open_hour",
            "close_hour",
            "is_24_7",
            "facility_type_code",
            "event_type_code",
            "load_type_code",
            "booking_type_code",
            "weight_lbs",
            "pieces",
            "arrival_hour",
            "day_of_week",
            "arrival_delay_min",
            "is_out_of_hours",
            "hourly_facility_load",
        ]
        self.facility_type_map = {
            "Cross-Dock": 0,
            "Distribution Center": 1,
            "Terminal": 2,
            "Warehouse": 3,
        }
        self.event_type_map = {"Pickup": 0, "Delivery": 1}
        self.load_type_map = {"Dry Van": 0, "Refrigerated": 1}
        self.booking_type_map = {"Spot": 0, "Contract": 1, "Dedicated": 2}
        self.metrics: Dict[str, Any] = {}

    def train(self, sample_size: int = 45000):
        print("[MLPredictor] Training HistGradientBoosting detention & slot predictor...")
        ev = self.store.events
        if len(ev) > sample_size:
            ev_sample = ev.sample(n=sample_size, random_state=42).copy()
        else:
            ev_sample = ev.copy()

        df = ev_sample.merge(
            self.store.facilities[
                [
                    "facility_id",
                    "facility_type",
                    "dock_doors",
                    "open_hour",
                    "close_hour",
                    "is_24_7",
                ]
            ],
            on="facility_id",
            how="left",
        ).merge(
            self.store.loads[
                ["load_id", "load_type", "booking_type", "weight_lbs", "pieces"]
            ],
            on="load_id",
            how="left",
        )

        df["facility_type_code"] = df["facility_type"].map(self.facility_type_map).fillna(0)
        df["event_type_code"] = df["event_type"].map(self.event_type_map).fillna(0)
        df["load_type_code"] = df["load_type"].map(self.load_type_map).fillna(0)
        df["booking_type_code"] = df["booking_type"].map(self.booking_type_map).fillna(0)
        df["arrival_hour"] = df["act_hour"]
        df["day_of_week"] = df["act_dt"].dt.dayofweek
        df["is_out_of_hours"] = (
            (~df["is_24_7"])
            & ((df["arrival_hour"] < df["open_hour"]) | (df["arrival_hour"] >= df["close_hour"]))
        ).astype(int)

        # Approximate hourly load pressure at facility
        hourly_counts = (
            df.groupby(["facility_id", "arrival_hour"])["event_id"]
            .transform("count")
            .fillna(1)
        )
        df["hourly_facility_load"] = hourly_counts / df["dock_doors"].clip(lower=1)

        # Target combines historical detention + operational bottleneck penalty when out of hours or heavy cargo
        # This ensures the ML model learns both historical detention patterns and physical dock constraints
        raw_y = df["detention_minutes"].fillna(60.0).values
        operational_penalty = (
            df["is_out_of_hours"].values * 65.0
            + (df["weight_lbs"].fillna(25000).values / 45000.0) * 18.0
            + (df["pieces"].fillna(12).values / 30.0) * 12.0
            + np.clip(df["arrival_delay_min"].fillna(0).values, 0, 180) * 0.15
        )
        y = np.clip(0.65 * raw_y + 0.35 * operational_penalty, 5.0, 300.0)

        X = df[self.feature_names].fillna(0).values
        self.model.fit(X, y)
        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        self.is_trained = True
        self.metrics = {
            "model_type": "HistGradientBoostingRegressor (scikit-learn)",
            "training_samples": int(len(df)),
            "mae_minutes": round(mae, 2),
            "features_used": len(self.feature_names),
        }
        print(f"[MLPredictor] Trained on {len(df)} events (MAE: {mae:.2f} min).")

    def estimate_service_duration_minutes(
        self,
        weight_lbs: float,
        pieces: int,
        load_type: str,
        event_type: str,
    ) -> int:
        """
        Calculates physical dock door occupation time (loading/unloading duration in minutes)
        based on cargo weight, pallet pieces, refrigeration checks, and event type.
        """
        base = 32.0 if event_type == "Pickup" else 38.0
        weight_factor = (float(weight_lbs or 25000) / 10000.0) * 6.5
        pieces_factor = float(pieces or 12) * 1.15
        reefer_inspection = 12.0 if load_type == "Refrigerated" else 0.0
        total = int(round(base + weight_factor + pieces_factor + reefer_inspection))
        return max(25, min(120, total))

    def predict_detention_and_best_slot(
        self,
        facility_id: str,
        load_type: str = "Refrigerated",
        booking_type: str = "Dedicated",
        event_type: str = "Delivery",
        weight_lbs: float = 34000,
        pieces: int = 18,
        arrival_hour: int = 18,
        arrival_delay_min: float = 45.0,
        day_of_week: int = 2,
    ) -> Dict[str, Any]:
        fac = self.store.facility_dict.get(facility_id)
        if not fac:
            fac = list(self.store.facility_dict.values())[0]
            facility_id = fac["facility_id"]

        open_h, close_h, is_24_7 = parse_operating_hours(fac.get("operating_hours", "24/7"))
        doors = int(fac.get("dock_doors", 50))
        fac_type_code = self.facility_type_map.get(fac.get("facility_type", "Cross-Dock"), 0)
        ev_code = self.event_type_map.get(event_type, 1)
        lt_code = self.load_type_map.get(load_type, 1)
        bt_code = self.booking_type_map.get(booking_type, 1)

        hourly_curve: List[Dict[str, Any]] = []
        for h in range(24):
            is_ooh = 0 if is_24_7 else int(h < open_h or h >= close_h)
            # Peak hours between 7-11 and 15-18 have higher load pressure
            peak_pressure = 1.8 if (7 <= h <= 11 or 15 <= h <= 18) else (0.9 if 6 <= h <= 20 else 0.4)
            hourly_load = peak_pressure * (40.0 / max(15, doors))
            delay_for_h = arrival_delay_min if h == arrival_hour else 0.0

            feat = np.array(
                [
                    [
                        doors,
                        open_h,
                        close_h,
                        int(is_24_7),
                        fac_type_code,
                        ev_code,
                        lt_code,
                        bt_code,
                        weight_lbs,
                        pieces,
                        h,
                        day_of_week,
                        delay_for_h,
                        is_ooh,
                        hourly_load,
                    ]
                ]
            )
            pred_det = float(self.model.predict(feat)[0])
            # Add explicit closed-gate wait time if arriving when warehouse is closed
            gate_wait = 0
            if is_ooh:
                if h < open_h:
                    gate_wait = (open_h - h) * 60
                else:
                    gate_wait = ((24 - h) + open_h) * 60
            effective_detention = round(pred_det + gate_wait * 0.45, 1)
            hourly_curve.append(
                {
                    "hour": h,
                    "time_label": f"{h:02d}:00",
                    "is_open": not bool(is_ooh),
                    "predicted_detention_min": effective_detention,
                    "gate_wait_min": gate_wait,
                }
            )

        current_slot_info = hourly_curve[arrival_hour % 24]
        open_slots = [s for s in hourly_curve if s["is_open"]]
        best_slot = min(open_slots, key=lambda x: x["predicted_detention_min"]) if open_slots else hourly_curve[0]
        service_duration = self.estimate_service_duration_minutes(
            weight_lbs, pieces, load_type, event_type
        )

        return {
            "facility_id": facility_id,
            "facility_name": fac.get("facility_name"),
            "city": fac.get("city"),
            "state": fac.get("state"),
            "operating_hours": fac.get("operating_hours"),
            "dock_doors": doors,
            "requested_hour": arrival_hour,
            "predicted_detention_min": current_slot_info["predicted_detention_min"],
            "is_requested_slot_open": current_slot_info["is_open"],
            "recommended_best_hour": best_slot["hour"],
            "recommended_best_time": best_slot["time_label"],
            "best_slot_detention_min": best_slot["predicted_detention_min"],
            "estimated_savings_min": round(
                max(0.0, current_slot_info["predicted_detention_min"] - best_slot["predicted_detention_min"]),
                1,
            ),
            "estimated_service_duration_min": service_duration,
            "gamma_sla_analytics": self.compute_gamma_analytics(facility_id, sla_limit_min=120.0),
            "hourly_detention_forecast": hourly_curve,
        }

    def compute_gamma_analytics(self, facility_id: str, sla_limit_min: float = 120.0) -> Dict[str, Any]:
        """
        Fits a Gamma(alpha, theta) distribution via Method of Moments to positive detention observations
        for the given facility and computes P90, P95, and P(T > sla_limit_min) using scipy.stats.gamma.
        """
        from scipy.stats import gamma as sp_gamma

        fac_ev = self.store.events[self.store.events["facility_id"] == facility_id]
        pos_det = fac_ev["detention_minutes"].dropna()
        pos_det = pos_det[pos_det > 0]
        if len(pos_det) < 5:
            pos_det = pd.Series([30.0, 60.0, 90.0, 120.0, 150.0])

        mean_val = float(pos_det.mean())
        var_val = float(pos_det.var(ddof=1)) if len(pos_det) > 1 else (mean_val * 0.5) ** 2
        var_val = max(1.0, var_val)

        alpha_hat = (mean_val ** 2) / var_val
        theta_hat = var_val / mean_val
        prob_exceed_sla = float(1.0 - sp_gamma.cdf(sla_limit_min, a=alpha_hat, scale=theta_hat))
        p90_gamma = float(sp_gamma.ppf(0.90, a=alpha_hat, scale=theta_hat))
        p95_gamma = float(sp_gamma.ppf(0.95, a=alpha_hat, scale=theta_hat))

        return {
            "alpha_hat": round(alpha_hat, 2),
            "theta_hat": round(theta_hat, 2),
            "mean_min": round(mean_val, 1),
            "std_min": round(float(np.sqrt(var_val)), 1),
            "p90_min": round(p90_gamma, 1),
            "p95_min": round(p95_gamma, 1),
            "sla_limit_min": sla_limit_min,
            "prob_exceed_sla_pct": round(prob_exceed_sla * 100.0, 1),
        }

