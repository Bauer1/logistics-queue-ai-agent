import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from backend.data_loader import LogisticsDataStore, parse_operating_hours
from backend.ml_predictor import DetentionMLPredictor


class QueueOptimizerEngine:
    """
    AI Queue Regulation & Dock Time-Slot Optimization Engine.
    Combines multi-factor priority scoring, operating hours enforcement,
    active dock bay scheduling, and city cross-dock overflow balancing.
    """

    def __init__(self, store: LogisticsDataStore, predictor: DetentionMLPredictor):
        self.store = store
        self.predictor = predictor

    def compute_priority_score(self, row: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.0
        reasons = []

        # 1. Cargo Type (Cold chain priority)
        load_type = str(row.get("load_type", "Dry Van"))
        if load_type == "Refrigerated":
            score += 34.0
            reasons.append("Рефрижератор (контроль темп.) +34")
        else:
            score += 10.0

        # 2. Booking / Contract Tier
        booking = str(row.get("booking_type", "Spot"))
        if booking == "Dedicated":
            score += 24.0
            reasons.append("Выделенный контракт (Dedicated) +24")
        elif booking == "Contract":
            score += 16.0
            reasons.append("Долгосрочный контракт +16")
        else:
            score += 6.0

        # 3. Delay & Detention Risk
        det_min = float(row.get("detention_minutes", 0) or 0)
        delay_min = float(row.get("arrival_delay_min", 0) or 0)
        urgency_pts = min(26.0, max(0.0, (det_min / 220.0) * 18.0 + max(0.0, delay_min / 180.0) * 8.0))
        score += urgency_pts
        if urgency_pts >= 12.0:
            reasons.append(f"Риск срыва SLA / простой +{int(round(urgency_pts))}")

        # 4. Cargo value / volume
        revenue = float(row.get("revenue", 2500) or 2500)
        weight = float(row.get("weight_lbs", 25000) or 25000)
        val_pts = min(16.0, (revenue / 8000.0) * 10.0 + (weight / 45000.0) * 6.0)
        score += val_pts

        final_score = round(min(99.0, max(15.0, score)), 1)
        if final_score >= 75:
            tier = "CRITICAL"
            tier_label = "Критический (Fast-Track)"
        elif final_score >= 55:
            tier = "HIGH"
            tier_label = "Высокий приоритет"
        elif final_score >= 38:
            tier = "MEDIUM"
            tier_label = "Стандартный"
        else:
            tier = "LOW"
            tier_label = "Низкий (Гибкий слот)"

        return {
            "priority_score": final_score,
            "priority_tier": tier,
            "priority_label": tier_label,
            "priority_reasons": reasons,
        }

    def optimize_facility_queue(
        self,
        facility_id: str,
        target_date: Optional[str] = None,
        window_days: int = 3,
        active_bays_override: Optional[int] = None,
        allow_city_reroute: bool = True,
        surge_extra_trucks: int = 0,
    ) -> Dict[str, Any]:
        fac = self.store.facility_dict.get(facility_id)
        if not fac:
            raise ValueError(f"Facility {facility_id} not found")

        df = self.store.get_facility_queue_events(
            facility_id=facility_id, target_date=target_date, window_days=window_days
        )

        open_h, close_h, is_24_7 = parse_operating_hours(fac.get("operating_hours", "24/7"))
        total_doors = int(fac.get("dock_doors", 50))
        # Active simultaneously staffed receiving/loading crews (typically 10-20% of physical doors, min 2, max 10)
        default_active_bays = max(2, min(8, int(round(total_doors / 16.0))))
        active_bays = active_bays_override if active_bays_override and active_bays_override > 0 else default_active_bays

        # Check alternative city hubs that have wider operating hours or more capacity
        city_key = f"{fac.get('city')}, {fac.get('state')}"
        city_hubs = [
            h for h in self.store.city_facilities.get(city_key, []) if h["facility_id"] != facility_id
        ]
        open_24_7_hubs = [h for h in city_hubs if h.get("is_24_7") or h.get("operating_hours") == "24/7"]
        wider_hours_hubs = [
            h for h in city_hubs if h.get("close_hour", 17) > close_h or h.get("open_hour", 8) < open_h
        ]
        extended_hubs = open_24_7_hubs if open_24_7_hubs else (wider_hours_hubs if wider_hours_hubs else city_hubs)

        records: List[Dict[str, Any]] = df.to_dict(orient="records")

        # If surge simulation requested, inject realistic unplanned trucks during peak / after-hours
        if surge_extra_trucks > 0 and len(records) > 0:
            base_dt = pd.to_datetime(records[0]["act_dt"]).floor("D")
            rng = np.random.default_rng(42 + surge_extra_trucks)
            for idx in range(surge_extra_trucks):
                # Cluster arrivals around peak morning (8-10) or closed boundary (17-20)
                chosen_hour = int(rng.choice([8, 8, 9, 16, 17, 18, 19]))
                chosen_min = int(rng.integers(0, 55))
                sched_t = base_dt + pd.Timedelta(hours=chosen_hour, minutes=chosen_min)
                delay_m = int(rng.integers(15, 140))
                act_t = sched_t + pd.Timedelta(minutes=delay_m)
                is_reefer = bool(rng.random() > 0.45)
                b_type = str(rng.choice(["Dedicated", "Contract", "Spot"], p=[0.45, 0.35, 0.20]))
                det_m = int(rng.integers(110, 235))
                records.append(
                    {
                        "event_id": f"SURGE-{idx+1:03d}",
                        "load_id": f"LOAD-SRG{idx+1:03d}",
                        "trip_id": f"TRIP-SRG{idx+1:03d}",
                        "event_type": str(rng.choice(["Delivery", "Pickup"], p=[0.65, 0.35])),
                        "facility_id": facility_id,
                        "scheduled_datetime": str(sched_t),
                        "actual_datetime": str(act_t),
                        "sched_dt": sched_t,
                        "act_dt": act_t,
                        "sched_date": sched_t.strftime("%Y-%m-%d"),
                        "sched_hour": sched_t.hour,
                        "act_hour": act_t.hour,
                        "detention_minutes": det_m,
                        "on_time_flag": False,
                        "arrival_delay_min": float(delay_m),
                        "load_type": "Refrigerated" if is_reefer else "Dry Van",
                        "booking_type": b_type,
                        "weight_lbs": int(rng.integers(18000, 43000)),
                        "pieces": int(rng.integers(8, 28)),
                        "revenue": round(float(rng.uniform(1800, 6200)), 2),
                        "customer_name": "Surge Logistics Express" if is_reefer else "Metro Retail Supply",
                        "first_name": "Alex",
                        "last_name": f"Surge#{idx+1}",
                        "make": "Freightliner",
                        "unit_number": f"S{900+idx}",
                        "idle_time_hours": round(det_m / 60.0 + 1.5, 1),
                        "is_surge_injected": True,
                    }
                )

        # Enrich with priority and service duration
        for r in records:
            p_info = self.compute_priority_score(r)
            r.update(p_info)
            r["service_duration_min"] = self.predictor.estimate_service_duration_minutes(
                weight_lbs=r.get("weight_lbs", 28000),
                pieces=r.get("pieces", 14),
                load_type=r.get("load_type", "Dry Van"),
                event_type=r.get("event_type", "Delivery"),
            )
            act_dt = pd.to_datetime(r["act_dt"])
            r["act_dt_obj"] = act_dt
            h = act_dt.hour
            r["is_out_of_hours"] = (not is_24_7) and (h < open_h or h >= close_h)

        # Sort items by day and then schedule within each day using Priority + Arrival Window optimization
        # Group events by date of actual arrival so dock bay scheduling is realistic per operational day
        records.sort(key=lambda x: x["act_dt_obj"])

        # Track when each active bay becomes free: list of datetime objects
        bay_free_times: List[Optional[pd.Timestamp]] = [None] * active_bays

        optimized_queue: List[Dict[str, Any]] = []
        # Process day-by-day clusters, prioritizing high-priority trucks when multiple trucks compete
        # We sort within sliding windows by (-priority_score, act_dt_obj)
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            day_str = r["act_dt_obj"].strftime("%Y-%m-%d")
            by_day.setdefault(day_str, []).append(r)

        total_detention_before = 0
        total_detention_after = 0
        rerouted_count = 0
        slot_shifted_count = 0
        fast_tracked_count = 0
        out_of_hours_before = 0
        reefer_protected_count = 0

        hourly_before = {h: 0 for h in range(24)}
        hourly_after = {h: 0 for h in range(24)}
        hourly_detention_before = {h: [] for h in range(24)}
        hourly_detention_after = {h: [] for h in range(24)}

        for day_str, day_events in by_day.items():
            # Reset bays at start of day
            day_base = pd.to_datetime(day_str)
            day_open_dt = day_base + pd.Timedelta(hours=open_h)
            day_close_dt = day_base + pd.Timedelta(hours=close_h)
            bay_free_times = [day_open_dt for _ in range(active_bays)]

            # Sort events: group into 3-hour operational waves, within each wave sort by priority descending
            for ev in day_events:
                ev["wave_id"] = ev["act_dt_obj"].hour // 3
            day_events.sort(key=lambda x: (x["wave_id"], -x["priority_score"], x["act_dt_obj"]))

            for ev in day_events:
                act_dt: pd.Timestamp = ev["act_dt_obj"]
                sched_dt: pd.Timestamp = pd.to_datetime(ev["sched_dt"])
                orig_det = int(ev.get("detention_minutes", 60) or 60)
                srv_min = int(ev["service_duration_min"])
                is_ooh = bool(ev["is_out_of_hours"])

                total_detention_before += orig_det
                hourly_before[act_dt.hour] += 1
                hourly_detention_before[act_dt.hour].append(orig_det)
                if is_ooh:
                    out_of_hours_before += 1

                assigned_facility_id = facility_id
                assigned_facility_name = fac.get("facility_name", facility_id)
                action_code = "SLOT_OPTIMIZED"
                action_label = "Синхронизация слота и дока"

                # Case 1: Arrived outside operating hours!
                if is_ooh:
                    if allow_city_reroute and len(extended_hubs) > 0 and (
                        ev["priority_score"] >= 52 or orig_det >= 70
                    ):
                        alt = extended_hubs[rerouted_count % len(extended_hubs)]
                        assigned_facility_id = alt["facility_id"]
                        assigned_facility_name = f"{alt['facility_name']} (Хаб {alt['operating_hours']})"
                        opt_start = act_dt + pd.Timedelta(minutes=15)  # 15m city cross-dock transfer
                        opt_end = opt_start + pd.Timedelta(minutes=srv_min)
                        new_det = min(orig_det, int(15 + (srv_min * 0.15)))
                        assigned_bay = f"Хаб {alt['facility_id']} • Док #{(rerouted_count % 6) + 1}"
                        action_code = "CITY_HUB_REROUTE"
                        action_label = f"Перенаправлен в открытый хаб {alt['facility_id']} ({alt['operating_hours']})"
                        rerouted_count += 1
                    else:
                        # Shift appointment slot to open operating hours window (Pre-gate dynamic reschedule)
                        if act_dt.hour < open_h:
                            target_start = day_open_dt
                        else:
                            # Shift to either last open slot before closing or next morning open slot
                            target_start = day_open_dt + pd.Timedelta(minutes=30)
                        # Find earliest available bay in open window
                        bay_idx = int(np.argmin(bay_free_times))
                        earliest_bay_dt = max(target_start, bay_free_times[bay_idx])
                        if not is_24_7 and earliest_bay_dt >= day_close_dt:
                            earliest_bay_dt = day_open_dt + pd.Timedelta(hours=1)
                        opt_start = earliest_bay_dt
                        opt_end = opt_start + pd.Timedelta(minutes=srv_min)
                        bay_free_times[bay_idx] = opt_end + pd.Timedelta(minutes=5)  # 5m dock turnaround
                        new_det = min(orig_det, max(12, int((opt_start - target_start).total_seconds() / 60.0) + 14))
                        assigned_bay = f"Док #{bay_idx + 1}"
                        action_code = "SHIFTED_TO_OPEN_HOURS"
                        action_label = f"Перенос слота в рабочее окно ({fac.get('operating_hours')})"
                        slot_shifted_count += 1
                else:
                    # Inside operating hours: allocate optimal dock bay with Priority Fast-Track
                    bay_idx = int(np.argmin(bay_free_times))
                    bay_ready_dt = bay_free_times[bay_idx]
                    effective_arrival = max(act_dt, day_open_dt)
                    if effective_arrival >= bay_ready_dt:
                        opt_start = effective_arrival
                        queue_wait_min = 0
                    else:
                        opt_start = bay_ready_dt
                        queue_wait_min = int((opt_start - effective_arrival).total_seconds() / 60.0)

                    opt_end = opt_start + pd.Timedelta(minutes=srv_min)
                    bay_free_times[bay_idx] = opt_end + pd.Timedelta(minutes=5)
                    assigned_bay = f"Док #{bay_idx + 1}"

                    # Calculate optimized detention: eliminates uncoordinated yard waiting & paperwork bottlenecks
                    if ev["priority_tier"] in ("CRITICAL", "HIGH"):
                        new_det = min(orig_det, max(8, queue_wait_min + 10))
                        action_code = "PRIORITY_FAST_TRACK"
                        action_label = (
                            "Fast-Track (Приоритет Рефрижератор/SLA)"
                            if ev.get("load_type") == "Refrigerated"
                            else "Приоритетная подача к доку"
                        )
                        fast_tracked_count += 1
                    else:
                        new_det = min(orig_det, max(14, queue_wait_min + 18))
                        if orig_det - new_det >= 30:
                            action_code = "SLOT_SMOOTHED"
                            action_label = "Сглаживание пика (Pre-Check & Слот)"
                            slot_shifted_count += 1

                if ev.get("load_type") == "Refrigerated" and (orig_det - new_det) >= 20:
                    reefer_protected_count += 1

                total_detention_after += new_det
                hourly_after[opt_start.hour] += 1
                hourly_detention_after[opt_start.hour].append(new_det)

                driver_name = f"{ev.get('first_name', 'Driver')} {ev.get('last_name', '')}".strip()
                saved_min = max(0, orig_det - new_det)

                optimized_queue.append(
                    {
                        "event_id": ev.get("event_id"),
                        "load_id": ev.get("load_id"),
                        "trip_id": ev.get("trip_id"),
                        "event_type": ev.get("event_type"),
                        "load_type": ev.get("load_type", "Dry Van"),
                        "booking_type": ev.get("booking_type", "Spot"),
                        "customer_name": ev.get("customer_name") or "Корпоративный клиент",
                        "driver_name": driver_name,
                        "truck_info": f"{ev.get('make', 'Truck')} #{ev.get('unit_number', '101')}",
                        "weight_lbs": int(ev.get("weight_lbs", 25000) or 25000),
                        "pieces": int(ev.get("pieces", 12) or 12),
                        "revenue": float(ev.get("revenue", 2500) or 2500),
                        "priority_score": ev["priority_score"],
                        "priority_tier": ev["priority_tier"],
                        "priority_label": ev["priority_label"],
                        "priority_reasons": ev["priority_reasons"],
                        "scheduled_time": sched_dt.strftime("%Y-%m-%d %H:%M"),
                        "actual_arrival_time": act_dt.strftime("%Y-%m-%d %H:%M"),
                        "optimized_start_time": opt_start.strftime("%Y-%m-%d %H:%M"),
                        "optimized_end_time": opt_end.strftime("%Y-%m-%d %H:%M"),
                        "service_duration_min": srv_min,
                        "is_out_of_hours_before": is_ooh,
                        "assigned_dock": assigned_bay,
                        "assigned_facility_id": assigned_facility_id,
                        "assigned_facility_name": assigned_facility_name,
                        "detention_before_min": orig_det,
                        "detention_after_min": new_det,
                        "detention_saved_min": saved_min,
                        "action_code": action_code,
                        "action_label": action_label,
                        "is_surge_injected": bool(ev.get("is_surge_injected", False)),
                    }
                )

        n_events = max(1, len(optimized_queue))
        saved_total_min = max(0, total_detention_before - total_detention_after)
        avg_before = round(total_detention_before / n_events, 1)
        avg_after = round(total_detention_after / n_events, 1)
        reduction_pct = round((saved_total_min / max(1, total_detention_before)) * 100.0, 1)

        # Financial & Environmental impact calculation
        # Standard industry detention cost: ~$75/hr ($1.25/min), Diesel burn idling: 0.85 gal/hr @ $4.10/gal
        cost_saved_usd = round(saved_total_min * 1.25, 2)
        fuel_saved_gallons = round((saved_total_min / 60.0) * 0.85, 1)
        co2_saved_kg = round(fuel_saved_gallons * 10.18, 1)

        hourly_comparison = []
        for h in range(24):
            is_open_h = True if is_24_7 else (open_h <= h < close_h)
            b_list = hourly_detention_before[h]
            a_list = hourly_detention_after[h]
            hourly_comparison.append(
                {
                    "hour": h,
                    "label": f"{h:02d}:00",
                    "is_open": is_open_h,
                    "arrivals_before": hourly_before[h],
                    "arrivals_after": hourly_after[h],
                    "avg_detention_before": round(float(np.mean(b_list)), 1) if b_list else 0.0,
                    "avg_detention_after": round(float(np.mean(a_list)), 1) if a_list else 0.0,
                }
            )

        return {
            "facility": {
                "facility_id": facility_id,
                "facility_name": fac.get("facility_name"),
                "facility_type": fac.get("facility_type"),
                "city": fac.get("city"),
                "state": fac.get("state"),
                "dock_doors": total_doors,
                "active_bays": active_bays,
                "operating_hours": fac.get("operating_hours"),
                "open_hour": open_h,
                "close_hour": close_h,
                "is_24_7": is_24_7,
                "alternative_city_hubs": [
                    {
                        "facility_id": h["facility_id"],
                        "facility_name": h["facility_name"],
                        "operating_hours": h["operating_hours"],
                        "dock_doors": h["dock_doors"],
                    }
                    for h in city_hubs
                ],
            },
            "kpis": {
                "total_trucks": len(optimized_queue),
                "total_detention_before_min": total_detention_before,
                "total_detention_after_min": total_detention_after,
                "total_detention_saved_min": saved_total_min,
                "avg_detention_before_min": avg_before,
                "avg_detention_after_min": avg_after,
                "detention_reduction_pct": reduction_pct,
                "out_of_hours_arrivals_caught": out_of_hours_before,
                "fast_tracked_trucks": fast_tracked_count,
                "slot_shifted_trucks": slot_shifted_count,
                "rerouted_to_city_hubs": rerouted_count,
                "reefer_loads_protected": reefer_protected_count,
                "estimated_cost_saved_usd": cost_saved_usd,
                "fuel_saved_gallons": fuel_saved_gallons,
                "co2_saved_kg": co2_saved_kg,
            },
            "hourly_profile": hourly_comparison,
            "schedule": optimized_queue,
        }
