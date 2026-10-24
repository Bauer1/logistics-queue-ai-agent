import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

DEFAULT_DATASET_PATH = r"c:\Users\zhuma\Downloads\archive"


def parse_operating_hours(oh_str: str):
    """Returns (open_hour, close_hour, is_24_7) from operating_hours string."""
    oh = str(oh_str).strip().upper()
    if oh == "24/7":
        return 0, 24, True
    if oh == "8AM-5PM":
        return 8, 17, False
    if oh == "7AM-7PM":
        return 7, 19, False
    if oh == "6AM-10PM":
        return 6, 22, False
    return 0, 24, True


class LogisticsDataStore:
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path
        self.facilities: pd.DataFrame = pd.DataFrame()
        self.events: pd.DataFrame = pd.DataFrame()
        self.loads: pd.DataFrame = pd.DataFrame()
        self.trips: pd.DataFrame = pd.DataFrame()
        self.drivers: pd.DataFrame = pd.DataFrame()
        self.trucks: pd.DataFrame = pd.DataFrame()
        self.customers: pd.DataFrame = pd.DataFrame()
        self.routes: pd.DataFrame = pd.DataFrame()

        self.facility_dict: Dict[str, Dict[str, Any]] = {}
        self.city_facilities: Dict[str, List[Dict[str, Any]]] = {}
        self.facility_stats: Dict[str, Dict[str, Any]] = {}
        self.top_congested_days: List[Dict[str, Any]] = []

    def load_all(self):
        print(f"[DataStore] Loading logistics dataset from {self.dataset_path}...")
        self.facilities = pd.read_csv(os.path.join(self.dataset_path, "facilities.csv"))
        self.loads = pd.read_csv(os.path.join(self.dataset_path, "loads.csv"))
        self.trips = pd.read_csv(os.path.join(self.dataset_path, "trips.csv"))
        self.drivers = pd.read_csv(os.path.join(self.dataset_path, "drivers.csv"))
        self.trucks = pd.read_csv(os.path.join(self.dataset_path, "trucks.csv"))
        self.customers = pd.read_csv(os.path.join(self.dataset_path, "customers.csv"))
        self.routes = pd.read_csv(os.path.join(self.dataset_path, "routes.csv"))
        self.events = pd.read_csv(os.path.join(self.dataset_path, "delivery_events.csv"))

        # Parse operating hours on facilities
        oh_parsed = self.facilities["operating_hours"].apply(parse_operating_hours)
        self.facilities["open_hour"] = [x[0] for x in oh_parsed]
        self.facilities["close_hour"] = [x[1] for x in oh_parsed]
        self.facilities["is_24_7"] = [x[2] for x in oh_parsed]

        # Parse timestamps
        self.events["sched_dt"] = pd.to_datetime(self.events["scheduled_datetime"])
        self.events["act_dt"] = pd.to_datetime(self.events["actual_datetime"])
        self.events["sched_date"] = self.events["sched_dt"].dt.strftime("%Y-%m-%d")
        self.events["sched_hour"] = self.events["sched_dt"].dt.hour
        self.events["act_hour"] = self.events["act_dt"].dt.hour
        self.events["arrival_delay_min"] = (
            (self.events["act_dt"] - self.events["sched_dt"]).dt.total_seconds() / 60.0
        ).round(1)

        # Build facility dictionary and city groupings
        for _, row in self.facilities.iterrows():
            raw_dict = row.to_dict()
            f_dict: Dict[str, Any] = {str(k): v for k, v in raw_dict.items()}
            fid = str(f_dict["facility_id"])
            self.facility_dict[fid] = f_dict
            city_key = f"{f_dict['city']}, {f_dict['state']}"
            self.city_facilities.setdefault(city_key, []).append(f_dict)

        # Compute aggregated facility stats
        ev_fac = self.events.merge(
            self.facilities[["facility_id", "open_hour", "close_hour", "is_24_7", "dock_doors"]],
            on="facility_id",
            how="left",
        )
        ev_fac["out_of_hours"] = (~ev_fac["is_24_7"]) & (
            (ev_fac["act_hour"] < ev_fac["open_hour"]) | (ev_fac["act_hour"] >= ev_fac["close_hour"])
        )

        grouped = ev_fac.groupby("facility_id").agg(
            total_events=("event_id", "count"),
            avg_detention=("detention_minutes", "mean"),
            max_detention=("detention_minutes", "max"),
            p90_detention=("detention_minutes", lambda s: float(np.percentile(s, 90))),
            total_detention_hours=("detention_minutes", lambda s: float(s.sum() / 60.0)),
            on_time_rate=("on_time_flag", "mean"),
            out_of_hours_rate=("out_of_hours", "mean"),
            avg_arrival_delay=("arrival_delay_min", "mean"),
        ).reset_index()

        for _, row in grouped.iterrows():
            fid = str(row["facility_id"])
            fac_info = self.facility_dict.get(fid, {})
            doors = max(1, int(fac_info.get("dock_doors", 50)))
            # Congestion index combines avg detention, out-of-hours rate, and event density per door
            density_score = (float(row["total_events"]) / doors) / 100.0
            congestion_index = min(
                100.0,
                round(
                    (float(row["avg_detention"]) / 120.0) * 45.0
                    + (float(row["out_of_hours_rate"]) * 35.0)
                    + min(20.0, density_score * 8.0),
                    1,
                ),
            )
            city_key = f"{fac_info.get('city', '')}, {fac_info.get('state', '')}"
            alt_hubs = [
                str(h["facility_id"])
                for h in self.city_facilities.get(city_key, [])
                if str(h["facility_id"]) != fid
            ]
            self.facility_stats[fid] = {
                **fac_info,
                "total_events": int(row["total_events"]),
                "avg_detention": round(float(row["avg_detention"]), 1),
                "max_detention": int(row["max_detention"]),
                "p90_detention": round(float(row["p90_detention"]), 1),
                "total_detention_hours": round(float(row["total_detention_hours"]), 1),
                "on_time_rate": round(float(row["on_time_rate"]) * 100.0, 1),
                "out_of_hours_rate": round(float(row["out_of_hours_rate"]) * 100.0, 1),
                "avg_arrival_delay": round(float(row["avg_arrival_delay"]), 1),
                "congestion_index": congestion_index,
                "alternative_city_hubs": alt_hubs,
            }

        # Identify top congested facility-dates for quick demo selection
        daily = (
            ev_fac.groupby(["facility_id", "sched_date"])
            .agg(
                events_count=("event_id", "count"),
                total_detention=("detention_minutes", "sum"),
                avg_detention=("detention_minutes", "mean"),
                out_of_hours_count=("out_of_hours", "sum"),
            )
            .reset_index()
        )
        top_daily = daily[daily["events_count"] >= 7].sort_values(
            by=["total_detention", "events_count"], ascending=False
        ).head(25)

        self.top_congested_days = []
        for _, r in top_daily.iterrows():
            fid = str(r["facility_id"])
            finfo = self.facility_dict.get(fid, {})
            self.top_congested_days.append(
                {
                    "facility_id": fid,
                    "facility_name": finfo.get("facility_name", fid),
                    "city": finfo.get("city", ""),
                    "state": finfo.get("state", ""),
                    "operating_hours": finfo.get("operating_hours", "24/7"),
                    "dock_doors": int(finfo.get("dock_doors", 50)),
                    "date": str(r["sched_date"]),
                    "events_count": int(r["events_count"]),
                    "total_detention": int(r["total_detention"]),
                    "avg_detention": round(float(r["avg_detention"]), 1),
                    "out_of_hours_count": int(r["out_of_hours_count"]),
                }
            )

        print(
            f"[DataStore] Loaded {len(self.facilities)} facilities, {len(self.events)} delivery events, {len(self.loads)} loads."
        )

    def get_facility_queue_events(
        self, facility_id: str, target_date: Optional[str] = None, window_days: int = 3
    ) -> pd.DataFrame:
        """
        Retrieves enriched queue events for a facility around target_date.
        If target_date is not provided, selects the busiest date for that facility.
        """
        fac_events = self.events[self.events["facility_id"] == facility_id].copy()
        if fac_events.empty:
            return pd.DataFrame()

        if not target_date:
            # Pick date with highest total detention & event count
            by_date = (
                fac_events.groupby("sched_date")
                .agg(cnt=("event_id", "count"), det=("detention_minutes", "sum"))
                .sort_values(by=["cnt", "det"], ascending=False)
            )
            target_date = str(by_date.index[0])

        start_dt = pd.to_datetime(target_date)
        end_dt = start_dt + pd.Timedelta(days=window_days)

        sub = fac_events[
            (fac_events["sched_dt"] >= start_dt) & (fac_events["sched_dt"] < end_dt)
        ].copy()

        # If window has very few events, grab top 20 events near that date
        if len(sub) < 8:
            sub = fac_events.sort_values("sched_dt").head(20).copy()

        # Enrich with loads, trips, customers, drivers, trucks
        sub = sub.merge(
            self.loads[
                [
                    "load_id",
                    "customer_id",
                    "route_id",
                    "load_type",
                    "weight_lbs",
                    "pieces",
                    "revenue",
                    "booking_type",
                ]
            ],
            on="load_id",
            how="left",
        )
        sub = sub.merge(
            self.trips[
                [
                    "trip_id",
                    "driver_id",
                    "truck_id",
                    "trailer_id",
                    "actual_distance_miles",
                    "idle_time_hours",
                ]
            ],
            on="trip_id",
            how="left",
        )
        sub = sub.merge(
            self.customers[["customer_id", "customer_name", "primary_freight_type"]],
            on="customer_id",
            how="left",
        )
        sub = sub.merge(
            self.drivers[["driver_id", "first_name", "last_name", "years_experience"]],
            on="driver_id",
            how="left",
        )
        sub = sub.merge(
            self.trucks[["truck_id", "make", "unit_number"]],
            on="truck_id",
            how="left",
        )

        return sub.sort_values("act_dt").reset_index(drop=True)
