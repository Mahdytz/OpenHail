"""
Two policies for choosing where idle vehicles wait in OpenHail.

Both keep OpenHail's "nearest" rules for serving customers and for charging.
They only change where a vehicle with nothing to do goes to wait:

  DemandAwareNearest ("last_hour")
      the reachable waiting spot with the most requests nearby in the last hour.
  XGBoostDemand ("xgboost")
      the reachable waiting spot with the highest XGBoost forecast of requests
      nearby in the next 30 minutes.
"""

import numpy as np
import pandas as pd

from openhail.agents.nearest import NearestAgent
from openhail.core.constants import JOB_NULL, ORIG, PROC_TIME, Q, TIME, TYPE
from openhail.utils import state_utilities

from settings import MEMORY_SECONDS

FEATURES = ["clock_hour", "spot", "near_last_15", "near_last_30",
            "near_last_60", "all_last_60"]


def clock_hour(sim_seconds):
    """OpenHail's simulated day starts at 3:00 am. Returns clock time, e.g. 17.5 = 5:30 pm."""
    return ((sim_seconds % 86400) / 3600 + 3) % 24


def make_features(now, past_times, past_origins, spots, radius):
    """Forecast inputs for every waiting spot at time `now`.

    Uses only requests that appeared before `now`. The same function builds the
    training data and the inputs inside the simulator, so both always match.
    """
    rows = []
    for spot_id, spot in enumerate(spots):
        if len(past_times) > 0:
            age = now - past_times
            near = np.linalg.norm(past_origins - spot, axis=1) <= radius
        else:
            age = np.array([])
            near = np.array([], dtype=bool)
        rows.append({
            "clock_hour": clock_hour(now),
            "spot": spot_id,
            "near_last_15": int(np.sum(near & (age <= 900))),
            "near_last_30": int(np.sum(near & (age <= 1800))),
            "near_last_60": int(np.sum(near & (age <= 3600))),
            "all_last_60": int(np.sum(age <= 3600)),
        })
    return rows


class DemandAwareNearest(NearestAgent):
    """Nearest assignment and charging; idle vehicles wait where demand was high."""

    def __init__(self, instance):
        super().__init__(instance, soc_threshold=0.2)
        self.recent_requests = []  # (time the request appeared, customer pickup point)

    def choose_action(self, state):
        now = float(state["time"])
        requests = state["request"]

        # 1. Remember new requests (empty rows in the request table have PROC_TIME = -1).
        for i in range(len(requests[PROC_TIME])):
            if requests[PROC_TIME][i] != -1:
                self.recent_requests.append((float(requests[TIME][i]), requests[ORIG][i].copy()))

        # 2. Forget requests older than one hour.
        self.recent_requests = [(t, o) for (t, o) in self.recent_requests
                                if t > now - MEMORY_SECONDS]

        # 3. Let nearest decide who serves whom and who charges.
        serve_action, move_action, epoch_action = super().choose_action(state)

        # 4. Idle vehicles: no job after those decisions, and battery above the threshold.
        vehicles = state_utilities.update_state(self.instance, state["V"], requests, serve_action)
        idle_vehicles = [v for v in range(len(vehicles[TYPE]))
                         if vehicles[TYPE][v] == JOB_NULL and vehicles[Q][v] > self.q_thresh]
        if len(idle_vehicles) == 0 or len(self.recent_requests) == 0:
            return serve_action, move_action, epoch_action

        # 5. Score the waiting spots.
        spot_scores = self.score_spots(now)

        # 6. Send each idle vehicle to the reachable spot with the highest score
        #    (the nearer spot wins a tie).
        reachable, distance = state_utilities.compute_rnr_feasibility(
            self.instance, vehicles, np.array(idle_vehicles))
        for row, v in enumerate(idle_vehicles):
            best_spot, best_value = None, -np.inf
            for spot in range(len(spot_scores)):
                if reachable[row, spot]:
                    value = spot_scores[spot] * 1000.0 - distance[row, spot]
                    if value > best_value:
                        best_spot, best_value = spot, value
            if best_spot is not None:
                move_action[v] = 2 * best_spot + 1  # OpenHail code: go to this spot, no charging

        return serve_action, move_action, epoch_action

    def score_spots(self, now):
        """Number of requests in the last hour that started within reach of each spot."""
        origins = np.array([o for (t, o) in self.recent_requests])
        radius = self.instance.max_distance_from_request
        return [np.sum(np.linalg.norm(origins - spot, axis=1) <= radius)
                for spot in self.instance.repo_coords]


class XGBoostDemand(DemandAwareNearest):
    """Same as DemandAwareNearest, but spots are scored by an XGBoost forecast."""

    def __init__(self, instance, model):
        super().__init__(instance)
        self.model = model

    def score_spots(self, now):
        times = np.array([t for (t, o) in self.recent_requests])
        origins = np.array([o for (t, o) in self.recent_requests])
        before = times < now
        rows = make_features(now, times[before], origins[before],
                             self.instance.repo_coords,
                             self.instance.max_distance_from_request)
        return list(self.model.predict(pd.DataFrame(rows)[FEATURES]))
