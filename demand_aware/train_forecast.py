"""
Train an XGBoost model that forecasts how many requests will start near each
waiting spot in the next 30 minutes, and compare it with the last-hour rule.

Run from the OpenHail root folder:
    python demand_aware/train_forecast.py
Outputs: models/demand_model.json, results/forecast_*.csv, figures/forecast_*.png
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from openhail.core.constants import ORIG, TIME
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils.arg_utilities import get_parser, load_config

from policies import FEATURES, make_features
from settings import (FLEET, FORECAST_SECONDS, HORIZON, INFRASTRUCTURE,
                      REQUESTS_PER_DAY, TEST_DAYS)

HERE = Path(__file__).parent
DRAWS_PER_DAY = 10   # random 2,000-trip draws per real day, made with OpenHail's generator
STEP_SECONDS = 300   # one training example every 5 minutes
N_FOLDS = 5


def score(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred)}


def build_table(instance):
    """One row per (real day, draw, 5-minute moment, waiting spot)."""
    generator = instance.request_generator
    trips = generator.requests_df
    spots = instance.repo_coords
    radius = instance.max_distance_from_request

    # Chance that a random point in each taxi zone falls within reach of each spot
    # (OpenHail places every request at a random point inside its zone).
    rng = np.random.RandomState(0)
    near_prob = {}
    for zone in generator.zone_ids:
        points = np.array([generator._rand_pt_in_zone(zone, rng) for _ in range(500)])
        near_prob[zone] = [float(np.mean(np.linalg.norm(points - s, axis=1) <= radius))
                           for s in spots]

    rows = []
    for day in sorted(int(d) for d in trips["doy"].unique()):
        day_trips = trips[trips["doy"] == day]
        scale = REQUESTS_PER_DAY / len(day_trips)
        order = np.argsort(day_trips["episode_time"].to_numpy())
        trip_times = day_trips["episode_time"].to_numpy()[order]
        trip_near = np.array([near_prob[z] for z in day_trips["puzone"]])[order]
        cumulative = np.vstack([np.zeros(len(spots)), np.cumsum(trip_near, axis=0)])

        for draw in range(DRAWS_PER_DAY):
            sample = generator._generate_doy(day, 1000 + draw, REQUESTS_PER_DAY, 0, 86400)
            times = sample[TIME].to_numpy()
            origins = np.stack(sample[ORIG].to_numpy())
            for now in range(STEP_SECONDS, 86400 - FORECAST_SECONDS + 1, STEP_SECONDS):
                seen = (times < now) & (times >= now - 3600)
                features = make_features(now, times[seen], origins[seen], spots, radius)
                a = np.searchsorted(trip_times, now)
                b = np.searchsorted(trip_times, now + FORECAST_SECONDS)
                expected = scale * (cumulative[b] - cumulative[a])  # target
                for spot_id, row in enumerate(features):
                    row.update(target=expected[spot_id], day=day, draw=draw, now=now)
                    rows.append(row)
        print(f"day {day}: done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    args = get_parser(evaluation="S1", city="nyc", horizon=HORIZON, fleet=FLEET,
                      infrastructure=INFRASTRUCTURE, agent="nearest",
                      simulation="debug_config").parse_args([])
    config, _ = load_config(args)
    data = build_table(OpenhailInstance(config))

    train_days = sorted(int(d) for d in data.day.unique() if d not in TEST_DAYS)
    train = data[data.day.isin(train_days)].reset_index(drop=True)
    test = data[data.day.isin(TEST_DAYS)].reset_index(drop=True)
    train["fold"] = train.day.map({d: i % N_FOLDS for i, d in enumerate(train_days)})

    # Grouped cross-validation on the training days to choose the settings.
    cv_rows = []
    for depth in [3, 5, 7]:
        for rate in [0.05, 0.1]:
            maes, trees = [], []
            for fold in range(N_FOLDS):
                fit, val = train[train.fold != fold], train[train.fold == fold]
                model = XGBRegressor(n_estimators=2000, max_depth=depth, learning_rate=rate,
                                     subsample=0.8, colsample_bytree=0.8,
                                     early_stopping_rounds=50, eval_metric="mae",
                                     tree_method="hist", random_state=0)
                model.fit(fit[FEATURES], fit.target,
                          eval_set=[(val[FEATURES], val.target)], verbose=False)
                maes.append(mean_absolute_error(val.target, model.predict(val[FEATURES])))
                trees.append(model.best_iteration + 1)
            cv_rows.append({"max_depth": depth, "learning_rate": rate,
                            "validation_MAE": np.mean(maes), "MAE_std": np.std(maes),
                            "trees": int(np.mean(trees))})
            print(f"depth {depth}, rate {rate}: validation MAE {np.mean(maes):.3f}", flush=True)
    cv = pd.DataFrame(cv_rows).sort_values("validation_MAE")
    best = cv.iloc[0]

    model = XGBRegressor(n_estimators=int(best.trees), max_depth=int(best.max_depth),
                         learning_rate=float(best.learning_rate), subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", random_state=0)
    model.fit(train[FEATURES], train.target)

    # Final test on the 8 days the simulation uses.
    test["last_hour"] = test.near_last_60 / 2   # the last-hour rule as a 30-min forecast
    test["xgboost"] = model.predict(test[FEATURES])
    results = pd.DataFrame({"last-hour rule": score(test.target, test.last_hour),
                            "XGBoost": score(test.target, test.xgboost)}).T
    moments = test.groupby(["day", "draw", "now"])
    true_best = moments.target.idxmax().values
    results["busiest_spot_found_%"] = [
        100 * np.mean(moments[col].idxmax().values == true_best)
        for col in ["last_hour", "xgboost"]]
    print(results.round(3).to_string())

    importance = pd.Series(model.get_booster().get_score(importance_type="gain"))
    importance = (100 * importance / importance.sum()).sort_values()

    # Save the model and the tables.
    model.save_model(HERE / "models" / "demand_model.json")
    (HERE / "models" / "demand_model_info.json").write_text(json.dumps(
        {"features": FEATURES, "infrastructure": INFRASTRUCTURE, "train_days": train_days,
         "test_days": TEST_DAYS, "max_depth": int(best.max_depth),
         "learning_rate": float(best.learning_rate), "trees": int(best.trees)}, indent=2))
    results.to_csv(HERE / "results" / "forecast_test_results.csv")
    cv.to_csv(HERE / "results" / "forecast_cross_validation.csv", index=False)

    # Figure 1: forecast accuracy (MAE and R2) on the test days.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    colors = ["#8c8c8c", "#2a7ab9"]
    for ax, metric, label in [(axes[0], "MAE", "MAE (requests per 30 min, lower is better)"),
                              (axes[1], "R2", "R² (higher is better)")]:
        bars = ax.bar(results.index, results[metric], color=colors)
        ax.bar_label(bars, fmt="%.2f")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, max(results[metric]) * 1.2)
    fig.suptitle("Forecast accuracy on the 8 test days")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "forecast_accuracy.png", dpi=150)

    # Figure 2: feature importance.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(importance.index, importance.values, color="#2a7ab9")
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("What the XGBoost model relies on")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "feature_importance.png", dpi=150)

    # Figure 3: one test day, one spot, over the day.
    example = test[(test.day == TEST_DAYS[3]) & (test.draw == 0) & (test.spot == 0)].sort_values("now")
    hours = example.now / 3600 + 3   # 3 = 3:00 am ... 26.5 = 2:30 am next morning
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hours, example.target, color="black", label="actual expected demand")
    ax.plot(hours, example.last_hour, color="#8c8c8c", label="last-hour rule")
    ax.plot(hours, example.xgboost, color="#2a7ab9", label="XGBoost")
    ax.set_xticks(range(3, 28, 3))
    ax.set_xticklabels([f"{h % 24}:00" for h in range(3, 28, 3)])
    ax.set_xlabel("Time of day (the simulated day starts at 3:00 am)")
    ax.set_ylabel("Requests near spot 0, next 30 min")
    ax.set_title(f"Example: test day Jan {TEST_DAYS[3]}, 2018, spot 0")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "forecast_example.png", dpi=150)
    print("saved model, results and figures")
