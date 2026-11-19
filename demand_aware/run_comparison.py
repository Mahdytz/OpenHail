"""
Compare three ways of choosing where idle vehicles wait:
  nearest    OpenHail's built-in policy (closest waiting spot)
  last_hour  the spot with the most requests nearby in the last hour
  xgboost    the spot with the highest XGBoost forecast for the next 30 minutes

Run from the OpenHail root folder, after train_forecast.py:
    python demand_aware/run_comparison.py
Outputs: results/comparison_*.csv and figures/comparison_*.png
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from openhail.agents.agent_initializer import initialize_agent
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils.arg_utilities import get_parser, load_config

from policies import DemandAwareNearest, XGBoostDemand, clock_hour
from settings import FLEET, HORIZON, INFRASTRUCTURE, POLICIES, SEEDS

HERE = Path(__file__).parent
LABELS = {"nearest": "nearest", "last_hour": "last-hour", "xgboost": "XGBoost"}
COLORS = {"nearest": "#8c8c8c", "last_hour": "#e08a2c", "xgboost": "#2a7ab9"}


def run_one(policy_name, seed, model):
    """Simulate one 2-day period with one policy and return the measurements."""
    args = get_parser(evaluation="S1", city="nyc", horizon=HORIZON, fleet=FLEET,
                      infrastructure=INFRASTRUCTURE, agent="nearest",
                      simulation="debug_config").parse_args([])
    config, extra = load_config(args)
    instance = OpenhailInstance(config)
    env = OpenhailEnv(instance, config["simulation"], {"render_mode": None})

    if policy_name == "nearest":
        policy = initialize_agent(extra["agent"], instance)
    elif policy_name == "last_hour":
        policy = DemandAwareNearest(instance)
    else:
        policy = XGBoostDemand(instance, model)

    state, _ = env.reset(seed=seed)
    profit = trip_profit = empty_cost = battery_time = total_time = 0.0
    served = total = longest_queue = 0
    added_by_hour, served_by_hour = [0] * 24, [0] * 24
    finished = False
    while not finished:
        hour_now = int(clock_hour(float(state["time"])))
        state, reward, terminated, truncated, info = env.step(policy.choose_action(state))
        profit += reward
        trip_profit += info["reward_components"]["service"]
        empty_cost += info["reward_components"]["reposition"]
        served += info["requests_served"]
        total += info["requests_added"]
        served_by_hour[hour_now] += info["requests_served"]
        added_by_hour[int(clock_hour(float(state["time"])))] += info["requests_added"]
        battery_time += info["mean_soc"] * info["delta_time"]
        total_time += info["delta_time"]
        longest_queue = max(longest_queue, int(np.sum(info["charger_queue"])))
        finished = terminated or truncated
    env.close()

    result = {"policy": policy_name, "seed": seed, "requests": total, "served": served,
              "lost": total - served, "served_%": 100 * served / max(total, 1),
              "profit": profit, "trip_profit": trip_profit, "empty_driving_cost": empty_cost,
              "profit_per_served": profit / max(served, 1),
              "avg_battery_%": 100 * battery_time / max(total_time, 1),
              "max_charger_queue": longest_queue}
    return result, added_by_hour, served_by_hour


def make_figures(results, hourly):
    """Save the comparison figures (can also be called on the saved CSV files)."""
    # Figure 1: profit and share served, mean over seeds with each seed as a dot.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, column, title in [(axes[0], "profit", "Profit over 2 days"),
                              (axes[1], "served_%", "Requests served (%)")]:
        for i, p in enumerate(POLICIES):
            values = results[results.policy == p][column]
            ax.bar(i, values.mean(), color=COLORS[p], alpha=0.85)
            ax.scatter([i] * len(values), values, color="black", s=12, zorder=3)
            ax.text(i, values.max() * 1.02, f"{values.mean():,.0f}" if column == "profit"
                    else f"{values.mean():.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(POLICIES)))
        ax.set_xticklabels([LABELS[p] for p in POLICIES])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, results[column].max() * 1.15)
        if column == "profit":
            ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.suptitle(f"{FLEET}, {INFRASTRUCTURE}, {len(SEEDS)} seeds (bars = mean, dots = seeds)")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "comparison_profit_served.png", dpi=150)

    # Figure 2: share served by hour of day (all seeds together).
    by_hour = hourly.groupby(["policy", "hour"])[["requests", "served"]].sum()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for p in POLICIES:
        part = by_hour.loc[p]
        ax.plot(part.index, 100 * part.served / part.requests.clip(lower=1),
                marker="o", color=COLORS[p], label=LABELS[p])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Requests served (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Share of requests served by hour (all seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "comparison_by_hour.png", dpi=150)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    pd.set_option("display.width", 200)
    model = XGBRegressor()
    model.load_model(HERE / "models" / "demand_model.json")

    rows, hourly_rows = [], []
    for policy_name in POLICIES:
        for seed in SEEDS:
            print(f"running {policy_name}, seed {seed}", flush=True)
            result, added, served = run_one(policy_name, seed, model)
            rows.append(result)
            hourly_rows += [{"policy": policy_name, "seed": seed, "hour": h,
                             "requests": added[h], "served": served[h]} for h in range(24)]
    results = pd.DataFrame(rows)
    hourly = pd.DataFrame(hourly_rows)

    print("\n=== Results per seed ===")
    print(results.round(1).to_string(index=False))

    summary = results.groupby("policy").agg(
        served_pct=("served_%", "mean"), served_pct_min=("served_%", "min"),
        served_pct_max=("served_%", "max"), profit=("profit", "mean"),
        profit_min=("profit", "min"), profit_max=("profit", "max"),
        empty_driving_cost=("empty_driving_cost", "mean"),
        profit_per_served=("profit_per_served", "mean"),
        avg_battery_pct=("avg_battery_%", "mean"),
        max_charger_queue=("max_charger_queue", "mean")).loc[POLICIES]
    base = summary.loc["nearest", "profit"]
    summary["profit_vs_nearest_%"] = 100 * (summary.profit - base) / base
    print("\n=== Mean over seeds ===")
    print(summary.round(1).T.to_string())

    near = results[results.policy == "nearest"].set_index("seed")
    paired = pd.DataFrame({
        f"{p}_profit_gain_%": 100 * (results[results.policy == p].set_index("seed").profit
                                     - near.profit) / near.profit
        for p in ["last_hour", "xgboost"]})
    print("\n=== Profit gain over nearest, same seed (%) ===")
    print(paired.round(1).to_string())

    results.to_csv(HERE / "results" / "comparison_per_seed.csv", index=False)
    summary.to_csv(HERE / "results" / "comparison_summary.csv")
    hourly.to_csv(HERE / "results" / "comparison_by_hour.csv", index=False)

    make_figures(results, hourly)
    print("\nsaved results and figures")
