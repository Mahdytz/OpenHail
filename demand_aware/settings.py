"""Settings shared by all scripts in this folder."""

SEEDS = [0, 1, 2, 3, 4]        # 5 simulation runs; each seed fixes which 2 real days and which trips are used
INFRASTRUCTURE = "S05_1_4"     # 5 waiting spots, 1 charging site with 4 chargers
FLEET = "fleet_20"             # 20 electric vehicles
HORIZON = "day_2"              # 2 simulated days, 2,000 requests per day
MEMORY_SECONDS = 3600          # the last-hour policy looks back 1 hour
FORECAST_SECONDS = 1800        # XGBoost forecasts the next 30 minutes
REQUESTS_PER_DAY = 2000        # must match the horizon file

# Real days (day of year, January 2018) that simulation seeds 0-4 use.
# The forecast model never sees these days during training.
TEST_DAYS = [2, 10, 11, 16, 19, 24, 29, 30]

POLICIES = ["nearest", "last_hour", "xgboost"]
