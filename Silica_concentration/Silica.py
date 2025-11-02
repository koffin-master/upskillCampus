#!/usr/bin/env python3
"""
predict_silica.py

Single-file pipeline:
- Load & clean CSV with mixed comma formats
- Resample to 1-minute frequency (changeable)
- Create lag + rolling features
- Train persistence baseline and RandomForest (XGBoost optional)
- Evaluate multi-step forecast (hours ahead)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# ----- CONFIG -----
CSV_PATH = "/MiningProcess_Flotation_Plant_Database.csv"  #Download the csv file with given link in Readme
DATETIME_COL = "date"
TARGET_COL = "% Silica Concentrate"
TIME_FREQ = "1T"   # 1 minute. change to "1H" etc if desired
LAG_MINUTES = [1, 5, 10, 30, 60]   # minutes
ROLL_WINDOWS = [5, 15, 60]         # minutes
TEST_DAYS = 7                       # last N days used as test (or use fraction split)
RANDOM_SEED = 42
USE_XGBOOST = False   # set True if xgboost installed and you want to use it

# ----- Helpers: cleaning numbers from your sample -----
def clean_number_str(s: str) -> str:
    """Clean numeric token that may contain commas used as decimal or thousands separators.
    Heuristic:
      - If single comma and right part length == 3 -> thousands separator: remove comma
      - If single comma and right part length != 3 -> decimal separator: replace comma->dot
      - If multiple commas -> remove all commas (assume grouping)
    Also strip spaces.
    """
    if pd.isna(s):
        return s
    s = str(s).strip()
    # if already a proper float string
    if s == "":
        return s
    # quick bail if contains no comma
    if "," not in s:
        return s
    parts = s.split(",")
    if len(parts) == 2:
        right_len = len(parts[1])
        if right_len == 3:
            # thousands grouping -> remove comma
            return parts[0] + parts[1]
        else:
            # decimal separator -> replace with dot
            return parts[0] + "." + parts[1]
    else:
        # multiple commas -> remove all commas
        return "".join(parts)

def try_parse_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except:
        try:
            # fallback: clean and parse
            return float(clean_number_str(x))
        except:
            return np.nan

# ----- Load & clean -----
def load_and_clean(path):
    # try reading with tab separator (sample uses tabs). Fallback to comma.
    raw = pd.read_csv(path, sep=",", dtype=str)

    # strip column names
    raw.columns = [c.strip() for c in raw.columns]

    # parse datetime
    if DATETIME_COL not in raw.columns:
        raise ValueError(f"Expected datetime column named '{DATETIME_COL}'. Found: {raw.columns.tolist()}")
    raw[DATETIME_COL] = raw[DATETIME_COL].str.strip()
    df = raw.copy()

    # Clean numeric columns (all except datetime)
    for col in df.columns:
        if col == DATETIME_COL:
            continue
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
        df[col] = df[col].apply(lambda s: clean_number_str(s) if pd.notna(s) else s)
        df[col] = df[col].apply(try_parse_float)

    # convert datetime to pandas datetime
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL]).set_index(DATETIME_COL).sort_index()
    return df

# ----- Resample to uniform grid (1-minute by default) -----
def resample_to_freq(df, freq=TIME_FREQ, agg="mean"):
    """
    Resample to freq. For high-frequency signals aggregated -> mean by default.
    For columns which are 'hourly only', forward-fill will be used so the value is available each minute.
    """
    # aggregate first by timestamp (if duplicates at same timestamp)
    df_agg = getattr(df, "resample")(freq).agg(agg)
    # forward-fill to carry hourly readings forward (reasonable for process sensors)
    df_ffill = df_agg.ffill()
    return df_ffill

# ----- Feature engineering: lags & rolling stats -----
def make_lag_features(df, target_col=TARGET_COL, lags=LAG_MINUTES, rolls=ROLL_WINDOWS):
    X = df.copy()
    # create lag features for all numeric columns (including target)
    for lag in lags:
        shifted = X.shift(lag)   # shift expects periods in same freq (we've resampled to minutes)
        shifted.columns = [f"{c}_lag{lag}m" for c in shifted.columns]
        X = pd.concat([X, shifted], axis=1)

    # rolling stats for numeric columns
    for w in rolls:
        rolled_mean = df.rolling(window=w, min_periods=1).mean().add_suffix(f"_roll_mean{w}m")
        rolled_std  = df.rolling(window=w, min_periods=1).std().add_suffix(f"_roll_std{w}m")
        X = pd.concat([X, rolled_mean, rolled_std], axis=1)

    # time features
    X["hour"] = X.index.hour
    X["minute"] = X.index.minute
    X["dayofweek"] = X.index.dayofweek

    # Drop rows with NaNs produced by shift (first max(lags) minutes)
    max_lag = max(lags) if lags else 0
    X = X.dropna(subset=[f"{target_col}_lag{max_lag}m"] if max_lag>0 else [])
    return X

# ----- Train/test split (time-based) -----
def time_split(df, test_days=TEST_DAYS):
    last_ts = df.index.max()
    split_ts = last_ts - pd.Timedelta(days=test_days)
    train = df[df.index <= split_ts]
    test  = df[df.index >  split_ts]
    return train, test

# ----- Models & evaluation -----
def persistence_forecast(train_y, test_y, horizon_minutes):
    """Persistence: predict last observed value (from t) for t+h"""
    # We assume train/test already aligned to minute grid. For persistence we use previous value at time t for horizon.
    preds = test_y.shift(horizon_minutes).values  # shift downwards; may give NaNs at start -> drop them outside
    return preds

def evaluate_preds(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def train_and_evaluate(X_train, y_train, X_test, y_test, exclude_iron=False):
    # Optionally drop % Iron Concentrate from features (and related derived columns)
    if exclude_iron:
        drop_cols = [c for c in X_train.columns if "% Iron Concentrate" in c]
        X_train = X_train.drop(columns=drop_cols, errors="ignore")
        X_test  = X_test.drop(columns=drop_cols, errors="ignore")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model

# ----- Multi-horizon experiment -----
def multi_horizon_evaluation(df, target_col=TARGET_COL, horizons_hours=[1,2,4,8,12,24], exclude_iron=False):
    results = []
    # Prepare features once
    X_all = make_lag_features(df, target_col=target_col, lags=LAG_MINUTES, rolls=ROLL_WINDOWS)
    y_all = X_all[target_col]
    X_all = X_all.drop(columns=[target_col])

    train_df, test_df = time_split(X_all.join(y_all), test_days=TEST_DAYS)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    for h in horizons_hours:
        h_minutes = h * 60
        # for each horizon, shift target so that model learns to predict y_{t+h} from features at t
        y_train_h = y_train.shift(-h_minutes).dropna()
        X_train_h = X_train.loc[y_train_h.index]

        y_test_h = y_test.shift(-h_minutes).dropna()
        X_test_h = X_test.loc[y_test_h.index]

        # Baseline: persistence (use last observed value)
        persistence_pred = X_test_h[f"{target_col}_lag{LAG_MINUTES[0]}m"].values if LAG_MINUTES else np.full(len(y_test_h), y_test_h.iloc[0])
        pers_metrics = evaluate_preds(y_test_h.values, persistence_pred)

        # Train model
        preds, model = train_and_evaluate(X_train_h, y_train_h.values, X_test_h, y_test_h.values, exclude_iron=exclude_iron)
        model_metrics = evaluate_preds(y_test_h.values, preds)

        results.append({
            "h_hours": h,
            "persistence": pers_metrics,
            "model": model_metrics,
            "n_train": len(X_train_h),
            "n_test": len(X_test_h)
        })
        print(f"H={h}h | persistence MAE={pers_metrics['MAE']:.4f} | model MAE={model_metrics['MAE']:.4f}")
    return results

# ----- Main run -----
def main():
    print("Loading and cleaning CSV...")
    df = load_and_clean(CSV_PATH)
    print(f"Loaded data with {len(df)} rows spanning {df.index.min()} -> {df.index.max()}")

    # quick inspect
    print("Columns:", df.columns.tolist())

    # Resample
    print(f"Resampling to {TIME_FREQ} frequency and forward-filling hourly signals...")
    df_min = resample_to_freq(df, freq=TIME_FREQ, agg="mean")
    print(f"After resample: {len(df_min)} rows")

    # check target exists
    if TARGET_COL not in df_min.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found after cleaning. Found: {df_min.columns.tolist()}")

    # Optionally drop rows with missing target
    df_min = df_min.dropna(subset=[TARGET_COL])
    print(f"Rows after dropping missing target: {len(df_min)}")

    # Quick plot target
    df_min[TARGET_COL].plot(title="% Silica Concentrate (raw)", figsize=(12,3))
    plt.tight_layout()
    plt.show()

    # Multi-horizon eval with %Iron included
    horizons = [1,2,4,8,12,24]
    print("\n=== Multi-horizon evaluation (including % Iron Concentrate) ===")
    results_incl = multi_horizon_evaluation(df_min, target_col=TARGET_COL, horizons_hours=horizons, exclude_iron=False)

    # Multi-horizon eval without %Iron
    print("\n=== Multi-horizon evaluation (excluding % Iron Concentrate) ===")
    results_excl = multi_horizon_evaluation(df_min, target_col=TARGET_COL, horizons_hours=horizons, exclude_iron=True)

    # Summarize
    print("\nSummary (MAE) per horizon (incl vs excl %Iron):")
    for r_in, r_ex in zip(results_incl, results_excl):
        print(f"{r_in['h_hours']:2d}h : incl MAE={r_in['model']['MAE']:.4f} | excl MAE={r_ex['model']['MAE']:.4f} | pers MAE={r_in['persistence']['MAE']:.4f}")

if __name__ == "__main__":
    main()
