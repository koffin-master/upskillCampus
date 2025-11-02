#!/usr/bin/env python3
"""
crop.py

Load, clean, merge 5 agriculture CSVs and run a baseline ML experiment
to predict yearly production per crop (and state when available).

"""

import re
import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# ---------------- CONFIG ----------------
CSV_A_COST = "/Users/rahmani/Downloads/Project4_Ag_Prediction of Agriculture Crop Production In India/Project4_Ag_Prediction of Agriculture Crop Production In India/datafile (1).csv"          # file 1: cost & yield by Crop+State (sample 1)
CSV_B_PROD_BY_YEARS = "/Users/rahmani/Downloads/Project4_Ag_Prediction of Agriculture Crop Production In India/Project4_Ag_Prediction of Agriculture Crop Production In India/datafile (2).csv"  # file 2: Production/Area/Yield wide table (sample 2)
CSV_C_VARIETY = "/Users/rahmani/Downloads/Project4_Ag_Prediction of Agriculture Crop Production In India/Project4_Ag_Prediction of Agriculture Crop Production In India/datafile (3).csv"    # file 3: Variety/Season/Recommended Zone (sample 3)
CSV_D_INDEX = "/Users/rahmani/Downloads/Project4_Ag_Prediction of Agriculture Crop Production In India/Project4_Ag_Prediction of Agriculture Crop Production In India/datafile.csv"       # file 4: Indexes by year (sample 4)
CSV_E_PRODUCTION_TIME = "/Users/rahmani/Downloads/Project4_Ag_Prediction of Agriculture Crop Production In India/Project4_Ag_Prediction of Agriculture Crop Production In India/produce.csv"  # file 5: timeseries table (sample 5)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# ---------------- Helpers: cleaning numeric strings ----------------
def clean_number_str(s):
    """Heuristic convert string with commas to a floatable string."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    # remove thousands separators and replace decimal commas
    # common heuristics:
    # - if there are multiple commas -> remove all commas
    # - if single comma and right side length == 3 -> thousands separator -> remove comma
    # - else replace comma with dot (decimal)
    if "," not in s:
        # maybe contains spaces or non-numeric chars
        s = s.replace(" ", "")
        s = re.sub(r"[^\d\.\-eE]", "", s)
        return s if s != "" else np.nan
    parts = s.split(",")
    if len(parts) == 2:
        right_len = len(parts[1])
        if right_len == 3:
            # thousands sep
            joined = parts[0] + parts[1]
            return re.sub(r"[^\d\.\-eE]", "", joined)
        else:
            # decimal separator
            replaced = parts[0] + "." + parts[1]
            return re.sub(r"[^\d\.\-eE]", "", replaced)
    else:
        # many commas -> remove them
        removed = "".join(parts)
        return re.sub(r"[^\d\.\-eE]", "", removed)

def try_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except:
        try:
            cleaned = clean_number_str(x)
            return float(cleaned) if cleaned is not None and cleaned != "" else np.nan
        except:
            return np.nan

# ---------------- Loaders for each CSV ----------------
def load_cost_file(path):
    """
    File 1 sample columns:
    Crop, State, Cost of Cultivation (`/Hectare) A2+FL, Cost of Cultivation (`/Hectare) C2, Cost of Production (`/Quintal) C2, Yield (Quintal/ Hectare)
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # normalize column names
    rename_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename_map)
    # apply numeric parsing to all except Crop/State
    for c in df.columns:
        if c.lower() not in ["crop", "state"]:
            df[c] = df[c].apply(try_float)
    return df

def melt_year_blockwide(df, prefix_filters: List[str], value_name_base: str):
    """
    Generic: given a dataframe where many columns are 'Name YEAR' or 'Name yyyy-yy',
    collapse columns that match prefix_filters into long form.
    Example: columns 'Production 2006-07', 'Production 2007-08', ...
    Returns df_long with Year as int (end-year) and value in column value_name_base.
    """
    # find relevant columns by pattern
    cols = [c for c in df.columns if any(p in c for p in prefix_filters)]
    # determine other id columns: likely Crop, maybe State
    id_cols = [c for c in df.columns if c not in cols]
    # melt
    df_long = df.melt(id_vars=id_cols, value_vars=cols, var_name="variable", value_name=value_name_base)
    # extract year from variable (try common patterns like '2006-07' or '2006-2007' or '2006')
    def parse_year(v):
        m = re.search(r"(20\d{2})\D*(\d{2,4})", v)
        if m:
            # if second group is 2-digit, get century from first
            y1 = int(m.group(1))
            y2 = m.group(2)
            if len(y2) == 2:
                # e.g. 2006-07 -> take the last two digits and convert to 2007
                y2_full = int(str(y1)[:2] + y2)
                # choose end year
                return y2_full
            else:
                return int(y2)
        # fallback: any 4-digit year
        m2 = re.search(r"(20\d{2})", v)
        if m2:
            return int(m2.group(1))
        return np.nan
    df_long["Year"] = df_long["variable"].apply(parse_year)
    df_long.drop(columns=["variable"], inplace=True)
    # numeric convert
    df_long[value_name_base] = df_long[value_name_base].apply(try_float)
    return df_long

def load_prod_by_years(path):
    """
    File 2 sample: rows by Crop with columns:
    Production 2006-07, Production 2007-08, ... and Area 2006-07, Yield 2006-07 etc.
    We'll melt production, area, yield into long form and then merge on crop+year.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # find blocks: Production, Area, Yield
    prod = melt_year_blockwide(df, ["Production"], "Production")
    area = melt_year_blockwide(df, ["Area"], "Area")
    yield_ = melt_year_blockwide(df, ["Yield"], "Yield")
    # merge on Crop + Year (and other id cols if present)
    id_cols = [c for c in df.columns if c.lower() not in [col for col in df.columns if any(k in col for k in ["Production","Area","Yield"])]]
    # ensure Crop exists
    key_vars = ['Crop'] if 'Crop' in df.columns else id_cols
    merged = prod.merge(area, on=key_vars + ["Year"], how="outer")
    merged = merged.merge(yield_, on=key_vars + ["Year"], how="outer")
    # clean crop name spaces
    if 'Crop' in merged.columns:
        merged['Crop'] = merged['Crop'].str.strip()
    return merged

def load_variety_file(path):
    """
    File 3 sample: Crop, Variety, Season/ duration in days, Recommended Zone
    Keep as-is, trim strings.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"": None})
    return df

def load_index_file(path):
    """
    File 4 sample: rows are categories (Rice, Wheat, ...) columns are years 2004-05, 2005-06...
    We'll melt to long with Year and Index value.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    id_cols = [c for c in df.columns if not re.search(r"20\d{2}", c)]
    # melt all year columns
    year_cols = [c for c in df.columns if c not in id_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="YearLabel", value_name="IndexVal")
    # parse year like '2004-05' -> end year 2005
    def parse_index_year(lbl):
        m = re.search(r"(20\d{2})\D*(\d{2})", lbl)
        if m:
            y1 = int(m.group(1))
            y2 = int(m.group(2))
            # create full year for end-year
            y_end = int(str(y1)[:2] + f"{y2:02d}")
            # if y_end < y1: might be century flip; keep y1 as fallback
            return y_end if y_end >= y1 else y1
        m2 = re.search(r"(20\d{2})", lbl)
        if m2:
            return int(m2.group(1))
        return np.nan
    df_long["Year"] = df_long["YearLabel"].apply(parse_index_year)
    df_long["IndexVal"] = df_long["IndexVal"].apply(try_float)
    # rename first id col to 'Category' if likely
    id0 = id_cols[0] if id_cols else None
    if id0 and id0.lower() != 'year':
        df_long = df_long.rename(columns={id0: "Category"})
    return df_long.drop(columns=["YearLabel"])

def load_production_timeseries(path):
    """
    File 5 sample: has columns like 'Particulars','Frequency','Unit','3-1993','3-1994',...
    We'll melt year columns '3-YYYY' to Year and Value.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    id_cols = ['Particulars', 'Frequency', 'Unit']
    actual_id_cols = [c for c in id_cols if c in df.columns]
    year_cols = [c for c in df.columns if re.search(r"\b(19|20)\d{2}\b", c) or re.search(r"\d{4}", c)]
    # Some columns appear as '3-1993' so include them
    if not year_cols:
        year_cols = [c for c in df.columns if re.search(r"\d{4}", c)]
    df_long = df.melt(id_vars=actual_id_cols, value_vars=year_cols, var_name="YearLabel", value_name="Value")
    # parse YearLabel like '3-1993' or '1993' -> get 1993
    def parse_yr(lbl):
        m = re.search(r"(19|20)\d{2}", str(lbl))
        if m:
            return int(m.group(0))
        return np.nan
    df_long["Year"] = df_long["YearLabel"].apply(parse_yr)
    df_long["Value"] = df_long["Value"].apply(try_float)
    return df_long.drop(columns=["YearLabel"])

# ---------------- Merge strategy ----------------
def build_master_table(cost_df, prod_by_years_df, variety_df, index_df, timeseries_df):
    """
    Aim to produce a table with columns:
    Crop, State (if available), Year, Production, Area, Yield, CostA2FL, CostC2, CostPerQuintal, Index values (if match), timeseries values (if match)
    Merge rules:
    - Merge prod_by_years_df (has Crop + Year + Production/Area/Yield) as base
    - Left-join cost_df on Crop + State (if state present in both)
    - Left-join variety_df (by Crop) to add season/zone info (no Year)
    - Left-join index_df by Category==Crop and Year if Category represents same as Crop (best-effort)
    - Left-join timeseries rows where Particulars contains Crop/Category keywords (best-effort)
    """
    # start with production-by-years (base)
    base = prod_by_years_df.copy()
    # clean crop names
    if 'Crop' in base.columns:
        base['Crop'] = base['Crop'].astype(str).str.strip()

    # Merge cost: if cost has State and base has State then merge on both; else merge on Crop only
    cost = cost_df.copy()
    if 'Crop' in cost.columns:
        cost['Crop'] = cost['Crop'].astype(str).str.strip()
    if 'State' in cost.columns and 'State' in base.columns:
        base = base.merge(cost, on=['Crop', 'State', 'Year'], how='left') if 'Year' in cost.columns else base.merge(cost, on=['Crop', 'State'], how='left')
    else:
        base = base.merge(cost, on=['Crop'], how='left')

    # Merge variety (no Year) on Crop
    var = variety_df.copy()
    if 'Crop' in var.columns:
        var['Crop'] = var['Crop'].astype(str).str.strip()
        # drop duplicates in var to avoid multiplications; keep first
        var_unique = var.drop_duplicates(subset=['Crop'])
        base = base.merge(var_unique, on='Crop', how='left')

    # Merge index by Category->Crop where possible
    idx = index_df.copy()
    if 'Category' in idx.columns:
        idx['Category'] = idx['Category'].astype(str).str.strip()
        # join idx where Category == Crop
        base = base.merge(idx.rename(columns={'Category': 'Crop', 'IndexVal': 'IndexVal'}), on=['Crop', 'Year'], how='left')

    # Merge timeseries by matching Particulars to Crop or Category keywords (best-effort)
    ts = timeseries_df.copy()
    # For simplicity attempt to get rows where 'Particulars' contains the crop name (case-insensitive)
    def find_ts_value(row):
        crop = str(row['Crop'])
        year = row['Year']
        mask = ts['Particulars'].str.contains(crop, case=False, na=False) & (ts['Year'] == year)
        vals = ts.loc[mask, 'Value']
        return vals.iloc[0] if len(vals) > 0 else np.nan

    if 'Particulars' in ts.columns:
        base['TimeseriesValue'] = base.apply(find_ts_value, axis=1)
    else:
        base['TimeseriesValue'] = np.nan

    return base

# ---------------- Simple EDA ----------------
def basic_eda(df):
    print("=== BASIC EDA ===")
    print("Rows,cols:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\nSample rows:")
    print(df.head(8))
    # simple target distribution if Production exists
    if 'Production' in df.columns:
        print("\nProduction stats:")
        print(df['Production'].describe())
        # quick histogram
        try:
            df['Production'].dropna().astype(float).hist(bins=40)
            plt.title("Production distribution")
            plt.show()
        except Exception:
            pass

# ---------------- Modeling ----------------
def prepare_features_for_model(df):
    """
    Choose a subset of columns as features and target.
    We'll predict 'Production' (numeric). Features include Area, Yield, Costs, IndexVal, TimeseriesValue, Year, and encoded Crop (via one-hot or label).
    For simplicity we will label-encode Crop and State (if present) and use numeric features.
    """
    df_model = df.copy()
    # drop rows without target
    df_model = df_model.dropna(subset=['Production'])
    # convert atomic numeric cols
    numeric_cols = ['Production', 'Area', 'Yield', 'Cost of Cultivation (`/Hectare) A2+FL',
                    'Cost of Cultivation (`/Hectare) C2', 'Cost of Production (`/Quintal) C2',
                    'Yield (Quintal/ Hectare)', 'IndexVal', 'TimeseriesValue']
    # find columns that exist and cast to float
    for c in numeric_cols:
        if c in df_model.columns:
            df_model[c] = pd.to_numeric(df_model[c], errors='coerce')

    # Feature engineering: if Area and Yield present but Production not, fill Production = Area * Yield
    if 'Production' in df_model.columns and 'Area' in df_model.columns and 'Yield' in df_model.columns:
        missing_prod_mask = df_model['Production'].isna() & df_model['Area'].notna() & df_model['Yield'].notna()
        df_model.loc[missing_prod_mask, 'Production'] = df_model.loc[missing_prod_mask, 'Area'] * df_model.loc[missing_prod_mask, 'Yield']

    # select features we will try
    feat_cols = []
    for c in ['Area', 'Yield', 'Cost of Cultivation (`/Hectare) A2+FL', 'Cost of Cultivation (`/Hectare) C2',
              'Cost of Production (`/Quintal) C2', 'Yield (Quintal/ Hectare)', 'IndexVal', 'TimeseriesValue']:
        if c in df_model.columns:
            feat_cols.append(c)

    # year
    if 'Year' in df_model.columns:
        df_model['Year'] = pd.to_numeric(df_model['Year'], errors='coerce')
        feat_cols.append('Year')

    # encode Crop and State as categorical codes
    for c in ['Crop', 'State']:
        if c in df_model.columns:
            df_model[c] = df_model[c].astype(str).str.strip()
            df_model[c + "_code"] = df_model[c].astype('category').cat.codes
            feat_cols.append(c + "_code")

    # drop rows with all features missing
    df_model = df_model.dropna(subset=feat_cols + ['Production'], how='any')
    X = df_model[feat_cols].fillna(0)
    y = df_model['Production'].astype(float)
    return X, y, df_model

def train_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"Test rows: {len(X_test)} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")
    return model, X_test, y_test, preds

# ---------------- Main ----------------
def main():
    print("Loading files...")
    cost_df = load_cost_file(CSV_A_COST)
    print("Loaded cost file:", cost_df.shape)
    prod_by_years_df = load_prod_by_years(CSV_B_PROD_BY_YEARS)
    print("Loaded prod_by_years:", prod_by_years_df.shape)
    variety_df = load_variety_file(CSV_C_VARIETY)
    print("Loaded variety file:", variety_df.shape)
    index_df = load_index_file(CSV_D_INDEX)
    print("Loaded index file:", index_df.shape)
    timeseries_df = load_production_timeseries(CSV_E_PRODUCTION_TIME)
    print("Loaded timeseries file:", timeseries_df.shape)

    print("\nBuilding master table by merging sources...")
    master = build_master_table(cost_df, prod_by_years_df, variety_df, index_df, timeseries_df)
    print("Master table shape:", master.shape)
    master.to_csv(os.path.join(OUTPUT_DIR, "master_pre_merge.csv"), index=False)

    basic_eda(master)

    print("\nPrepare features & target...")
    X, y, df_model = prepare_features_for_model(master)
    print("Feature matrix shape:", X.shape)

    if X.shape[0] < 20:
        print("Not enough rows for modeling after merges and cleaning. Check merges and data coverage.")
        return

    print("\nTraining baseline RandomForest...")
    model, X_test, y_test, preds = train_evaluate(X, y)

    # save predictions sample
    results_df = df_model.loc[X_test.index].copy()
    results_df["y_true"] = y_test
    results_df["y_pred"] = preds
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results_sample.csv"), index=False)
    print(f"Saved results sample to {os.path.join(OUTPUT_DIR, 'model_results_sample.csv')}")

    # save model
    joblib.dump(model, os.path.join(OUTPUT_DIR, "rf_production_model.joblib"))
    print(f"Saved model to {os.path.join(OUTPUT_DIR, 'rf_production_model.joblib')}")

    print("\nTop feature importances:")
    try:
        importances = model.feature_importances_
        for f, imp in sorted(zip(X.columns.tolist(), importances), key=lambda x: x[1], reverse=True)[:20]:
            print(f"{f}: {imp:.4f}")
    except Exception as e:
        print("Could not compute feature importances:", e)

if __name__ == "__main__":
    main()