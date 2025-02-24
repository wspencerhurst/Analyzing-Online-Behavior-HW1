#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from dateutil import parser as date_parser
from dateutil import tz
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Local timezone
local_tz = tz.gettz("America/New_York")

# Threshold for inactivity period (gaps) to consider as potential sleep
SLEEP_INACTIVITY_THRESHOLD = 240  # 4 hours

def read_chrome_history(chrome_history_path):
    if not os.path.exists(chrome_history_path):
        print(f"Chrome History file not found: {chrome_history_path}")
        return pd.DataFrame()

    with open(chrome_history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history_records = data.get("Browser History", [])
    rows = []
    for record in history_records:
        time_usec = record.get("time_usec", None)
        if time_usec is not None:
            epoch_start = datetime(1601, 1, 1, tzinfo=tz.UTC)
            dt_utc = epoch_start + timedelta(microseconds=int(time_usec))
            dt = dt_utc.astimezone(local_tz)
        else:
            dt = None
        rows.append({
            "datetime": dt,
            "title": record.get("title", ""),
            "url": record.get("url", ""),
            "source": "Chrome"
        })
    df = pd.DataFrame(rows)
    df.dropna(subset=["datetime"], inplace=True)
    return df

def read_youtube_json(json_path, event_label=None):
    if not os.path.exists(json_path):
        print(f"YouTube JSON file not found: {json_path}")
        return pd.DataFrame(columns=['datetime', 'title', 'event_type', 'source'])
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        raw_time = item.get("time")
        try:
            dt_utc = date_parser.parse(raw_time)
            dt = dt_utc.astimezone(local_tz)
        except:
            dt = None
        rows.append({
            "datetime": dt,
            "title": item.get("title", ""),
            "event_type": event_label,
            "source": "YouTube"
        })
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    df.dropna(subset=["datetime"], inplace=True)
    return df

def read_youtube_history(watch_history_path, search_history_path):
    w = read_youtube_json(watch_history_path, "watch")
    s = read_youtube_json(search_history_path, "search")
    return pd.concat([w, s], ignore_index=True)

def read_garmin_data(garmin_path):
    if not os.path.exists(garmin_path):
        print(f"Garmin CSV file not found: {garmin_path}")
        return pd.DataFrame()
    df = pd.read_csv(garmin_path, encoding="utf-8")
    df.rename(columns={
        'Sleep Score 4 Weeks': 'DateStr',
        'Bedtime': 'Bedtime',
        'Wake Time': 'WakeTime'
    }, inplace=True)
    if 'Score' in df.columns:
        df = df[df["Score"] != "--"]
    def parse_garmin_date(date_str):
        parsed = datetime.strptime(date_str, "%b %d")
        if parsed.month == 12:
            year = 2024
        else:
            year = 2025
        return datetime(year, parsed.month, parsed.day).date()
    df["Date"] = df["DateStr"].apply(parse_garmin_date)
    def parse_time_str(t):
        if not isinstance(t, str) or t.strip() == "--":
            return None
        try:
            return datetime.strptime(t, "%I:%M %p").time()
        except:
            return None
    df["ParsedBedtime"] = df["Bedtime"].apply(parse_time_str)
    df["ParsedWakeTime"] = df["WakeTime"].apply(parse_time_str)
    def adjust_bedtime_date(row):
        bt = row["ParsedBedtime"]
        wt = row["ParsedWakeTime"]
        d = row["Date"]
        if bt is None or wt is None:
            return None
        if bt < wt:
            bed_date = d
        else:
            bed_date = d - timedelta(days=1)
        naive = datetime.combine(bed_date, bt)
        return naive.replace(tzinfo=local_tz)
    df["BedtimeDT"] = df.apply(adjust_bedtime_date, axis=1)
    df["WakeTimeDT"] = df.apply(
        lambda row: datetime.combine(row["Date"], row["ParsedWakeTime"]).replace(tzinfo=local_tz)
        if row["ParsedWakeTime"] is not None else None, axis=1)
    return df

def identify_sleep_gaps(all_usage, threshold=SLEEP_INACTIVITY_THRESHOLD):
    if all_usage.empty:
        print("No activity data available for sleep gap identification.")
        return pd.DataFrame(columns=["bedtime", "wake_time"])
    all_usage = all_usage.sort_values("datetime")
    all_usage["time_diff"] = all_usage["datetime"].diff().dt.total_seconds() / 60
    sleep_gaps = all_usage[all_usage["time_diff"] > threshold].copy()
    if sleep_gaps.empty:
        print("No sleep gaps identified. Check activity data or threshold.")
        return pd.DataFrame(columns=["bedtime", "wake_time"])
    sleep_gaps["bedtime"] = pd.to_datetime(sleep_gaps["datetime"].shift(1), errors='coerce')
    sleep_gaps["wake_time"] = pd.to_datetime(sleep_gaps["datetime"], errors='coerce')
    sleep_gaps = sleep_gaps.dropna(subset=["bedtime", "wake_time"])
    return sleep_gaps[["bedtime", "wake_time"]]

def build_features_for_each_day(garmin_df, all_usage, sleep_gaps):
    rows = []
    all_usage["datetime"] = pd.to_datetime(all_usage["datetime"], errors='coerce')
    for _, row in garmin_df.iterrows():
        d = row["Date"]
        bed_dt = row["BedtimeDT"]
        wake_dt = row["WakeTimeDT"]

        if wake_dt and not sleep_gaps.empty and pd.notnull(wake_dt):
            wake_time = pd.to_datetime(wake_dt)
            wake_date = wake_time.date()
            potential_gaps = sleep_gaps[
                sleep_gaps["wake_time"].apply(
                    lambda x: pd.to_datetime(x, errors='coerce').date() if pd.notnull(x) else None
                ) == wake_date
            ]
            if not potential_gaps.empty:
                time_diffs = potential_gaps["wake_time"].apply(
                    lambda x: abs(pd.to_datetime(x, errors='coerce') - wake_time).total_seconds() / 60
                    if pd.notnull(x) else float('inf')
                )
                min_idx = time_diffs.idxmin()
                if pd.notnull(min_idx):
                    pass
        else:
            wake_time = None

        if wake_dt and pd.notnull(wake_dt):
            wake_time = pd.to_datetime(wake_dt)
        else:
            wake_time = None

        # --- Bed Time: Always Use Garmin ---
        if bed_dt and pd.notnull(bed_dt):
            bed_time = pd.to_datetime(bed_dt)
        else:
            bed_time = None

        if bed_time:
            same_day_acts = all_usage[all_usage["datetime"].dt.date == bed_time.date()]
            day_activities_before_bed = same_day_acts[same_day_acts["datetime"] <= bed_time]
            if not day_activities_before_bed.empty:
                idx_max = day_activities_before_bed["datetime"].idxmax()
                bed_last_act = day_activities_before_bed.loc[idx_max, "datetime"]
                source_last = day_activities_before_bed.loc[idx_max, "source"]
                last_act_is_youtube = 1 if source_last == "YouTube" else 0
            else:
                bed_last_act = None
                last_act_is_youtube = 0
        else:
            bed_last_act = None
            last_act_is_youtube = 0

        if wake_time:
            same_day_acts_w = all_usage[all_usage["datetime"].dt.date == wake_time.date()]
            day_activities_after_wake = same_day_acts_w[same_day_acts_w["datetime"] >= wake_time]
            if not day_activities_after_wake.empty:
                wake_first_act = day_activities_after_wake["datetime"].min()
            else:
                wake_first_act = None
        else:
            wake_first_act = None

        weekend = 1 if d.weekday() >= 5 else 0
        rows.append({
            "Date": d,
            "bed_last_act": bed_last_act,
            "wake_first_act": wake_first_act,
            "weekend": weekend,
            "day_of_week": d.weekday(),
            "activity_count": same_day_acts.shape[0],
            "last_act_is_youtube": last_act_is_youtube,
            "BedtimeDT": bed_dt,
            "WakeTimeDT": wake_dt
        })
    feats_df = pd.DataFrame(rows)
    print("Features DataFrame shape:", feats_df.shape)
    return feats_df

def convert_dt_to_mod_and_date(dt):
    if pd.isnull(dt):
        return None, None
    local_dt = dt.astimezone(local_tz)
    mins = local_dt.hour * 60 + local_dt.minute
    day = local_dt.date()
    return day, mins

def compute_sleep_duration(bed_date, bed_mins, wake_date, wake_mins):
    if bed_date is None or wake_date is None or bed_mins is None or wake_mins is None:
        return None
    day_diff = (wake_date - bed_date).days
    return 1440 * day_diff + (wake_mins - bed_mins)

def cyclic_to_minutes(pred_array):
    angles = np.arctan2(pred_array[:,0], pred_array[:,1])
    angles = np.where(angles < 0, angles + 2*np.pi, angles)
    return angles * 1440 / (2 * np.pi)

def recenter_times(times, offset, day_minutes=1440):
    times = np.array(times)
    shifted = times - offset
    shifted = np.where(shifted > day_minutes/2, shifted - day_minutes, shifted)
    shifted = np.where(shifted < -day_minutes/2, shifted + day_minutes, shifted)
    return shifted

def make_time_formatter(offset, day_minutes=1440):
    def time_formatter(x, pos):
        absolute = (x + offset) % day_minutes
        hour = int(round(absolute/60)) % 24
        return f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"
    return time_formatter

def absolute_time_formatter(x, pos):
    hour = int(round(x/60)) % 24
    return f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"

def main():
    chrome_history_path = os.path.join("takeout", "chrome", "History.json")
    chrome_df = read_chrome_history(chrome_history_path)
    print("Chrome History read: ", chrome_df.shape)

    watch_history_path = os.path.join("takeout", "youtube", "history", "watch-history.json")
    search_history_path = os.path.join("takeout", "youtube", "history", "search-history.json")
    youtube_df = read_youtube_json(watch_history_path, search_history_path)
    print("Youtube History read: ", youtube_df.shape)

    garmin_path = "garmin.csv"
    garmin_df = read_garmin_data(garmin_path)
    print("Garmin Data read: ", garmin_df.shape)

    start_date = datetime(2024, 12, 2, tzinfo=local_tz)
    end_date = datetime(2025, 2, 21, 23, 59, 59, tzinfo=local_tz)

    all_usage = pd.concat([chrome_df[["datetime", "source"]], youtube_df[["datetime", "source"]]], ignore_index=True)
    all_usage["datetime"] = pd.to_datetime(all_usage["datetime"], errors='coerce')
    all_usage.dropna(subset=["datetime"], inplace=True)
    all_usage = all_usage[(all_usage["datetime"] >= start_date) & (all_usage["datetime"] <= end_date)]

    sleep_gaps = identify_sleep_gaps(all_usage)

    feats = build_features_for_each_day(garmin_df, all_usage, sleep_gaps)
    print("Feats after BedtimeDT/WakeTimeDT dropna:", feats.shape)
    feats.dropna(subset=["BedtimeDT", "WakeTimeDT"], inplace=True)
    print("Feats after all drops:", feats.shape)

    bed_dates = []
    bed_mins = []
    wake_dates = []
    wake_mins = []
    for i, row in feats.iterrows():
        b_day, b_mod = convert_dt_to_mod_and_date(row["BedtimeDT"])
        w_day, w_mod = convert_dt_to_mod_and_date(row["WakeTimeDT"])
        bed_dates.append(b_day)
        bed_mins.append(b_mod)
        wake_dates.append(w_day)
        wake_mins.append(w_mod)
    feats["bed_date"] = bed_dates
    feats["bed_mins"] = bed_mins
    feats["wake_date"] = wake_dates
    feats["wake_mins"] = wake_mins

    feats["bed_last_act_mins"] = feats["bed_last_act"].apply(lambda x: convert_dt_to_mod_and_date(x)[1] if x and pd.notnull(x) else None)
    feats["wake_first_act_mins"] = feats["wake_first_act"].apply(lambda x: convert_dt_to_mod_and_date(x)[1] if x and pd.notnull(x) else None)

    feats["weekend"] = feats["Date"].apply(lambda d: 1 if d.weekday() >= 5 else 0)
    feats["day_of_week"] = feats["Date"].apply(lambda x: x.weekday())

    activity_counts = all_usage.groupby(all_usage["datetime"].dt.date).size()
    feats["activity_count"] = feats["Date"].apply(lambda d: activity_counts.get(d, 0))

    feats_before_drop = feats.copy()
    feats.dropna(subset=["bed_mins", "wake_mins", "bed_last_act_mins", "wake_first_act_mins"], inplace=True)
    if feats.empty:
        print("Warning: No data remaining after dropping NaN values.")
        print("Data before dropping NaN:", feats_before_drop.shape)
        print("Missing values in key columns:")
        print(feats_before_drop.isnull().sum())
        return

    feats["bed_sin"] = np.sin(2 * np.pi * feats["bed_mins"] / 1440)
    feats["bed_cos"] = np.cos(2 * np.pi * feats["bed_mins"] / 1440)

    X_bed = feats[["bed_last_act_mins", "weekend", "day_of_week", "activity_count", "last_act_is_youtube"]].dropna()
    X_wake = feats[["wake_first_act_mins", "weekend", "day_of_week", "activity_count"]].dropna()
    y_bed = feats.loc[X_bed.index, ["bed_sin", "bed_cos"]]
    y_wake = feats.loc[X_wake.index, "wake_mins"]

    if len(X_bed) == 0 or len(X_wake) == 0:
        print("Error: No data available for model training after filtering. Check data quality.")
        return

    X_bed_train, X_bed_test, y_bed_train, y_bed_test = train_test_split(X_bed, y_bed, test_size=0.33, random_state=39)
    X_wake_train, X_wake_test, y_wake_train, y_wake_test = train_test_split(X_wake, y_wake, test_size=0.33, random_state=39)

    bed_model = RandomForestRegressor(n_estimators=1000, max_depth=20, random_state=39)
    bed_model.fit(X_bed_train, y_bed_train)

    wake_model = RandomForestRegressor(n_estimators=1000, max_depth=20, random_state=39)
    wake_model.fit(X_wake_train, y_wake_train)

    bed_pred_cyclic = bed_model.predict(X_bed_test)
    wake_pred = wake_model.predict(X_wake_test)

    bed_pred = cyclic_to_minutes(bed_pred_cyclic)
    y_bed_test_minutes = cyclic_to_minutes(y_bed_test.to_numpy())

    offset = np.median(y_bed_test_minutes)
    wrapped_actual = recenter_times(y_bed_test_minutes, offset)
    wrapped_pred = recenter_times(bed_pred, offset)

    bed_mae = mean_absolute_error(y_bed_test_minutes, bed_pred)
    bed_r2 = r2_score(y_bed_test_minutes, bed_pred)
    wake_mae = mean_absolute_error(y_wake_test, wake_pred)
    wake_r2 = r2_score(y_wake_test, wake_pred)
    print(f"Bedtime MAE: {bed_mae:.2f}, R-squared: {bed_r2:.2f}")
    print(f"Wake time MAE: {wake_mae:.2f}, R-squared: {wake_r2:.2f}")

    bedtime_formatter = FuncFormatter(make_time_formatter(offset))
    
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    ax1.scatter(wrapped_actual, wrapped_pred, alpha=0.6, label="Bedtime Predictions (wrapped)")
    min_val = -720
    max_val = 720
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_xlabel("Actual Bedtime")
    ax1.set_ylabel("Predicted Bedtime")
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_xticks(np.arange(min_val, max_val+1, 120))
    ax1.set_yticks(np.arange(min_val, max_val+1, 120))
    ax1.xaxis.set_major_formatter(bedtime_formatter)
    ax1.yaxis.set_major_formatter(bedtime_formatter)
    plt.legend()

    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    ax2.scatter(y_wake_test, wake_pred, alpha=0.6, label="Wake Predictions", color='orange')
    ax2.plot([0, 1440], [0, 1440], 'r--')
    ax2.set_xlabel("Actual Wake")
    ax2.set_ylabel("Predicted Wake")
    ax2.set_xlim(0, 1440)
    ax2.set_ylim(0, 1440)
    ax2.set_xticks(np.arange(0, 1440+1, 120))
    ax2.set_yticks(np.arange(0, 1440+1, 120))
    ax2.xaxis.set_major_formatter(FuncFormatter(absolute_time_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(absolute_time_formatter))
    plt.legend()

    plt.suptitle("Bedtime and Wake Time Predictions vs Actual")
    plt.show()

    feats = feats.reset_index(drop=True)
    bed_pred_full_cyclic = bed_model.predict(X_bed)
    wake_pred_full = wake_model.predict(X_wake)
    bed_pred_full = cyclic_to_minutes(bed_pred_full_cyclic)
    predicted_duration = []
    for i in range(len(feats)):
        b_min = bed_pred_full[i]
        w_min = wake_pred_full[i]
        b_date = feats.at[i, "bed_date"]
        w_date = feats.at[i, "wake_date"]
        dur = compute_sleep_duration(b_date, b_min, w_date, w_min)
        predicted_duration.append(dur)
    feats["predicted_duration"] = predicted_duration

    feats["actual_sleep_duration"] = feats.apply(
        lambda row: compute_sleep_duration(row["bed_date"], row["bed_mins"], row["wake_date"], row["wake_mins"]), axis=1)

    actual_dur = feats["actual_sleep_duration"]
    pred_dur = feats["predicted_duration"]
    valid_mask = (~actual_dur.isna()) & (~pd.Series(pred_dur).isna()) & (actual_dur >= 0) & (pd.Series(pred_dur) >= 0)
    actual_dur = actual_dur[valid_mask]
    pred_dur = pd.Series(pred_dur)[valid_mask]
    dur_mae = mean_absolute_error(actual_dur, pred_dur)
    print(f"Sleep Duration MAE: {dur_mae:.2f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(actual_dur, pred_dur, alpha=0.6)
    plt.plot([0, 1000], [0, 1000], 'r--')
    plt.xlabel("Actual Sleep Duration (min)")
    plt.ylabel("Predicted Sleep Duration (min)")
    plt.title("Sleep Duration Prediction vs Actual")
    plt.show()
    
    ts = feats.copy().reset_index(drop=True)
    ts_bed_pred = cyclic_to_minutes(bed_model.predict(ts[["bed_last_act_mins", "weekend", "day_of_week", "activity_count", "last_act_is_youtube"]]))
    ts_wake_pred = wake_model.predict(ts[["wake_first_act_mins", "weekend", "day_of_week", "activity_count"]])
    plt.figure(figsize=(14, 6))
    plt.subplot(1,2,1)
    plt.plot(ts["bed_date"], ts["bed_mins"], marker="o", label="Actual Bedtime")
    plt.plot(ts["bed_date"], ts_bed_pred, marker="x", label="Predicted Bedtime")
    plt.title("Time Series of Bedtimes vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Bedtime")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(absolute_time_formatter))
    plt.gcf().autofmt_xdate()
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(ts["wake_date"], ts["wake_mins"], marker="o", label="Actual Wake Time")
    plt.plot(ts["wake_date"], ts_wake_pred, marker="x", label="Predicted Wake Time")
    plt.title("Time Series of Wake Times vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Wake Time")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(absolute_time_formatter))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    sleep_duration = feats["actual_sleep_duration"].dropna()
    bedtimes = feats["bed_mins"].dropna()
    wake_times = feats["wake_mins"].dropna()
    plt.figure(figsize=(27, 8))
    plt.subplot(1, 3, 1)
    plt.hist(sleep_duration, bins=20, color="purple", edgecolor="black")
    plt.title("Distribution of Sleep Duration")
    plt.xlabel("Sleep Duration (minutes)")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 3, 2)
    plt.hist(bedtimes, bins=24, color="skyblue", edgecolor="black")
    plt.title("Distribution of Bedtimes")
    plt.xlabel("Bedtime")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(absolute_time_formatter))
    plt.xticks(np.linspace(0,1440,13))
    
    plt.subplot(1, 3, 3)
    plt.hist(wake_times, bins=24, color="lightgreen", edgecolor="black")
    plt.title("Distribution of Wake Times")
    plt.xlabel("Wake Time")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(absolute_time_formatter))
    plt.xticks(np.linspace(0,1440,13))
    
    plt.tight_layout()
    plt.show()
    
    descriptive_labels = {
        "bed_last_act_mins": "Last Activity Time",
        "weekend": "Weekend",
        "day_of_week": "Day of Week",
        "activity_count": "Activity Count",
        "last_act_is_youtube": "Last Activity: YouTube?",
        "wake_first_act_mins": "First Activity Time"
    }
    features_bed = X_bed.columns.tolist()
    importances_bed = bed_model.feature_importances_
    indices_bed = np.argsort(importances_bed)[::-1]
    
    features_wake = X_wake.columns.tolist()
    importances_wake = wake_model.feature_importances_
    indices_wake = np.argsort(importances_wake)[::-1]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Feature Importance - Bedtime Model", fontsize=14)
    plt.bar(range(len(features_bed)), importances_bed[indices_bed], color="skyblue", align="center")
    plt.xticks(range(len(features_bed)), [descriptive_labels[features_bed[i]] for i in indices_bed], rotation=45, fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    
    plt.subplot(1, 2, 2)
    plt.title("Feature Importance - Wake Time Model", fontsize=14)
    plt.bar(range(len(features_wake)), importances_wake[indices_wake], color="lightgreen", align="center")
    plt.xticks(range(len(features_wake)), [descriptive_labels[features_wake[i]] for i in indices_wake], rotation=45, fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
