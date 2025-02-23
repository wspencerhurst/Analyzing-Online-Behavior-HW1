#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from dateutil import parser as date_parser
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def read_chrome_history(chrome_history_path):
    with open(chrome_history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history_records = data.get("Browser History", [])
    rows = []
    for record in history_records:
        title = record.get("title", "")
        url = record.get("url", "")
        time_usec = record.get("time_usec", None)

        if time_usec is not None:
            epoch_start = datetime(1601, 1, 1) # chrome epoch start
            dt = epoch_start + timedelta(microseconds=int(time_usec))
        else:
            dt = None

        rows.append({
            "title": title,
            "url": url,
            "datetime": dt
        })

    df = pd.DataFrame(rows)
    df.dropna(subset=["datetime"], inplace=True) # drop rows with no datetime
    return df

def read_youtube_history(youtube_history_path):
    #poop
     









def main():
    chrome_history_path = os.path.join("takeout", "chrome", "History.json")
    chrome_history = read_chrome_history(chrome_history_path)

    youtube_history_path = os.path.join("takeout", "youtube", "history", "watch-history.json")

if __name__ == "__main__":
    main()
