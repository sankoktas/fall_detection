#!/usr/bin/env python3

import csv
import sys
import json

def convert_labelstudio_tsdata_with_timeserieslabels(input_csv, output_csv):
    """
    Reads a Label Studio exported CSV with:
      - 'tsData' -> dictionary-of-arrays for sensor data
      - 'tsLabels' -> array of intervals, each like:
           {
             "start": float,
             "end": float,
             "instant": bool,
             "timeserieslabels": ["FALL"] or ["MOTION"] or ["NO_MOTION"]
           }

    Then merges sensor data into row-based format (accel_x..timestamp_s)
    and assigns label among {FALL, MOTION, NO_MOTION}.
      - If no interval contains the timestamp, label = "NO_MOTION" by default.
      - If an interval includes the timestamp and 'timeserieslabels' includes "FALL",
        the label is "FALL" (highest priority).
      - Else if it includes "MOTION", label = "MOTION".
      - Else if it includes "NO_MOTION", label = "NO_MOTION".

    Output columns:
      accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, timestamp_s, label
    """

    # Increase field size limit for large JSON in tsData/tsLabels columns
    csv.field_size_limit(sys.maxsize)

    with open(input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        all_rows = []

        for row in reader:
            ts_str = row.get('tsData', '')
            if not ts_str:
                # No sensor data in this row, skip
                continue

            # Parse the sensor dictionary-of-arrays
            try:
                ts_dict = json.loads(ts_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse tsData JSON. Skipping. Error: {e}")
                continue

            # Parse intervals from 'tsLabels'
            intervals_str = row.get('tsLabels', '[]')
            try:
                intervals = json.loads(intervals_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse tsLabels JSON. Using empty intervals. Error: {e}")
                intervals = []

            # Expected sensor keys
            sensor_keys = ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z","timestamp_s"]
            missing = [k for k in sensor_keys if k not in ts_dict]
            if missing:
                print(f"Warning: Missing keys {missing} in tsData. Skipping row.")
                continue

            # Convert dictionary-of-arrays -> row-based
            length = len(ts_dict["timestamp_s"])
            for i in range(length):
                row_data = {}
                for k in sensor_keys:
                    arr = ts_dict[k]
                    row_data[k] = arr[i] if i < len(arr) else None

                # Determine the label (FALL, MOTION, or NO_MOTION)
                row_data["label"] = determine_label_timeserieslabels(
                    row_data["timestamp_s"],
                    intervals
                )
                all_rows.append(row_data)

    # Write final CSV
    out_cols = sensor_keys + ["label"]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(all_rows)

def determine_label_timeserieslabels(timestamp, intervals):
    """
    For each interval object, we have:
      {
        "start": float,
        "end": float,
        "instant": bool,
        "timeserieslabels": ["FALL"] or ["MOTION"] or ["NO_MOTION"]
      }

    Priority logic within [start, end):
      1) If 'FALL' is present in timeserieslabels, label = FALL
      2) else if 'MOTION' is present, label = MOTION
      3) else if 'NO_MOTION' is present, label = NO_MOTION
      4) If no interval contains timestamp, label = "NO_MOTION" (default)
    """
    if timestamp is None:
        return "NO_MOTION"

    for interval in intervals:
        start = interval.get("start")
        end = interval.get("end")
        ts_labels = interval.get("timeserieslabels", [])

        # Check if timestamp falls into [start, end)
        if (start is not None) and (end is not None):
            if start <= timestamp < end:
                # Convert all labels to uppercase
                labels_upper = [lbl.upper() for lbl in ts_labels]

                # Priority: FALL > MOTION > NO_MOTION
                if "FALL" in labels_upper:
                    return "FALL"
                elif "MOTION" in labels_upper:
                    return "MOTION"
                elif "NO MOTION" in labels_upper:
                    return "NO MOTION" 
                else:
                    return "MOTION"  # if an interval is found but no recognized label

    # If no interval matched the timestamp, default to "MOTION"
    return "MOTION"

if __name__ == "__main__":
    input_file = "data/mar_11.csv"                   # Label Studio export with 'FALL','MOTION','NO_MOTION'
    output_file = "data/mar_11_timeseries_data.csv"  # row-based CSV with label
    convert_labelstudio_tsdata_with_timeserieslabels(input_file, output_file)
    print(f"Done! Wrote row-based CSV with label to '{output_file}'")