#!/usr/bin/env python3
import pandas as pd

def main():
    # >>> 1) CONFIGURE FILE PATHS <<<
    # Provide the path to your input CSV file here:
    input_csv = "newData.csv"
    # Output CSV file (will be created or overwritten):
    output_csv = "newDataMerged.csv"

    # >>> 2) READ THE CSV <<<
    # We assume the CSV has these columns among others:
    # sensortime_wakeup_ns, sensortime_nonwakeup_ns
    df = pd.read_csv(input_csv)

    # >>> 3) MERGE THE TIMESTAMPS <<<
    # If 'sensortime_wakeup_ns' is "*", we use 'sensortime_nonwakeup_ns'.
    # Otherwise, we keep 'sensortime_wakeup_ns'.
    df["timestamp_ns"] = df["sensortime_wakeup_ns"].mask(
        df["sensortime_wakeup_ns"] == "*",
        df["sensortime_nonwakeup_ns"]
    )

    # Convert to numeric if needed (some rows might be "*")
    # 'coerce' turns invalid strings into NaN
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")

    # >>> 4) OPTIONAL: CONVERT NANOS TO SECONDS <<<
    # If you want a human-friendly version in seconds:
    df["timestamp_s"] = df["timestamp_ns"] / 1e9

    # >>> 5) WRITE TO A NEW CSV <<<
    df.to_csv(output_csv, index=False)
    print(f"Merged timestamps saved to {output_csv}")

if __name__ == "__main__":
    main()
