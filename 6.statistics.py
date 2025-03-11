import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def main():
    # PARAMETERS
    WINDOW_SIZE = 0.150  # 150 ms window
    STEP_SIZE   = 0.120  # 120 ms hop => 30 ms overlap
    INPUT_CSV   = "data/mar_11_timeseries_data.csv"
    OUTPUT_CSV  = "data/mar_11_features_3class.csv"

    # 1) LOAD DATA
    df = pd.read_csv(INPUT_CSV)

    # 1a) DROP ROWS WITH ALL SENSOR COLUMNS MISSING
    sensor_cols = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']
    df = df.dropna(subset=sensor_cols, how='all')

    # Ensure timestamps are floats
    df['timestamp_s'] = df['timestamp_s'].astype(float)

    # Sort by timestamp just in case
    df.sort_values(by='timestamp_s', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # For convenience, define a function for magnitude
    def magnitude(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    # Create two new columns for ACC magnitude and GYRO magnitude
    df['acc_mag'] = magnitude(df['accel_x'], df['accel_y'], df['accel_z'])
    df['gyro_mag'] = magnitude(df['gyro_x'], df['gyro_y'], df['gyro_z'])

    # 2) CREATE TIME WINDOWS
    start_time = df['timestamp_s'].min()
    end_time   = df['timestamp_s'].max()
    
    current_start = start_time
    windowed_data = []

    while current_start < end_time:
        current_end = current_start + WINDOW_SIZE

        # 3) SLICE THE DATA FOR THIS WINDOW
        mask = (df['timestamp_s'] >= current_start) & (df['timestamp_s'] < current_end)
        df_window = df[mask]

        if len(df_window) > 0:
            print(f"Window from {current_start} to {current_end} has {len(df_window)} rows.")

            acc_vals = df_window['acc_mag'].values
            gyro_vals = df_window['gyro_mag'].values

            # Mean
            acc_mean  = np.mean(acc_vals)
            gyro_mean = np.mean(gyro_vals)

            # Variance
            acc_var   = np.var(acc_vals)
            gyro_var  = np.var(gyro_vals)

            # Kurtosis
            acc_kurt  = kurtosis(acc_vals)
            gyro_kurt = kurtosis(gyro_vals)

            # Skewness
            acc_skew  = skew(acc_vals)
            gyro_skew = skew(gyro_vals)

            # Median
            acc_median  = np.median(acc_vals)
            gyro_median = np.median(gyro_vals)

            # Percentiles (10%, 25%, 50%, 75%)
            acc_p10, acc_p25, acc_p50, acc_p75 = np.percentile(acc_vals, [10, 25, 50, 75])
            gyro_p10, gyro_p25, gyro_p50, gyro_p75 = np.percentile(gyro_vals, [10, 25, 50, 75])

            # Determine label
            labels_in_window = df_window['label'].unique()
            if 'FALL' in labels_in_window:
                window_label = 'FALL'
            else:
                window_label = df_window['label'].value_counts().idxmax()

            row_dict = {
                'start_time': current_start,
                'end_time': current_end,
                'acc_mean': acc_mean,
                'acc_var': acc_var,
                'acc_kurtosis': acc_kurt,
                'acc_skewness': acc_skew,
                'acc_median': acc_median,
                'acc_p10': acc_p10,
                'acc_p25': acc_p25,
                'acc_p50': acc_p50,
                'acc_p75': acc_p75,
                'gyro_mean': gyro_mean,
                'gyro_var': gyro_var,
                'gyro_kurtosis': gyro_kurt,
                'gyro_skewness': gyro_skew,
                'gyro_median': gyro_median,
                'gyro_p10': gyro_p10,
                'gyro_p25': gyro_p25,
                'gyro_p50': gyro_p50,
                'gyro_p75': gyro_p75,
                'label': window_label
            }
            windowed_data.append(row_dict)
        else:
            print(f"No data for window {current_start} - {current_end}")

        current_start += STEP_SIZE

    # Convert to DataFrame
    features_df = pd.DataFrame(windowed_data)

    # 7) DROP ANY ROW WHERE ALL FEATURE COLUMNS ARE NaN
    #    (This cleans out lines with no actual stats.)
    feature_cols = [
        'acc_mean','acc_var','acc_kurtosis','acc_skewness','acc_median','acc_p10','acc_p25','acc_p50','acc_p75',
        'gyro_mean','gyro_var','gyro_kurtosis','gyro_skewness','gyro_median','gyro_p10','gyro_p25','gyro_p50','gyro_p75'
    ]
    features_df.dropna(subset=feature_cols, how='all', inplace=True)

    # 8) SAVE THE FINAL FEATURE DATA
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Feature extraction complete. Saved to {OUTPUT_CSV}")
    print(features_df.head())

if __name__ == "__main__":
    main()
