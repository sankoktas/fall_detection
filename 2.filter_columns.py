import pandas as pd

def main():
    input_csv = "newDataMerged.csv"
    output_csv = "newDataMerged_Filtered.csv"

    df = pd.read_csv(input_csv)

    # Define the columns you want to keep in the final CSV
    columns_to_keep = [
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "timestamp_s"
    ]

    # Filter the DataFrame to only these columns
    df_filtered = df[columns_to_keep]

    # Save the filtered DataFrame to a new CSV
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved to {output_csv}")

if __name__ == "__main__":
    main()
