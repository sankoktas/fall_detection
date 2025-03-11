import numpy as np
import pandas as pd
import json

class NoNaNJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder that replaces NaN/Inf with None."""
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None  # This becomes 'null' in final JSON
        return super().default(obj)

def create_labelstudio_timeseries_json(input_csv, output_json):
    # 1) Read CSV
    df = pd.read_csv(input_csv)
    
    # 2) Replace "*" with NaN
    df.replace("*", np.nan, inplace=True)
    
    # 3) Convert all NaN -> None
    df = df.where(pd.notnull(df), None)
    
    # 4) Convert columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # 5) Build column-based dict: each column name -> array of values
    #    This is the recommended approach for 'valueType="json"'
    #    in Label Studio's <TimeSeries> tag.
    timeseries_dict = {}
    for col in df.columns:
        # Each column is a list of values
        timeseries_dict[col] = df[col].tolist()
    
    # 6) Wrap in a single-task structure
    #    If you want multiple tasks, you'd create multiple objects in the 'tasks' list.
    task = {
        "data": {
            "tsData": timeseries_dict
        }
    }
    
    tasks = [task]
    
    # 7) Dump to JSON with our custom encoder
    json_str = json.dumps(tasks, indent=2, cls=NoNaNJSONEncoder)
    # 8) Post-process any leftover 'NaN' tokens (defensive measure)
    json_str = json_str.replace("NaN", "null")
    
    # 9) Write final JSON to file
    with open(output_json, "w") as f:
        f.write(json_str)

# Example usage:
if __name__ == "__main__":
    create_labelstudio_timeseries_json(
        input_csv="newDataMerged.csv",
        output_json="newInput.json"
    )
