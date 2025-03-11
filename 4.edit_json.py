#!/usr/bin/env python3
import json

def remove_first_n_values(input_json, output_json, skip_count=1871):
    """
    Removes the first `skip_count` values from each array in the 'tsData' dictionary
    of the first (or each) task in the Label Studio time-series JSON file.
    """

    # 1) Read the JSON file
    with open(input_json, "r") as f:
        tasks = json.load(f)  # tasks is a list of task objects

    # 2) For each task in the file:
    #    tasks[i]["data"]["tsData"] should be a dict of columns -> array of values
    for i, task in enumerate(tasks):
        ts_data = task.get("data", {}).get("tsData", {})
        if not ts_data:
            # If a task doesn't have tsData, skip or do something else
            continue

        # 3) For each column in tsData, remove the first `skip_count` items
        for column_name, values in ts_data.items():
            # Ensure it's a list
            if isinstance(values, list):
                # Slice off the first `skip_count` elements
                ts_data[column_name] = values[skip_count:]
            else:
                # If it's not a list, skip or handle differently
                pass

        # Put the modified ts_data back into the task
        tasks[i]["data"]["tsData"] = ts_data

    # 4) Write the updated tasks to output_json
    with open(output_json, "w") as f:
        json.dump(tasks, f, indent=2)

# Usage example:
if __name__ == "__main__":
    input_file = "newInput.json"
    output_file = "newOutput.json"
    remove_first_n_values(input_file, output_file, skip_count=4)
