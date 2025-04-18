import pandas as pd
import os
import time

# Simulate some input parameters and performance metrics
params = {
    "Model": "TestModel",
    "Dataset": "SampleDataset",
    "BatchSize": 16
}

# Simulated metrics
avg_wer = 2
eer = 0.045
start = time.time()
time.sleep(1.2)  # Simulate some processing time
end = time.time()

# Create a DataFrame with the new result
new_result = pd.DataFrame([{**params, "WER": avg_wer, "EER": eer, "Runtime (s)": end - start}])

# Check if file exists
file_path = "results.csv"
file_exists = os.path.isfile(file_path)

# Append to CSV, add header only if file does not exist
new_result.to_csv(file_path, mode='a', header=not file_exists, index=False)

print(f"Result added to {file_path}:")
print(new_result)
