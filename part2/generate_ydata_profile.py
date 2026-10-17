import os
import pandas as pd
from ydata_profiling import ProfileReport

# 1. LOCATE DATASET
csv_paths = [
    "part2/dataset/predictive_maintenance.csv",
    "dataset/predictive_maintenance.csv",
    "../part2/dataset/predictive_maintenance.csv",
    "predictive_maintenance.csv"
]
data_path = next((p for p in csv_paths if os.path.exists(p)), None)

if data_path is None:
    raise FileNotFoundError("Could not locate 'predictive_maintenance.csv'.")

print(f"--- Loading Dataset from: {data_path} ---")
df = pd.read_csv(data_path)

# 2. GENERATE PROFILE REPORT
print("--- Generating ydata-profiling Report (this may take a minute)... ---")
profile = ProfileReport(
    df,
    title="Predictive Maintenance Dataset - Exploratory Profiling Report",
    explorative=True
)

# 3. SAVE HTML REPORT
out_dir = os.path.dirname(data_path)
output_report_path = os.path.join(out_dir, "predictive_maintenance_report.html")

profile.to_file(output_report_path)
print(f"\nSuccessfully generated profiling report at: '{output_report_path}'")