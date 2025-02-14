import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    """Parse a log file to extract execution time, matrix size, operation type, and execution type."""
    with open(filepath, 'r') as file:
        content = file.readlines()
    
    filename = os.path.basename(filepath)
    match = re.match(r'(\d{8}_\d{6})_(\w+)_([a-zA-Z]+)_([\dx]+)\.log', filename)
    
    if not match:
        return None
    
    timestamp, operation, value_type, matrix_size = match.groups()
    execution_type = filepath.split(os.sep)[-3]  # Adjusted to match new directory structure
    execution_time = None
    
    for line in content:
        if "Elapsed (wall clock) time" in line:
            time_match = re.search(r'([0-9]+):([0-9]+\.[0-9]+)', line)
            if time_match:
                minutes, seconds = map(float, time_match.groups())
                execution_time = minutes * 60 + seconds
                break
        elif "Addition terminée en" in line:
            time_match = re.search(r'([0-9]+\.?[0-9]*) ms', line)
            if time_match:
                execution_time = float(time_match.group(1)) / 1000  # Convert ms to seconds
                break
    
    if execution_time is None or execution_time == 0:
        return None  # Ignore invalid or zero execution times
    
    return {
        "Timestamp": timestamp,
        "Operation": operation,
        "Value Type": value_type,
        "Matrix Size": matrix_size,
        "Execution Type": execution_type,
        "Execution Time (s)": execution_time
    }

def process_logs(log_dir):
    """Walk through the log directory, parse files, and collect data."""
    data = []
    
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                filepath = os.path.join(root, file)
                parsed_data = parse_log_file(filepath)
                if parsed_data:
                    data.append(parsed_data)
    
    return pd.DataFrame(data)

def plot_execution_time(df):
    """Generate plots for each timestamp and operation with float and int data separately."""
    df = df.dropna()  # Remove NaN values
    df["Matrix Size"] = df["Matrix Size"].apply(lambda x: int(x.split('x')[0]))
    df.sort_values(by="Matrix Size", inplace=True)
    
    for timestamp in df["Timestamp"].unique():
        subset_timestamp = df[df["Timestamp"] == timestamp]
        for operation in subset_timestamp["Operation"].unique():
            subset_operation = subset_timestamp[subset_timestamp["Operation"] == operation]
            for value_type in ["float", "int"]:
                subset = subset_operation[subset_operation["Value Type"] == value_type]
                
                if subset.empty:
                    continue
                
                plt.figure(figsize=(12, 6))
                for exec_type in subset["Execution Type"].unique():
                    exec_subset = subset[subset["Execution Type"] == exec_type]
                    plt.plot(exec_subset["Matrix Size"], exec_subset["Execution Time (s)"], marker='o', linestyle='-', markersize=5, alpha=0.7, label=exec_type)
                
                plt.xlabel("Matrix Size")
                plt.ylabel("Execution Time (s)")
                plt.title(f"Execution Time vs Matrix Size ({operation} - {value_type}) - {timestamp}")
                plt.yscale("log")  # Apply logarithmic scale for better visibility
                plt.legend(loc='best')
                plt.grid()
                plt.savefig(f"execution_time_{timestamp}_{operation}_{value_type}.png")

# Exemple d'utilisation
df_logs = process_logs("../../../res/raw")
if not df_logs.empty:
    plot_execution_time(df_logs)
