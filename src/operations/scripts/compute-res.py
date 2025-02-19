import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(filepath, timestamp, operation, execution_type):
    """Parse a log file to extract execution time, matrix size, operation type, and execution type."""
    with open(filepath, 'r') as file:
        content = file.readlines()
    
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+x\d+)_([a-zA-Z]+)\.log', filename)
    
    if not match:
        return None
    
    matrix_size, value_type = match.groups()
    execution_time = None
    
    for line in content:
        if "Elapsed (wall clock) time" in line:
            time_match = re.search(r'([0-9]+):([0-9]+\.[0-9]+)', line)
            if time_match:
                minutes, seconds = map(float, time_match.groups())
                execution_time = minutes * 60 + seconds
                break
        elif "terminé" in line or "terminée" in line:
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
    
    for root, dirs, _ in os.walk(log_dir):
        for operation_folder in dirs:
            operation_path = os.path.join(root, operation_folder)
            operation_parts = operation_folder.split('_', 1)
            if len(operation_parts) < 2:
                continue  # Skip folders that don't match expected pattern
            timestamp, operation = operation_parts
            
            for exec_type in os.listdir(operation_path):
                exec_path = os.path.join(operation_path, exec_type, "log")
                if os.path.exists(exec_path):
                    for file in os.listdir(exec_path):
                        if file.endswith(".log"):
                            filepath = os.path.join(exec_path, file)
                            parsed_data = parse_log_file(filepath, timestamp, operation, exec_type)
                            if parsed_data:
                                data.append(parsed_data)
    
    return pd.DataFrame(data)

def plot_execution_time(df):
    """Generate plots for each timestamp and operation with float and int data separately."""
    df = df.dropna()  # Remove NaN values
    df["Matrix Size"] = df["Matrix Size"].apply(lambda x: int(x.split('x')[0]))
    df.sort_values(by="Matrix Size", inplace=True)
    
    os.makedirs('./res/graphs', exist_ok=True)
    
    # Palette optimisée pour daltoniens avec forte distinction des couleurs
    exec_styles = {
        "cuda": ("#E6194B", "o"),  # Rouge foncé
        "opencl": ("#3CB44B", "s"),  # Vert standard
        "cpu_opti_O3": ("#FF8000", "D"),  # Orange vif
        "cpu_opti_O2": ("#4363D8", "^"),  # Bleu intense
        "cpu": ("#800080", "v")  # Violet foncé
    }
    
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
                    if exec_type in exec_styles:
                        color, marker = exec_styles[exec_type]
                    else:
                        color, marker = "#911eb4", "*"  # Violet par défaut
                    
                    exec_subset = subset[subset["Execution Type"] == exec_type]
                    plt.plot(
                        exec_subset["Matrix Size"], 
                        exec_subset["Execution Time (s)"], 
                        marker=marker, 
                        linestyle='-', 
                        markersize=8,  # Augmentation de la taille des points
                        alpha=0.9, 
                        linewidth=3,  # Épaississement des lignes
                        color=color, 
                        label=exec_type
                    )
                
                plt.xlabel("Matrix Size")
                plt.ylabel("Execution Time (s)")
                plt.title(f"Execution Time {timestamp} {operation} ({value_type})")
                plt.yscale("log")  # Apply logarithmic scale for better visibility
                plt.legend(loc='best')
                plt.grid()
                plt.savefig(f"./res/graphs/execution_time_{timestamp}_{operation}_{value_type}.png")

# Exemple d'utilisation
df_logs = process_logs("./res/raw")
if not df_logs.empty:
    plot_execution_time(df_logs)
