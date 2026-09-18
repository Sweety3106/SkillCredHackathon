import pandas as pd
import os
from config import PATHS, SENSOR_COLS

def main():
    # Read NASA C-MAPSS FD001
    # Columns: unit, cycle, op1, op2, op3, s1...s21
    cols = ["machine_id", "timestamp", "op1", "op2", "op3"] + SENSOR_COLS
    
    # Path to the real dataset uploaded by user (relative to where the script is run from usually)
    # We will run it from machine-early-warning folder
    data_path = "../archive (1)/train_FD001.txt"
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}")
        return
        
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=cols)
    
    # calculate rul = max(cycle for that unit) - cycle
    max_cycles = df.groupby('machine_id')['timestamp'].max().reset_index()
    max_cycles.rename(columns={'timestamp': 'max_cycle'}, inplace=True)
    df = df.merge(max_cycles, on='machine_id')
    df['rul'] = df['max_cycle'] - df['timestamp']
    df.drop('max_cycle', axis=1, inplace=True)
    
    # failure_event: 1 only on the very last row of a machine that failed, else 0
    df['failure_event'] = (df['rul'] == 0).astype(int)
    
    # drop op1, op2, op3 as they are not used based on IF-01 schema
    df = df.drop(columns=["op1", "op2", "op3"])
    
    os.makedirs(os.path.dirname(PATHS['raw']), exist_ok=True)
    df.to_parquet(PATHS['raw'], index=False)
    print(f"Real data loaded and saved to {PATHS['raw']}")
    print(f"Shape: {df.shape}, Machines: {df['machine_id'].nunique()}")

if __name__ == "__main__":
    main()
