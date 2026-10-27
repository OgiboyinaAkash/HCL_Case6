import os
import pandas as pd
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def export_split(dataset_dict, name, max_rows_for_csv=20000):
    folder = os.path.join(DATA_DIR, name)
    os.makedirs(folder, exist_ok=True)
    print(f"\nProcessing '{name}'...")
    for split_name in dataset_dict.keys():
        split = dataset_dict[split_name]
        df = split.to_pandas()
        
        # Save Parquet (efficient, preserves types)
        parquet_path = os.path.join(folder, f"{split_name}.parquet")
        df.to_parquet(parquet_path, index=False)
        
        # Save CSV (up to max_rows_for_csv to keep files snappy and easy to inspect)
        csv_path = os.path.join(folder, f"{split_name}.csv")
        if len(df) > max_rows_for_csv:
            df.head(max_rows_for_csv).to_csv(csv_path, index=False)
            print(f"  [+] {split_name}: {len(df)} rows -> {parquet_path} (Full) & {csv_path} (First {max_rows_for_csv} rows)")
        else:
            df.to_csv(csv_path, index=False)
            print(f"  [+] {split_name}: {len(df)} rows -> {parquet_path} & {csv_path}")

print("=" * 60)
print("DOWNLOADING & EXPORTING PUBLIC BENCHMARKS FOR ROUTING & CACHING")
print("=" * 60)

# 1. GSM8K (Math reasoning queries of varying difficulty)
print("\n[1/4] Loading GSM8K (openai/gsm8k)...")
gsm8k = load_dataset("openai/gsm8k", "main")
export_split(gsm8k, "gsm8k")

# 2. MMLU (Multi-domain multiple choice QA across 57 subjects)
print("\n[2/4] Loading MMLU All (cais/mmlu)...")
mmlu = load_dataset("cais/mmlu", "all")
export_split(mmlu, "mmlu")

# 3. RouteLLM - MMLU Battles (Model comparison battles for router training/eval)
print("\n[3/4] Loading RouteLLM MMLU Battles (routellm/mmlu_battles)...")
route_mmlu = load_dataset("routellm/mmlu_battles")
export_split(route_mmlu, "routellm_mmlu_battles")

# 4. RouteLLM - GPT-4 Dataset (Query routing benchmark with scores)
print("\n[4/4] Loading RouteLLM GPT4 Dataset (routellm/gpt4_dataset)...")
route_gpt4 = load_dataset("routellm/gpt4_dataset")
export_split(route_gpt4, "routellm_gpt4_dataset")

print("\n" + "=" * 60)
print("ALL DATASETS SUCCESSFULLY DOWNLOADED & EXPORTED TO:", os.path.abspath(DATA_DIR))
print("=" * 60)

