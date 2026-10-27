"""
datacleaning.py
Combines, cleans, and labels queries from GSM8K, MMLU, and RouteLLM into a unified dataset/database.
Assigns each query to an appropriate model tier:
  - 0: 'small'  (e.g., Llama-3-8B, Mistral-7B, GPT-3.5-Turbo - fast & cheap for simple factual/1-2 step queries)
  - 1: 'medium' (e.g., Mixtral 8x7B, Claude 3 Haiku, Llama-3-70B - balanced for multi-step reasoning/intermediate QA)
  - 2: 'large'  (e.g., GPT-4o, Claude 3.5 Sonnet - complex STEM, multi-step math proofs, abstract logic)

Outputs:
  - SQLite Database: data/combined_queries.db (table: queries)
  - Parquet:         data/combined_queries.parquet
  - CSV:             data/combined_queries.csv
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_DB = os.path.join(DATA_DIR, "combined_queries.db")
OUTPUT_PARQUET = os.path.join(DATA_DIR, "combined_queries.parquet")
OUTPUT_CSV = os.path.join(DATA_DIR, "combined_queries.csv")
ROOT_CLEANED_CSV = "cleaned_data.csv"

LABEL_MAP = {"small": 0, "medium": 1, "large": 2}


# =====================================================================
# 1. Text Cleaning & Normalization
# =====================================================================
def clean_text(text: str) -> str:
    """Cleans encoding artifacts, trims excess whitespace, and normalizes quotes."""
    if not isinstance(text, str):
        return ""
    
    # Common Unicode & mojibake replacements
    replacements = {
        "\ufffd": "'",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€”": "-",
        "â€“": "-",
        "\xa0": " ",
        "\r\n": "\n",
        "\r": "\n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse repeated horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =====================================================================
# 2. Process GSM8K (Math Word Problems)
# =====================================================================
def load_and_clean_gsm8k() -> pd.DataFrame:
    """
    GSM8K queries vary by reasoning steps (marked by <<...>> in GSM8K answers).
      <= 2 calculation steps -> 'small'
      3-4 calculation steps  -> 'medium'
      >= 5 calculation steps -> 'large'
    """
    records = []
    for split in ["train", "test"]:
        path = os.path.join(DATA_DIR, "gsm8k", f"{split}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            q = clean_text(row["question"])
            ans = clean_text(row["answer"])
            if not q or not ans:
                continue

            # Number of reasoning calculation steps
            calc_steps = ans.count("<<")
            if calc_steps <= 2:
                tier = "small"
                reason = f"Simple math arithmetic ({calc_steps} calculation steps)"
            elif calc_steps <= 4:
                tier = "medium"
                reason = f"Moderate multi-step word problem ({calc_steps} calculation steps)"
            else:
                tier = "large"
                reason = f"Complex multi-step math reasoning ({calc_steps} calculation steps)"

            records.append({
                "query": q,
                "answer": ans,
                "model_tier": tier,
                "class_label": LABEL_MAP[tier],
                "source": "gsm8k",
                "complexity_reason": reason
            })

    print(f"[GSM8K] Processed {len(records)} queries.")
    return pd.DataFrame(records)


# =====================================================================
# 3. Process MMLU (Academic Multi-Choice Questions)
# =====================================================================
# Categorize MMLU subjects into difficulty tiers
MMLU_TIER_MAP = {
    # Small: Elementary & straightforward high school / factual humanities
    "small": [
        "elementary_mathematics", "high_school_geography", "high_school_government_and_politics",
        "high_school_psychology", "high_school_us_history", "high_school_world_history",
        "marketing", "management", "nutrition", "pop_culture", "miscellaneous",
        "world_religions", "moral_scenarios", "global_facts", "human_aging",
        "high_school_european_history"
    ],
    # Large: Advanced college STEM, advanced math/logic, professional law & medicine
    "large": [
        "abstract_algebra", "college_mathematics", "college_physics", "college_computer_science",
        "college_chemistry", "formal_logic", "econometrics", "machine_learning",
        "professional_law", "professional_medicine", "clinical_knowledge",
        "medical_genetics", "virology", "electrical_engineering"
    ]
    # Any other subject defaults to 'medium'
}

def load_and_clean_mmlu() -> pd.DataFrame:
    """Formats MMLU into query with choices and answer, categorized by subject complexity."""
    records = []
    letters = ["A", "B", "C", "D"]

    for split in ["test", "validation", "dev"]:
        path = os.path.join(DATA_DIR, "mmlu", f"{split}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            q_raw = clean_text(row["question"])
            choices = [clean_text(str(c)) for c in row["choices"]]
            ans_idx = int(row["answer"])
            subject = row.get("subject", "general")

            # Determine tier
            if subject in MMLU_TIER_MAP["small"]:
                tier = "small"
                reason = f"Introductory/High-School level topic: {subject}"
            elif subject in MMLU_TIER_MAP["large"]:
                tier = "large"
                reason = f"College STEM / Advanced professional domain: {subject}"
            else:
                tier = "medium"
                reason = f"Intermediate academic subject: {subject}"

            # Format full query with multiple choice options
            choices_formatted = "\n".join([f"{letters[i]}) {choices[i]}" for i in range(len(choices))])
            query = f"{q_raw}\n\nChoices:\n{choices_formatted}"
            
            ans_letter = letters[ans_idx] if 0 <= ans_idx < len(letters) else str(ans_idx)
            ans_text = choices[ans_idx] if 0 <= ans_idx < len(choices) else ""
            answer = f"{ans_letter}) {ans_text}".strip()

            records.append({
                "query": query,
                "answer": answer,
                "model_tier": tier,
                "class_label": LABEL_MAP[tier],
                "source": f"mmlu_{subject}",
                "complexity_reason": reason
            })

    print(f"[MMLU] Processed {len(records)} queries.")
    return pd.DataFrame(records)


# =====================================================================
# 4. Process RouteLLM (Evaluated Router Benchmarks)
# =====================================================================
def load_and_clean_routellm() -> pd.DataFrame:
    """
    RouteLLM contains model battle evaluations:
      - routellm/gpt4_dataset: Mixtral (medium) vs GPT-4 (large) score [1..5].
      - routellm/mmlu_battles: Head-to-head model battles.
    """
    records = []

    # 1. GPT-4 vs Mixtral dataset
    gpt4_path = os.path.join(DATA_DIR, "routellm_gpt4_dataset", "validation.parquet")
    if os.path.exists(gpt4_path):
        df = pd.read_parquet(gpt4_path)
        for _, row in df.iterrows():
            q = clean_text(row["prompt"])
            ans = clean_text(row["gpt4_response"])
            score = float(row.get("mixtral_score", 3.0))
            word_count = len(q.split())

            if not q or not ans:
                continue

            # If Mixtral got full score 5 on a concise query, small/medium model is sufficient
            if score >= 5.0 and word_count < 35:
                tier = "small"
                reason = "Simple query, smaller model achieved perfect score"
            elif score >= 4.0:
                tier = "medium"
                reason = "Moderate difficulty, medium model achieved high score"
            else:
                tier = "large"
                reason = f"High complexity (medium model scored only {score:.0f}/5, GPT-4 needed)"

            records.append({
                "query": q,
                "answer": ans,
                "model_tier": tier,
                "class_label": LABEL_MAP[tier],
                "source": "routellm_gpt4",
                "complexity_reason": reason
            })

    # 2. MMLU Battles dataset
    battles_path = os.path.join(DATA_DIR, "routellm_mmlu_battles", "train.parquet")
    if os.path.exists(battles_path):
        df = pd.read_parquet(battles_path)
        for _, row in df.iterrows():
            q = clean_text(row["prompt"])
            ans = clean_text(row.get("response_a", ""))
            if not q or not ans:
                continue

            # Model A is GPT-4, Model B is smaller model
            if row.get("winner_model_a", 0) == 1:
                tier = "large"
                reason = "GPT-4 strictly outperformed smaller model in head-to-head battle"
            elif row.get("winner_tie", 0) == 1:
                tier = "medium"
                reason = "Smaller model tied with GPT-4 quality"
            else:
                tier = "small"
                reason = "Smaller model performed satisfactorily"

            records.append({
                "query": q,
                "answer": ans,
                "model_tier": tier,
                "class_label": LABEL_MAP[tier],
                "source": "routellm_battles",
                "complexity_reason": reason
            })

    print(f"[RouteLLM] Processed {len(records)} queries.")
    return pd.DataFrame(records)


# =====================================================================
# 5. Combine, Balance, and Save to SQLite / Parquet / CSV
# =====================================================================
def create_combined_database(samples_per_class: int = 2500):
    """
    Combines datasets, cleans duplicates, balances class distribution (small, medium, large),
    and saves to SQLite, Parquet, and CSV formats.
    """
    print("\n" + "=" * 60)
    print("STARTING DATA CLEANING & COMBINATION PIPELINE")
    print("=" * 60)

    df_gsm8k = load_and_clean_gsm8k()
    df_mmlu = load_and_clean_mmlu()
    df_routellm = load_and_clean_routellm()

    df_all = pd.concat([df_gsm8k, df_mmlu, df_routellm], ignore_index=True)
    print(f"\nTotal raw queries loaded: {len(df_all)}")

    # Deduplicate queries
    df_all = df_all.drop_duplicates(subset=["query"]).reset_index(drop=True)
    print(f"Total after deduplication: {len(df_all)}")

    # Balance across tiers
    balanced_dfs = []
    print("\nBalancing dataset across model tiers:")
    for tier in ["small", "medium", "large"]:
        subset = df_all[df_all["model_tier"] == tier]
        n_avail = len(subset)
        n_take = min(n_avail, samples_per_class)
        sampled = subset.sample(n=n_take, random_state=42)
        balanced_dfs.append(sampled)
        print(f"  - Tier '{tier}' (Class {LABEL_MAP[tier]}): selected {n_take} / {n_avail} samples")

    df_final = pd.concat(balanced_dfs, ignore_index=True)
    # Shuffle
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"\nFinal balanced dataset size: {len(df_final)} queries.")

    # -------------------------------------------------------------
    # 1. Save to SQLite Database
    # -------------------------------------------------------------
    print(f"\n1. Exporting to SQLite Database: {OUTPUT_DB}...")
    conn = sqlite3.connect(OUTPUT_DB)
    # Create indexed table
    df_final.to_sql("queries", conn, if_exists="replace", index_label="id")
    with conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_tier ON queries (model_tier);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_class_label ON queries (class_label);")
    conn.close()
    print("   -> SQLite table 'queries' created with indexes on model_tier and class_label.")

    # -------------------------------------------------------------
    # 2. Save to Parquet & CSV
    # -------------------------------------------------------------
    print(f"2. Exporting to Parquet: {OUTPUT_PARQUET}...")
    df_final.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"3. Exporting to CSV: {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"4. Exporting to workspace root CSV: {ROOT_CLEANED_CSV}...")
    df_final.to_csv(ROOT_CLEANED_CSV, index=False)

    print("\n" + "=" * 60)
    print("DATABASE CREATION COMPLETE!")
    print("=" * 60)

    # Display preview sample
    print("\nSample Preview (First 3 rows):")
    sample_preview = df_final[["query", "answer", "model_tier", "complexity_reason"]].head(3)
    for idx, row in sample_preview.iterrows():
        print(f"\n--- [Sample {idx + 1}] Model Tier: {row['model_tier'].upper()} ---")
        q_snippet = row['query'][:120] + ('...' if len(row['query']) > 120 else '')
        a_snippet = row['answer'][:100] + ('...' if len(row['answer']) > 100 else '')
        print(f"Query:  {q_snippet}")
        print(f"Answer: {a_snippet}")
        print(f"Reason: {row['complexity_reason']}")


if __name__ == "__main__":
    create_combined_database(samples_per_class=2500)

