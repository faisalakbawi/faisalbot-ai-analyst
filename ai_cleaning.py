
import pandas as pd
import numpy as np
from transformers import pipeline

pipe = pipeline("text-generation", model="teknium/OpenHermes-2.5-Mistral-7B", device_map="auto")

def ai_fix_cell(cell_value, column_name, context=""):
    prompt = f"""You are a helpful AI for cleaning CSV data.

Column: {column_name}
Value: {cell_value}
Context: {context}

Fix this value if it's incorrect or incomplete. Otherwise, return it as-is."""
    try:
        out = pipe(prompt, max_new_tokens=50, do_sample=False)[0]['generated_text']
        return out.split("Value:")[1].split("\n")[0].strip()
    except:
        return cell_value

def ai_cleaning(df):
    log = "[AI Cleaning Log:]\n"
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        log += f"🔍 Cleaning column: {col}\n"
        df_cleaned[col] = df_cleaned[col].astype(str).apply(lambda x: ai_fix_cell(x, col))
    
    log += f"✅ Final shape: {df_cleaned.shape}\n"
    return df_cleaned, log
