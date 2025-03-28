
import gradio as gr
import pandas as pd
from ai_cleaning import ai_cleaning

df_cleaned = None

def analyze(file):
    global df_cleaned
    try:
        df = pd.read_csv(file.name)
        df_cleaned, cleaning_log = ai_cleaning(df)
        df_cleaned.to_csv("cleaned_data.csv", index=False)
        return cleaning_log, df_cleaned, "cleaned_data.csv"
    except Exception as e:
        return f"[ERROR] {str(e)}", pd.DataFrame(), None

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 FaisalBot + OpenHermes Cleaner")
    with gr.Row():
        file_input = gr.File(label="📁 Upload File")
        analyze_btn = gr.Button("Clean & Analyze")
    summary = gr.Textbox(label="🧼 Cleaning Log")
    preview = gr.Dataframe(label="🔍 Cleaned Preview")
    download = gr.File(label="📥 Download Cleaned CSV")
    analyze_btn.click(analyze, inputs=file_input, outputs=[summary, preview, download])

print("🔧 Running on http://localhost:7860")
demo.launch(debug=True, share=False)
