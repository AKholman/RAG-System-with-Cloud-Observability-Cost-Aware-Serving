import json
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# =========================
# Load evaluation results
# =========================
with open("evaluation/generation_eval.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# =========================
# Compute summary metrics
# =========================
avg_similarity = df["semantic_similarity"].mean()

# if rubric exists
if "rubric" in df.columns:
    df_rubric = pd.json_normalize(df["rubric"])
    avg_factual = df_rubric["factual_correctness"].mean()
    avg_relevance = df_rubric["relevance"].mean()
    avg_hallucination = df_rubric["hallucination"].mean()
else:
    avg_factual = avg_relevance = avg_hallucination = None

# =========================
# Plot 1: similarity scores
# =========================
plt.figure()
plt.hist(df["semantic_similarity"], bins=10)
plt.title("Semantic Similarity Distribution (RAG vs LLM baseline)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.savefig("evaluation/similarity_hist.png")

# =========================
# Plot 2: RAG vs LLM length
# =========================
df["rag_len"] = df["rag_answer"].apply(len)
df["llm_len"] = df["llm_only_answer"].apply(len)

plt.figure()
plt.plot(df["rag_len"], label="RAG")
plt.plot(df["llm_len"], label="LLM-only")
plt.legend()
plt.title("Answer Length Comparison")
plt.savefig("evaluation/length_comparison.png")

# =========================
# Create PDF report
# =========================
pdf = FPDF()
pdf.add_page()

pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="RAG Evaluation Report", ln=True)

pdf.cell(200, 10, txt=f"Avg Semantic Similarity: {avg_similarity:.3f}", ln=True)

if avg_factual is not None:
    pdf.cell(200, 10, txt=f"Avg Factual Score: {avg_factual:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Avg Relevance Score: {avg_relevance:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Avg Hallucination Score: {avg_hallucination:.2f}", ln=True)

pdf.cell(200, 10, txt="Plots saved in evaluation/", ln=True)

pdf.output("evaluation/rag_report.pdf")

print("✅ Report generated: evaluation/rag_report.pdf")
