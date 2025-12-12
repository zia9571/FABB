import os
import gradio as gr
from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import find_number_in_text, calc_pct_change, normalize_number_str
from agents.reporting_agent import generate_report

# init agents
retriever = RetrievalAgent()

def run_query(user_query: str, from_filter: str, to_filter: str, k:int=12):
    if from_filter:
        from_docs = retriever.retrieve(user_query, k=k, source_filter=[from_filter])
    else:
        from_docs = retriever.retrieve(user_query, k=k)

    if to_filter:
        to_docs = retriever.retrieve(user_query, k=k, source_filter=[to_filter])
    else:
        to_docs = retriever.retrieve(user_query, k=k)

    from_ex = None
    for d in from_docs:
        ex = find_number_in_text(d["content"])
        if ex:
            from_ex = {"value": ex["value"], "context": ex["context"], "source": d["metadata"].get("source")}
            break

    to_ex = None
    for d in to_docs:
        ex = find_number_in_text(d["content"])
        if ex:
            to_ex = {"value": ex["value"], "context": ex["context"], "source": d["metadata"].get("source")}
            break

    if not from_ex or not to_ex:
        return "Could not extract both numbers. Try different source filters or increase k.", "", ""

    pct = calc_pct_change(from_ex["value"], to_ex["value"])
    numeric_trace = {"from_value": str(from_ex["value"]), "to_value": str(to_ex["value"]), "pct_change": str(pct)}
    citations = [
        {"source": from_ex["source"], "context": from_ex["context"]},
        {"source": to_ex["source"], "context": to_ex["context"]}
    ]
    answer = generate_report(numeric_trace, citations, user_query)
    table = f"FROM: {from_ex['source']} = {from_ex['value']}\nTO: {to_ex['source']} = {to_ex['value']}\nYoY% = {pct}"
    citation_text = f"From snippet: {from_ex['context'][:300]}\n\nTo snippet: {to_ex['context'][:300]}"
    return answer, table, citation_text

with gr.Blocks() as demo:
    gr.Markdown("# FAB Multi-Agent Financial Analyzer")
    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(lines=2, placeholder="e.g. YoY change in Net Profit Q3 2023 to Q3 2024", label="User query")
            from_filter = gr.Textbox(lines=1, placeholder="Substring to identify FROM source filename (e.g., Q3-2023 or 2023)", label="From source filter")
            to_filter = gr.Textbox(lines=1, placeholder="Substring to identify TO source filename (e.g., Q3-2024)", label="To source filter")
            run_btn = gr.Button("Run Analysis")
        with gr.Column(scale=2):
            output = gr.Markdown("Results will appear here")
    with gr.Row():
        report = gr.Textbox(label="LLM Report", lines=8)
        numbers = gr.Textbox(label="Numeric Trace & Table", lines=6)
    with gr.Row():
        citations_box = gr.Textbox(label="Citations / Snippets", lines=6)

    def on_click(q, f1, f2):
        ans, tab, cit = run_query(q, f1, f2)
        return ans, tab, cit

    run_btn.click(on_click, inputs=[user_query, from_filter, to_filter], outputs=[report, numbers, citations_box])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
