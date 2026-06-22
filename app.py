import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / '.env')
sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
from src.rag_pipeline import RAGPipeline

print("Starting CrediTrust Complaint Analysis Chatbot...")
pipeline = RAGPipeline(vector_store_path='./vector_store/chroma_db')
print("Ready!")


def answer_question(question: str) -> tuple:
    if not question.strip():
        return "Please enter a question.", ""

    result = pipeline.ask(question)

    sources_html = ""
    for i, source in enumerate(result['sources']):
        meta = source['metadata']
        sources_html += f"""
<div class="source-card">
  <div class="source-header">
    <span class="source-num">Source {i+1}</span>
    <span class="source-product">{meta.get('product_category', 'N/A')}</span>
    <span class="source-issue">{meta.get('issue', 'N/A')}</span>
  </div>
  <div class="source-meta">
    <span>🏢 {meta.get('company', 'N/A')}</span>
    <span>📅 {meta.get('date_received', 'N/A')}</span>
    <span>📍 {meta.get('state', 'N/A')}</span>
  </div>
  <div class="source-excerpt">"{source['text'][:220]}..."</div>
</div>"""

    return result['answer'], sources_html


def clear_all():
    return "", "", ""


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 0 24px 60px !important;
}

/* Header */
.header-wrap {
    padding: 52px 0 40px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 36px;
}

.header-eyebrow {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a96e;
    margin-bottom: 14px;
}

.header-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 38px;
    font-weight: 600;
    color: #f0ece4;
    line-height: 1.15;
    margin-bottom: 14px;
    letter-spacing: -0.5px;
}

.header-sub {
    font-size: 15px;
    color: rgba(240,236,228,0.45);
    font-weight: 300;
    line-height: 1.6;
    max-width: 560px;
}

/* Product pills */
.pills-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 24px;
}

.pill {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 5px 14px;
    border-radius: 100px;
    border: 1px solid rgba(201,169,110,0.3);
    color: #c9a96e;
    background: rgba(201,169,110,0.06);
}

/* Fix Gradio Default Layout Blocks Padding */
.gradio-container .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Input section and generic textarea styling overrides */
.input-section { margin-bottom: 20px; }

.gradio-container .input-section label,
.gradio-container .answer-section label {
    display: block !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: rgba(240,236,228,0.5) !important;
    margin-bottom: 10px !important;
}

/* Deep UI injection targeting specific internal Gradio parts */
.gradio-container textarea,
.gradio-container .input-section textarea,
.gradio-container .input-section .scroll-hide {
    background-color: #141420 !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(201, 169, 110, 0.3) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 300 !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    resize: none !important;
    display: block !important;
    width: 100% !important;
    min-height: 100px !important;
}

.gradio-container textarea:focus,
.gradio-container .input-section textarea:focus {
    border-color: rgba(201,169,110,0.7) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(201,169,110,0.15) !important;
}

.gradio-container textarea::placeholder {
    color: rgba(240, 236, 228, 0.25) !important;
}

/* Buttons */
.btn-row {
    display: flex;
    gap: 10px;
    margin-bottom: 28px;
}

.btn-ask {
    background: linear-gradient(135deg, #c9a96e 0%, #a8853d 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0a0a0f !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
}

.btn-ask:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

.btn-clear {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: rgba(240,236,228,0.45) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s !important;
}

.btn-clear:hover {
    border-color: rgba(255,255,255,0.2) !important;
    color: rgba(240,236,228,0.7) !important;
}

/* Answer output styling overrides */
.answer-section { margin-bottom: 20px; }

.gradio-container .answer-section textarea,
.gradio-container .answer-section .scroll-hide {
    background: rgba(201,169,110,0.05) !important;
    border: 1.5px solid rgba(201,169,110,0.25) !important;
    border-radius: 12px !important;
    color: #f0ece4 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 300 !important;
    line-height: 1.75 !important;
    padding: 20px !important;
    min-height: 200px !important;
}

/* Divider */
.section-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 28px 0;
}

/* Sources label */
.sources-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(240,236,228,0.4);
    margin-bottom: 16px;
    padding: 0;
}

/* Source cards */
.source-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}

.source-card:hover { border-color: rgba(201,169,110,0.2); }

.source-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.source-num {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c9a96e;
    background: rgba(201,169,110,0.1);
    border: 1px solid rgba(201,169,110,0.2);
    padding: 3px 10px;
    border-radius: 100px;
}

.source-product {
    font-size: 12px;
    font-weight: 500;
    color: rgba(240,236,228,0.8);
    background: rgba(255,255,255,0.05);
    padding: 3px 10px;
    border-radius: 100px;
}

.source-issue {
    font-size: 12px;
    color: rgba(240,236,228,0.4);
}

.source-meta {
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: rgba(240,236,228,0.3);
    margin-bottom: 12px;
    flex-wrap: wrap;
}

.source-excerpt {
    font-size: 13px;
    font-style: italic;
    color: rgba(240,236,228,0.45);
    line-height: 1.7;
    border-left: 2px solid rgba(201,169,110,0.3);
    padding-left: 14px;
}

/* Footer */
.footer-wrap {
    margin-top: 52px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-brand {
    font-size: 12px;
    font-weight: 500;
    color: #c9a96e;
    letter-spacing: 0.08em;
}

.footer-stack {
    font-size: 11px;
    color: rgba(240,236,228,0.2);
    letter-spacing: 0.06em;
}

/* Hide gradio footer */
footer { display: none !important; }
.prose { color: #f0ece4 !important; }
"""

with gr.Blocks(title="CrediTrust — Complaint Intelligence", css=css) as app:
    gr.HTML("""
    <div class="header-wrap">
        <div class="header-eyebrow">Internal Intelligence Tool</div>
        <div class="header-title">Complaint Analysis<br>Assistant</div>
        <div class="header-sub">
            Ask plain-English questions about customer complaints.
            Get synthesized, evidence-backed answers in seconds.
        </div>
        <div class="pills-row">
            <span class="pill">Credit Cards</span>
            <span class="pill">Personal Loans</span>
            <span class="pill">Savings Accounts</span>
            <span class="pill">Money Transfers</span>
        </div>
    </div>
    """)

    with gr.Column(elem_classes="input-section"):
        question_input = gr.Textbox(
            label="Ask a question",
            placeholder="e.g. Why are people unhappy with credit cards?",
            lines=3,
        )

    with gr.Row(elem_classes="btn-row"):
        ask_btn = gr.Button("Analyze →", elem_classes="btn-ask")
        clear_btn = gr.Button("Clear", elem_classes="btn-clear")

    with gr.Column(elem_classes="answer-section"):
        answer_output = gr.Textbox(
            label="AI Generated Answer",
            lines=9,
            interactive=False,
            placeholder="Your answer will appear here..."
        )

    gr.HTML('<div class="section-divider"></div>')
    gr.HTML('<div class="sources-label">Retrieved Sources</div>')

    sources_output = gr.HTML(
        value='<div style="color:rgba(240,236,228,0.2);font-size:13px;padding:8px 0;">Source complaints will appear here after you ask a question.</div>'
    )

    gr.HTML("""
    <div class="footer-wrap">
        <span class="footer-brand">CrediTrust Financial</span>
        <span class="footer-stack">RAG · ChromaDB · LLaMA 3 · all-MiniLM-L6-v2</span>
    </div>
    """)

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, answer_output, sources_output]
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    app.launch(share=False)