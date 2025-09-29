# streamlit_app.py
import streamlit as st
from agent.eda_engine import load_csv, basic_summary, column_distribution, outliers_iqr, correlation_matrix, kmeans_clusters, top_frequent_values
from agent.plot_utils import save_histogram, save_scatter, buf_to_base64_png
from agent.llm_client import call_gemini
from agent.memory import init_db, save_interaction, last_k_interactions
from agent.pdf_report import SimpleReport
import pandas as pd
import io
import os

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Agente EDA (CSV)", layout="wide")
init_db()

st.title("Agente E.D.A. — Pergunte sobre qualquer CSV")

uploaded = st.file_uploader("Carregue um arquivo CSV", type=["csv","zip"], help="Você pode carregar o arquivo 'creditcard.csv' fornecido no curso.")

# quick load sample if exists
if uploaded is None and os.path.exists("data/creditcard.csv"):
    st.info("Usando data/creditcard.csv local (Kaggle creditcard).")
    uploaded = open("data/creditcard.csv","rb")

if uploaded:
    # handle zip or csv
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.write("Preview dos dados:")
    st.dataframe(df.head(200))

    # show basic summary
    if st.button("Gerar resumo básico"):
        summary = basic_summary(df)
        st.json(summary)

    # user question
    st.subheader("Faça uma pergunta (ex.: 'Qual a taxa de fraudes?')")
    user_q = st.text_input("Pergunta do usuário:")
    n_context = st.slider("Quantas interações passadas incluir na memória?", 0, 10, 3)

    if st.button("Enviar pergunta"):
        # 1) compute a few helpful stats automatically
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        corr = correlation_matrix(df)
        shape = df.shape
        top_cols = numeric_cols[:5]

        quick_stats = {
            "shape": shape,
            "numeric_columns_sample": top_cols,
            "corr_top": {k: corr[k] for k in list(corr.keys())[:5]} if isinstance(corr, dict) else corr
        }

        # 2) create context prompt for LLM
        memory = last_k_interactions(n_context)
        memory_text = "\n".join([f"{m['timestamp']} | Q: {m['user_query']} | A: {m['agent_response']}" for m in memory])

        prompt = f"""
You are an EDA assistant. The user asked: {user_q}
Dataset shape: {shape}
Numeric sample columns: {top_cols}
Provide:
1) A concise answer to user's question.
2) If relevant, list of steps and a short Python snippet to reproduce results.
3) If the answer requires numeric summary, include JSON with the necessary numbers.
Memory:
{memory_text}
"""
        # call Gemini
        try:
            llm_answer = call_gemini(prompt, max_tokens=600, temperature=0.0)
        except Exception as e:
            llm_answer = f"Erro ao chamar LLM: {e}"

        st.markdown("### Resposta do Agente (LLM)")
        st.write(llm_answer)

        save_interaction(user_q, llm_answer, metadata={"dataset_shape": shape})

        # if user asked for histogram-like keywords, try to auto-generate a plot
        if "hist" in user_q.lower() or "distribui" in user_q.lower() or "distribuição" in user_q.lower():
            # pick a numeric column (simple heuristic: Amount or first numeric)
            col = "Amount" if "Amount" in df.columns else numeric_cols[0]
            buf = save_histogram(df, col)
            st.image(buf.getvalue(), use_column_width=True)
            st.success(f"Histograma de {col} gerado automaticamente.")

    st.sidebar.header("Funções rápidas")
    if st.sidebar.button("Detectar outliers (IQR) em primeira coluna numérica"):
        col = df.select_dtypes(include=['number']).columns[0]
        out = outliers_iqr(df, col)
        st.sidebar.json(out)

    if st.sidebar.button("Gerar PDF com resumo e 1 gráfico"):
        rep = SimpleReport()
        rep.add_title("Relatório automático - Agente EDA")
        rep.add_paragraph(f"Dataset: {uploaded if hasattr(uploaded,'name') else 'uploaded'}")
        rep.add_paragraph("Resumo básico:")
        rep.add_paragraph(str(basic_summary(df)))
        # add histogram
        col = "Amount" if "Amount" in df.columns else df.select_dtypes(include=['number']).columns[0]
        buf = save_histogram(df, col)
        rep.add_image_from_buf(buf)
        path = rep.output()
        with open(path,"rb") as f:
            st.download_button("Baixar PDF", f, file_name=path)

    st.sidebar.subheader("Memória (últimas interações)")
    st.sidebar.write(last_k_interactions(5))
