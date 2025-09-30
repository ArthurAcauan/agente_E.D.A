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
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Agente EDA (CSV)", layout="wide")
init_db()

st.title("Agente E.D.A. — Pergunte sobre qualquer CSV")

uploaded = st.file_uploader("Carregue um arquivo CSV", type=["csv","zip"], help="Você pode carregar o arquivo 'creditcard.csv' fornecido no curso.")

if uploaded is None and os.path.exists("data/creditcard.csv"):
    st.info("Usando data/creditcard.csv local (Kaggle creditcard).")
    uploaded = open("data/creditcard.csv","rb")

if uploaded:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.write("Preview dos dados:")
    st.dataframe(df.head(200))

    if st.button("Gerar resumo básico"):
        summary = basic_summary(df)
        st.json(summary)

    st.subheader("Faça uma pergunta (ex.: 'Qual a taxa de fraudes?')")
    user_q = st.text_input("Pergunta do usuário:")
    n_context = st.slider("Quantas interações passadas incluir na memória?", 0, 10, 3)

    if st.button("Enviar pergunta"):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        corr = correlation_matrix(df)
        shape = df.shape
        top_cols = numeric_cols[:5]

        quick_stats = {
            "shape": shape,
            "numeric_columns_sample": top_cols,
            "corr_top": {k: corr[k] for k in list(corr.keys())[:5]} if isinstance(corr, dict) else corr
        }

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
        try:
            llm_answer = call_gemini(prompt, max_tokens=600, temperature=0.0)
        except Exception as e:
            llm_answer = f"Erro ao chamar LLM: {e}"

        st.markdown("### Resposta do Agente (LLM)")
        st.write(llm_answer)

        save_interaction(user_q, llm_answer, metadata={"dataset_shape": shape})

        if "hist" in user_q.lower() or "distribui" in user_q.lower() or "distribuição" in user_q.lower():
            col = "Amount" if "Amount" in df.columns else numeric_cols[0]
            buf = save_histogram(df, col)
            st.image(buf.getvalue(), use_column_width=True)
            st.success(f"Histograma de {col} gerado automaticamente.")

        if "correlação" in user_q.lower() or "heatmap" in user_q.lower():
            if 'Class' in numeric_cols:
                class_correlations = correlation_matrix['Class'].sort_values(ascending=False)
                top_features = class_correlations.head(10).index.tolist()
                bottom_features = class_correlations.tail(10).index.tolist()
                selected_features = list(set(top_features + bottom_features))

                plt.figure(figsize=(12, 10))
                sns.heatmap(df[selected_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title('Heatmap: Variáveis mais correlacionadas com Class')
                st.pyplot()
            else:
                st.error("A coluna 'Class' não foi encontrada no dataset.")

        if "fraudes" in user_q.lower() and "comparar" in user_q.lower():
            fraud_counts = df['Class'].value_counts()
            fig, ax = plt.subplots()
            fraud_counts.plot(kind='bar', ax=ax, color=['blue', 'red'])
            ax.set_title("Comparação: Fraudes vs Não Fraudes")
            ax.set_xlabel("Categoria")
            ax.set_ylabel("Contagem")
            st.pyplot(fig)

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
        col = "Amount" if "Amount" in df.columns else df.select_dtypes(include=['number']).columns[0]
        buf = save_histogram(df, col)
        rep.add_image_from_buf(buf)
        path = rep.output()
        with open(path,"rb") as f:
            st.download_button("Baixar PDF", f, file_name=path)

    st.sidebar.subheader("Memória (últimas interações)")
    st.sidebar.write(last_k_interactions(5))
