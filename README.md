# agente_E.D.A

# Agente E.D.A. — Projeto (Entrega: Agentes Autônomos – Atividade Extra)

Resumo: Agente em Streamlit que carrega CSVs, realiza EDA, gera gráficos, responde perguntas via LLM (Gemini) e guarda memória.

Como rodar:
1. cp .env.example .env  -> preencha GEMINI_API_KEY e GEMINI_API_URL
2. pip install -r requirements.txt
3. streamlit run streamlit_app.py

Arquivos:
- streamlit_app.py -> interface
- agent/eda_engine.py -> funções EDA
- agent/llm_client.py -> wrapper Gemini
- agent/memory.py -> SQLite memory
- agent/pdf_report.py -> gerar PDF

Exemplos de perguntas:
- "Qual a taxa de fraude?" -> resposta com números e plot.
- "Há outliers em Amount?" -> IQR e sugestão.
- ...

Observações:
- Não incluir chaves no repositório.
- Link para teste: <inserir link de deploy>