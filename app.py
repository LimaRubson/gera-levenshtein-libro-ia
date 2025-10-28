# app.py
# -*- coding: utf-8 -*-
import os
from io import BytesIO
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import Levenshtein
from math import ceil

# ===================== Configuração de Página =====================
st.set_page_config(
    page_title="CorreigeAI • Similaridade Levenshtein",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== Utilidades =====================
def _clean_env(s: str | None) -> str:
    """Remove colchetes/aspas acidentais do .env (ex.: ["usuario"] -> usuario)."""
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s

def build_mysql_url(conn: str, host: str, port: str, db: str, user: str, pwd: str) -> str:
    conn = _clean_env(conn).lower()
    if conn != "mysql":
        raise ValueError(f"DB_CONNECTION={conn} não suportado. Use 'mysql'.")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"

def build_mysql_url_from_env() -> str:
    conn = _clean_env(os.getenv("DB_CONNECTION", "mysql"))
    host = _clean_env(os.getenv("DB_HOST", "localhost"))
    port = _clean_env(os.getenv("DB_PORT", "3306"))
    db   = _clean_env(os.getenv("DB_DATABASE", "corrigeai"))
    user = _clean_env(os.getenv("DB_USERNAME", "root"))
    pwd  = _clean_env(os.getenv("DB_PASSWORD", ""))
    return build_mysql_url(conn, host, port, db, user, pwd)

def parse_ids_arg(arg_val: str | None) -> List[int]:
    """Converte '12,34,56' -> [12, 34, 56]."""
    if not arg_val:
        return []
    tokens = [t.strip() for t in arg_val.split(",") if t.strip()]
    ids: List[int] = []
    for t in tokens:
        if not t.isdigit() and not (t.startswith("-") and t[1:].isdigit()):
            raise ValueError(f"Valor inválido para prompt_id: '{t}'")
        ids.append(int(t))
    return ids

def calcular_metricas(texto_modelo: str | None, texto_digitado: str | None) -> Tuple[int, str, str, str, float]:
    """Retorna (dist, motivo, tipos, localizacao, similaridade%)."""
    t1 = "" if texto_modelo is None else str(texto_modelo)
    t2 = "" if texto_digitado is None else str(texto_digitado)

    dist = Levenshtein.distance(t1, t2)
    tmax = max(len(t1), len(t2))
    similaridade = (1 - dist / tmax) * 100 if tmax > 0 else 100.0

    editops = Levenshtein.editops(t1, t2)
    insercoes = [f"Pos {op[2]}: '{t2[op[2]]}'" for op in editops if op[0] == "insert" and op[2] < len(t2)]
    delecoes = [f"Pos {op[1]}: '{t1[op[1]]}'" for op in editops if op[0] == "delete" and op[1] < len(t1)]
    substituicoes = [
        f"Pos {op[1]}: '{t1[op[1]]}' -> '{t2[op[2]]}'"
        for op in editops if op[0] == "replace" and op[1] < len(t1) and op[2] < len(t2)
    ]

    if dist == 0:
        motivo = "Textos idênticos"
    elif dist <= 5:
        motivo = "Pequenas diferenças (erros de digitação ou OCR)"
    elif dist <= 15:
        motivo = "Diferenças moderadas (edições ou ajustes no texto)"
    else:
        motivo = "Grandes diferenças (texto muito alterado ou reconhecimento incorreto)"

    tipos = f"Inserções: {len(insercoes)}, Deleções: {len(delecoes)}, Substituições: {len(substituicoes)}"
    localizacao = (
        f"Inserções: {', '.join(insercoes) if insercoes else '—'} | "
        f"Deleções: {', '.join(delecoes) if delecoes else '—'} | "
        f"Substituições: {', '.join(substituicoes) if substituicoes else '—'}"
    )
    return dist, motivo, tipos, localizacao, round(similaridade, 2)

@st.cache_resource(show_spinner=False)
def get_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)

def montar_query(prompt_ids: List[int]) -> tuple[str, dict]:
    SELECT_BASE = """
    SELECT
        tai.id                AS temp_id,
        tai.redacao_id        AS redacao_id,
        tai.prompt_id         AS prompt_id,
        tai.essay_text        AS essay_text,
        tai.theme             AS theme,
        tai.comp_1_corr, tai.comp_2_corr, tai.comp_3_corr, tai.comp_4_corr, tai.comp_5_corr,
        tai.nota_corr,
        tai.fdbk_comp_1_corr, tai.fdbk_comp_2_corr, tai.fdbk_comp_3_corr, tai.fdbk_comp_4_corr, tai.fdbk_comp_5_corr,
        tai.fdbk_geral_corr,
        tai.melhoria_prompt_id,
        tai.co_correcoes_ia_id,
        tai.levenshtein       AS levenshtein_antigo,
        td.texto_digitado_paragrafos_separados AS texto_digitado
    FROM temp_analise_correcao_ia tai
    LEFT JOIN textos_digitados td
           ON td.redacao_id = tai.redacao_id
    """
    params: dict = {}
    if prompt_ids:
        placeholders = []
        for i, val in enumerate(prompt_ids):
            key = f"p{i}"
            placeholders.append(f":{key}")
            params[key] = val
        query = SELECT_BASE + f" WHERE tai.prompt_id IN ({', '.join(placeholders)})"
    else:
        query = SELECT_BASE
    return query, params

def carregar_dados(engine: Engine, prompt_ids: List[int]) -> pd.DataFrame:
    query, params = montar_query(prompt_ids)
    return pd.read_sql(text(query), con=engine, params=params)

def aplicar_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas linha a linha com barra de progresso percentual (0–100%)."""
    if df.empty:
        return df

    progress = st.progress(0, text="Calculando métricas de similaridade... 0%")
    resultados = []
    n = len(df)

    # Atualiza a cada bloco de linhas para reduzir overhead de UI
    step = max(1, n // 100)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        resultados.append(calcular_metricas(row.get("essay_text"), row.get("texto_digitado")))

        if (idx % step == 0) or (idx == n):
            pct = int((idx / n) * 100)
            progress.progress(pct, text=f"Calculando métricas de similaridade... {pct}%")

    progress.empty()

    cols = ["levenshtein_distancia", "motivo_levenshtein", "tipos_de_alteracoes", "localizacao_alteracoes", "similaridade_textos"]
    df[cols] = pd.DataFrame(resultados, index=df.index)
    return df

def atualizar_banco(engine: Engine, df: pd.DataFrame, chunk_size: int = 500) -> int:
    """Atualiza 'temp_analise_correcao_ia.levenshtein' com barra de progresso percentual."""
    updates = df.loc[df["temp_id"].notna(), ["temp_id", "similaridade_textos"]].copy()
    if updates.empty:
        return 0

    updates["similaridade_textos"] = updates["similaridade_textos"].astype(float)
    params_all = [{"sim": float(r["similaridade_textos"]), "tid": int(r["temp_id"])} for _, r in updates.iterrows()]

    total = len(params_all)
    n_chunks = max(1, ceil(total / chunk_size))
    progress = st.progress(0, text="Atualizando banco de dados... 0%")

    with engine.begin() as conn:
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            batch = params_all[start:end]
            conn.execute(
                text("""
                    UPDATE temp_analise_correcao_ia
                       SET levenshtein = :sim
                     WHERE id = :tid
                """),
                batch
            )
            pct = int((end / total) * 100)
            progress.progress(pct, text=f"Atualizando banco de dados... {pct}%")

    progress.empty()
    return total

def exportar_excel(df: pd.DataFrame) -> tuple[str, BytesIO]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"resultado_levenshtein_corrigeai_{stamp}.xlsx"
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return nome, buffer

# ===================== Layout =====================
st.title("🧮 Similaridade Levenshtein entre Redações")
#st.caption("Comparação entre `essay_text` e `texto_digitado` com filtro por `prompt_id`, atualização no banco e exportação para Excel. (Sem preview de colunas)")

with st.sidebar:
    load_dotenv()  # carrega .env se existir

    try:
        default_url = build_mysql_url_from_env()
    except Exception as e:
        st.error(f"Erro no .env: {e}")
        default_url = ""
    url = default_url

    st.divider()
    st.subheader("Filtro")
    prompt_ids_str = st.text_input("prompt_id (separe por vírgula)", value=_clean_env(os.getenv("PROMPT_ID", "")))
    try:
        prompt_ids = parse_ids_arg(prompt_ids_str)
        if prompt_ids:
            st.info(f"Filtrando por prompt_id IN {prompt_ids}")
        else:
            st.warning("Nenhum prompt_id informado — serão processadas TODAS as linhas.", icon="⚠️")
    except ValueError as e:
        st.error(str(e))
        prompt_ids = []

    st.divider()
    do_update = st.toggle("Atualizar coluna `levenshtein` no banco", value=True,
                          help="Quando ativado, grava `similaridade_textos` em `temp_analise_correcao_ia.levenshtein`.")

    run_btn = st.button("▶️ Executar", type="primary", use_container_width=True)

# ===================== Execução =====================
if run_btn:
    if not url:
        st.error("Informe uma URL de conexão válida.")
        st.stop()

    with st.spinner("Conectando ao banco..."):
        try:
            engine = get_engine(url)
            # Ping simples
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            st.error(f"Falha na conexão: {e}")
            st.stop()

    st.success("Conexão estabelecida com sucesso.")

    with st.spinner("Carregando dados..."):
        try:
            df = carregar_dados(engine, prompt_ids)
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            st.stop()

    if df.empty:
        st.warning("Nenhuma linha encontrada para o filtro informado." if prompt_ids else "Nenhuma linha encontrada.")
        # Gera planilha vazia para rastreabilidade
        cols = [
            "temp_id","redacao_id","prompt_id","essay_text","theme","comp_1_corr","comp_2_corr","comp_3_corr",
            "comp_4_corr","comp_5_corr","nota_corr","fdbk_comp_1_corr","fdbk_comp_2_corr","fdbk_comp_3_corr",
            "fdbk_comp_4_corr","fdbk_comp_5_corr","fdbk_geral_corr","melhoria_prompt_id","co_correcoes_ia_id",
            "levenshtein_antigo","texto_digitado","levenshtein_distancia","motivo_levenshtein",
            "tipos_de_alteracoes","localizacao_alteracoes","similaridade_textos"
        ]
        empty_df = pd.DataFrame(columns=cols)
        nome, buff = exportar_excel(empty_df)
        st.download_button("⬇️ Baixar planilha (vazia)", data=buff, file_name=nome, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.stop()

    # Métricas com barra de progresso %
    df = aplicar_metricas(df)

    # Resumo (sem preview de colunas)
    colA, colB, colC, colD = st.columns(4)
    try:
        media_sim = float(df["similaridade_textos"].mean())
        mediana_sim = float(df["similaridade_textos"].median())
        max_sim = float(df["similaridade_textos"].max())
        min_sim = float(df["similaridade_textos"].min())
    except Exception:
        media_sim = mediana_sim = max_sim = min_sim = 0.0

    colA.metric("Média de Similaridade (%)", f"{media_sim:.2f}")
    colB.metric("Mediana de Similaridade (%)", f"{mediana_sim:.2f}")
    colC.metric("Máxima Similaridade (%)", f"{max_sim:.2f}")
    colD.metric("Mínima Similaridade (%)", f"{min_sim:.2f}")

    st.info(f"Linhas processadas: {len(df)}")

    # Atualização opcional no banco com barra de progresso %
    if do_update:
        try:
            total_upd = atualizar_banco(engine, df, chunk_size=500)
            if total_upd > 0:
                st.success(f"Atualização concluída: {total_upd} linha(s) atualizada(s) em `temp_analise_correcao_ia.levenshtein`.")
            else:
                st.info("Nenhuma linha para atualizar em `temp_analise_correcao_ia`.")
        except Exception as e:
            st.error(f"Erro ao atualizar o banco: {e}")

    # Download do Excel (sem exibir a tabela)
    nome, buff = exportar_excel(df)
    st.download_button(
        "⬇️ Baixar planilha Excel",
        data=buff,
        file_name=nome,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ===================== Rodapé =====================
with st.expander("ℹ️ Ajuda & Notas"):
    st.markdown(
        """
- **Sem preview de colunas**: a aplicação não exibe a tabela de resultados; utilize o download para análise detalhada.
- **Filtro `prompt_id`**: informe IDs separados por vírgula (ex.: `12,34,56`). Deixe vazio para processar todas as linhas.
- **Atualização no banco**: quando ativada, grava `similaridade_textos` em `temp_analise_correcao_ia.levenshtein`.
- **Progresso**: barras percentuais no cálculo e na atualização do banco.
"""
    )
