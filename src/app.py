"""
Plataforma de Análise de Créditos de Carbono

Sistema principal de análise preditiva para mercado de créditos de carbono.
Baseado em 458,302 transações reais coletadas entre 2002-2025.

Características:
- Análise sazonal e timing de mercado
- Calculadora de risco e volatilidade  
- Análise de tendências e momentum
- Previsão de volumes futuros
- Identificação de oportunidades geográficas
"""

import streamlit as st
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from modules.ui.config import (
    setup_page_config, 
    setup_main_header,
    display_error_message,
    add_footer
)

from modules.data_loader import load_all_data

from modules.ui.tabs.predictive_calculators import render_predictive_calculators_tab

from modules.analysis.descriptive import render_descriptive_analysis_tab
from modules.analysis.inferential import render_inferential_analysis_tab  
from modules.analysis.modeling import render_modeling_tab

from modules.plotting.distribution_charts import render_distribution_charts_tab
from modules.plotting.comparative_charts import render_comparative_charts_tab


def main():
    
    setup_page_config()
    setup_main_header()
    
    credits_df, projects_df = load_all_data()
    
    if credits_df is not None and projects_df is not None:
        handle_navigation(credits_df)
    else:
        display_error_message("Não foi possível carregar os dados necessários")
        return
    
    add_footer()


def handle_navigation(df: pd.DataFrame):
    
    page_options = {
        "🔮 Calculadoras Preditivas": "calculators",
        "📊 Análise Exploratória": "descriptive", 
        "🔬 Análise Inferencial": "inferential",
        "🤖 Modelagem": "modeling",
        "📈 Gráficos Distribuição": "distribution_charts",
        "📊 Gráficos Comparativos": "comparative_charts"
    }
    
    st.sidebar.header("Navegação")
    selected_page = st.sidebar.selectbox(
        "Escolha uma análise:",
        list(page_options.keys()),
        index=0
    )
    
    page_key = page_options[selected_page]
    
    if page_key == "calculators":
        render_calculators_page(df)
    elif page_key == "descriptive":
        render_descriptive_analysis_tab(df)
    elif page_key == "inferential":
        render_inferential_analysis_tab(df)
    elif page_key == "modeling":
        render_modeling_tab(df)
    elif page_key == "distribution_charts":
        render_distribution_charts_tab(df)
    elif page_key == "comparative_charts":
        render_comparative_charts_tab(df)


def render_calculators_page(df: pd.DataFrame):
    
    st.markdown("""
    <div class="calculator-header">
        <h2>🔮 Calculadoras Preditivas</h2>
        <p>Ferramentas avançadas de análise preditiva para otimização de estratégias</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_predictive_calculators_tab(df)


if __name__ == "__main__":
    main() 