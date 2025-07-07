"""
Page Configuration Module for Carbon Credits Analytics Platform

This module handles all Streamlit page configuration and setup.
Clean, modern interface without sidebar.
"""

import streamlit as st
from modules.ui.styles import load_custom_css


def setup_page_config():
    
    st.set_page_config(
        page_title="Plataforma de Análise de Créditos de Carbono",
        layout="wide", 
        page_icon="🌍",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Plataforma de Análise de Créditos de Carbono"
        }
    )


def setup_main_header():
    
    load_custom_css()
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; font-size: 2.5rem; font-weight: 600; margin-bottom: 0.5rem;">
            Plataforma de Análise de Créditos de Carbono
        </h1>
        <p style="color: #6c757d; font-size: 1.2rem; margin-bottom: 0;">
            Inteligência de Mercado para Investimento e Negociação
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")


def display_loading_message():
    
    with st.spinner('Carregando dados de créditos de carbono...'):
        pass


def display_error_message(error_msg: str):
    
    st.error(f"**Erro:** {error_msg}")
    
    with st.expander("Ajuda para Resolução"):
        st.markdown("""
        **Soluções possíveis:**
        - Verifique se os arquivos de dados existem no diretório `data/`
        - Verifique as permissões dos arquivos
        - Certifique-se de que as dependências estão instaladas
        - Consulte o console para detalhes do erro
        """)


def display_no_data_message():
    
    st.warning("**Nenhum dado disponível para a seleção atual.**")
    
    st.info("""
    **Sugestões:**
    - Ajuste os filtros de seleção
    - Verifique se há dados para o período escolhido
    - Considere expandir os critérios de análise
    """)


def display_success_message(message: str):
    
    st.success(message)


def display_info_message(message: str):
    
    st.info(message)


def add_footer():
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <strong>Plataforma de Análise de Créditos de Carbono</strong>
    </div>
    """, unsafe_allow_html=True) 