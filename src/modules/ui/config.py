"""
Page Configuration Module for Carbon Credits Analytics Platform

This module handles all Streamlit page configuration and setup.
Clean, modern interface without sidebar.
"""

import streamlit as st
from modules.ui.styles import load_custom_css


def setup_page_config():
    """
    Configure Streamlit page settings for a clean, modern interface.
    
    This function should be called once at the beginning of the app.
    """
    
    st.set_page_config(
        page_title="Plataforma de Análise de Créditos de Carbono",
        layout="wide", 
        page_icon="🌍",
        initial_sidebar_state="collapsed",  # Start with sidebar collapsed
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Plataforma de Análise de Créditos de Carbono"
        }
    )


def setup_main_header():
    """
    Setup the main application header with clean, modern design.
    
    This includes the main title and subtitle in a professional format.
    """
    
    # Load custom CSS first
    load_custom_css()
    
    # Main title with clean styling
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
    
    # Add spacing
    st.markdown("---")


def display_loading_message():
    """
    Display a loading message while data is being processed.
    """
    
    with st.spinner('Carregando dados de créditos de carbono...'):
        pass


def display_error_message(error_msg: str):
    """
    Display a standardized error message.
    
    Args:
        error_msg: Error message to display
    """
    
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
    """
    Display message when no data matches the current selection.
    """
    
    st.warning("**Nenhum dado disponível para a seleção atual.**")
    
    st.info("""
    **Sugestões:**
    - Ajuste os filtros de seleção
    - Verifique se há dados para o período escolhido
    - Considere expandir os critérios de análise
    """)


def display_success_message(message: str):
    """
    Display a standardized success message.
    
    Args:
        message: Success message to display
    """
    
    st.success(message)


def display_info_message(message: str):
    """
    Display a standardized info message.
    
    Args:
        message: Info message to display
    """
    
    st.info(message)


def add_footer():
    """
    Add a footer to the application with additional information.
    """
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <strong>Plataforma de Análise de Créditos de Carbono</strong>
    </div>
    """, unsafe_allow_html=True) 