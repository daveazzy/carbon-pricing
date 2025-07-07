import pandas as pd
import streamlit as st

@st.cache_data
def carregar_e_preparar_dados():
    try:
        # low_memory=False para silenciar o DtypeWarning
        credits_df = pd.read_csv('data/credits.csv', low_memory=False)
        projects_df = pd.read_csv('data/projects.csv', low_memory=False)
    except FileNotFoundError:
        st.error("Erro: Ficheiros 'credits.csv' ou 'projects.csv' não encontrados na pasta 'data'.")
        return None

    merged_df = pd.merge(credits_df, projects_df, on='project_id', how='left')
    rename_dict = {
        'quantity': 'credits_quantity',
        'vintage': 'credit_vintage_year',
        'project_type': 'project_category',
        'country': 'project_country'
    }
    merged_df.rename(columns=rename_dict, inplace=True)

    current_year = pd.to_datetime('today').year
    historical_df = merged_df[merged_df['credit_vintage_year'] <= current_year].copy()

    historical_df['transaction_date'] = pd.to_datetime(historical_df['transaction_date'], errors='coerce')
    historical_df.dropna(subset=['transaction_date'], inplace=True)
    historical_df['credit_age_at_transaction'] = historical_df['transaction_date'].dt.year - historical_df['credit_vintage_year']

    cols_to_check = ['credits_quantity', 'credit_age_at_transaction', 'project_category', 'project_country']
    historical_df.dropna(subset=cols_to_check, inplace=True)

    return historical_df
