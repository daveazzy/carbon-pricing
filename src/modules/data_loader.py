import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_credits_data():
    
    try:
        credits_path = Path("data/credits.csv")
        credits = pd.read_csv(credits_path)
        credits['transaction_date'] = pd.to_datetime(credits['transaction_date'])
        return credits
    except Exception as e:
        st.error(f"Erro ao carregar dados de créditos: {e}")
        return None


@st.cache_data  
def load_projects_data():
    
    try:
        projects_path = Path("data/projects.csv")
        projects = pd.read_csv(projects_path)
        return projects
    except Exception as e:
        st.error(f"Erro ao carregar dados de projetos: {e}")
        return None


def load_all_data():
    
    credits = load_credits_data()
    projects = load_projects_data()
    
    if credits is not None and projects is not None:
        return credits, projects
    else:
        return None, None