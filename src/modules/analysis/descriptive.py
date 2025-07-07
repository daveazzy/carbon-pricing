import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_descriptive_analysis_tab(df):
    
    st.header("📊 Análise Exploratória")
    
    tab1, tab2, tab3 = st.tabs(["Visão Geral", "Análise por Categoria", "Análise Temporal"])
    
    with tab1:
        render_overview_analysis(df)
    
    with tab2:
        render_category_analysis(df)
    
    with tab3:
        render_temporal_analysis(df)


def render_overview_analysis(df):
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Transações", f"{len(df):,}")
    
    with col2:
        st.metric("Volume Total", f"{df['credits_quantity'].sum():,.0f} tCO₂")
    
    with col3:
        st.metric("Categorias de Projeto", len(df['project_category'].unique()))
    
    with col4:
        st.metric("Países", len(df['project_country'].unique()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Categoria")
        category_counts = df['project_category'].value_counts().head(10)
        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 10 Categorias por Número de Transações"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Volume por País")
        country_volume = df.groupby('project_country')['credits_quantity'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=country_volume.values,
            y=country_volume.index,
            orientation='h',
            title="Top 10 Países por Volume"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_category_analysis(df):
    
    selected_category = st.selectbox(
        "Selecione uma categoria para análise detalhada:",
        df['project_category'].unique()
    )
    
    category_data = df[df['project_category'] == selected_category]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Transações", len(category_data))
    
    with col2:
        st.metric("Volume Total", f"{category_data['credits_quantity'].sum():,.0f} tCO₂")
    
    with col3:
        st.metric("Volume Médio", f"{category_data['credits_quantity'].mean():,.0f} tCO₂")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Volume")
        fig = px.histogram(
            category_data,
            x='credits_quantity',
            title=f"Distribuição de Volume - {selected_category}",
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Volume ao Longo do Tempo")
        monthly_volume = category_data.groupby(category_data['transaction_date'].dt.to_period('M'))['credits_quantity'].sum()
        
        fig = px.line(
            x=monthly_volume.index.astype(str),
            y=monthly_volume.values,
            title=f"Volume Mensal - {selected_category}"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_temporal_analysis(df):
    
    st.subheader("Análise Temporal")
    
    df['year'] = df['transaction_date'].dt.year
    df['month'] = df['transaction_date'].dt.month
    
    yearly_stats = df.groupby('year').agg({
        'credits_quantity': ['count', 'sum', 'mean']
    }).round(2)
    
    yearly_stats.columns = ['Transações', 'Volume Total', 'Volume Médio']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            x=yearly_stats.index,
            y=yearly_stats['Transações'],
            title="Número de Transações por Ano"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            x=yearly_stats.index,
            y=yearly_stats['Volume Total'],
            title="Volume Total por Ano"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sazonalidade Mensal")
    monthly_avg = df.groupby('month')['credits_quantity'].mean()
    
    fig = px.bar(
        x=monthly_avg.index,
        y=monthly_avg.values,
        title="Volume Médio por Mês"
    )
    fig.update_xaxis(tickmode='array', tickvals=list(range(1, 13)), 
                     ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                              'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    st.plotly_chart(fig, use_container_width=True)