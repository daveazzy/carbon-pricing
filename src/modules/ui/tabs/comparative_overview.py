"""
Comparative Overview Tab Module

This module contains all functionality for the Comparative Overview tab,
including volume analysis, descriptive statistics, and comparative charts.
"""

import streamlit as st
import pandas as pd
from modules.analysis.descriptive import calcular_analise_enriquecida_por_categoria
from modules.plotting.comparative_charts import (
    plotar_volume_por_categoria,
    plotar_volume_por_pais,
    plotar_evolucao_por_safra
)


def render_comparative_overview_tab(df_filtrado: pd.DataFrame) -> None:
    """
    Render the complete Comparative Overview tab content.
    
    Args:
        df_filtrado: Filtered DataFrame to analyze
    """
    
    st.header("Análise de Volume e Comparação")
    
    if df_filtrado.empty:
        st.warning("Nenhum dado disponível para os filtros selecionados.")
        return
    
    # Generate descriptive analysis
    with st.spinner("Calculando estatísticas descritivas..."):
        analise_descritiva = calcular_analise_enriquecida_por_categoria(df_filtrado)
    
    # Display descriptive statistics table
    _render_descriptive_statistics(analise_descritiva)
    
    # Display comparative charts
    _render_comparative_charts(analise_descritiva, df_filtrado)
    
    # Display summary insights
    _render_summary_insights(df_filtrado, analise_descritiva)


def _render_descriptive_statistics(analise_descritiva: pd.DataFrame) -> None:
    """
    Render the descriptive statistics table.
    
    Args:
        analise_descritiva: DataFrame containing descriptive analysis
    """
    
    st.subheader("📈 Estatísticas Descritivas por Categoria")
    
    if analise_descritiva.empty:
        st.warning("Nenhum dado disponível para análise descritiva.")
        return
    
    # Format the dataframe for better display
    formatted_df = analise_descritiva.copy()
    
    # Add download button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            formatted_df.style.format({
                'total_credits_volume': '{:,.0f}',
                'number_of_transactions': '{:,.0f}',
                'mean_transaction_volume': '{:,.2f}',
                'median_transaction_volume': '{:,.2f}',
                'std_transaction_volume': '{:,.2f}'
            }),
            use_container_width=True
        )
    
    with col2:
        # Convert to CSV for download
        csv = formatted_df.to_csv()
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name="descriptive_statistics.csv",
            mime="text/csv"
        )
    
    # Add explanation
    with st.expander("📖 Como Interpretar Esta Tabela"):
        st.markdown("""
        **Explicação das Colunas:**
        - **Total Credits Volume**: Soma de todas as quantidades de créditos para esta categoria
        - **Number of Transactions**: Contagem de transações individuais
        - **Mean Transaction Volume**: Volume médio por transação
        - **Median Transaction Volume**: Valor mediano dos volumes de transação
        - **Std Transaction Volume**: Desvio padrão (medida de variabilidade)
        
        **Principais Insights:**
        - Categorias com alto volume total mas poucas transações indicam projetos individuais grandes
        - Alto desvio padrão sugere tamanhos diversos de projetos dentro de uma categoria
        - Compare média vs mediana para entender a assimetria da distribuição de volume
        """)


def _render_comparative_charts(analise_descritiva: pd.DataFrame, df_filtrado: pd.DataFrame) -> None:
    """
    Render comparative charts section.
    
    Args:
        analise_descritiva: DataFrame containing descriptive analysis
        df_filtrado: Filtered DataFrame for additional charts
    """
    
    st.subheader("📊 Gráficos Comparativos")
    
    # Create tabs for different chart types
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "🏷️ Por Categoria", 
        "🌍 Por País", 
        "📅 Por Ano Vintage"
    ])
    
    with chart_tab1:
        st.markdown("#### Distribuição de Volume por Categoria de Projeto")
        try:
            fig_category = plotar_volume_por_categoria(analise_descritiva)
            st.pyplot(fig_category)
            
            # Add category insights
            if not analise_descritiva.empty:
                top_category = analise_descritiva.index[0]
                top_volume = analise_descritiva.iloc[0]['total_credits_volume']
                st.info(f"🏆 **Categoria de Maior Volume**: {top_category} com {top_volume:,.0f} tCO₂")
        
        except Exception as e:
            st.error(f"Erro ao gerar gráfico por categoria: {str(e)}")
    
    with chart_tab2:
        st.markdown("#### Distribuição de Volume por País")
        try:
            fig_country = plotar_volume_por_pais(df_filtrado)
            st.pyplot(fig_country)
            
            # Add country insights
            country_volumes = df_filtrado.groupby('project_country')['credits_quantity'].sum().sort_values(ascending=False)
            if not country_volumes.empty:
                top_country = country_volumes.index[0]
                top_country_volume = country_volumes.iloc[0]
                st.info(f"🌍 **País de Maior Volume**: {top_country} com {top_country_volume:,.0f} tCO₂")
        
        except Exception as e:
            st.error(f"Erro ao gerar gráfico por país: {str(e)}")
    
    with chart_tab3:
        st.markdown("#### Evolução por Ano Vintage")
        try:
            fig_vintage = plotar_evolucao_por_safra(df_filtrado)
            st.pyplot(fig_vintage)
            
            # Add temporal insights
            yearly_volumes = df_filtrado.groupby('credit_vintage_year')['credits_quantity'].sum().sort_values(ascending=False)
            if not yearly_volumes.empty:
                peak_year = yearly_volumes.index[0]
                peak_volume = yearly_volumes.iloc[0]
                st.info(f"📅 **Ano de Pico**: {peak_year} com {peak_volume:,.0f} tCO₂")
        
        except Exception as e:
            st.error(f"Erro ao gerar gráfico de evolução vintage: {str(e)}")


def _render_summary_insights(df_filtrado: pd.DataFrame, analise_descritiva: pd.DataFrame) -> None:
    """
    Render summary insights about the filtered data.
    
    Args:
        df_filtrado: Filtered DataFrame
        analise_descritiva: Descriptive analysis results
    """
    
    st.subheader("🎯 Principais Insights")
    
    # Calculate key metrics
    total_transactions = len(df_filtrado)
    total_volume = df_filtrado['credits_quantity'].sum()
    avg_transaction_size = df_filtrado['credits_quantity'].mean()
    unique_categories = len(df_filtrado['project_category'].unique())
    unique_countries = len(df_filtrado['project_country'].unique())
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total de Transações",
            f"{total_transactions:,}",
            help="Número de transações individuais de créditos"
        )
    
    with col2:
        st.metric(
            "Volume Total",
            f"{total_volume:,.0f} tCO₂",
            help="Soma de todas as quantidades de créditos"
        )
    
    with col3:
        st.metric(
            "Transação Média",
            f"{avg_transaction_size:,.0f} tCO₂",
            help="Média de créditos por transação"
        )
    
    with col4:
        st.metric(
            "Categorias",
            unique_categories,
            help="Número de diferentes categorias de projeto"
        )
    
    with col5:
        st.metric(
            "Países",
            unique_countries,
            help="Número de diferentes países"
        )
    
    # Generate insights
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("#### 📈 Concentração de Mercado")
        
        if not analise_descritiva.empty:
            # Top 3 categories by volume
            top_3_categories = analise_descritiva.head(3)
            top_3_volume = top_3_categories['total_credits_volume'].sum()
            concentration = (top_3_volume / total_volume) * 100
            
            st.write("**Top 3 Categorias por Volume:**")
            for i, (category, data) in enumerate(top_3_categories.iterrows(), 1):
                volume_share = (data['total_credits_volume'] / total_volume) * 100
                st.write(f"{i}. {category}: {volume_share:.1f}%")
            
            st.info(f"🎯 As 3 principais categorias representam {concentration:.1f}% do volume total")
    
    with insights_col2:
        st.markdown("#### 🌍 Distribuição Geográfica")
        
        # Top countries by transaction count
        country_counts = df_filtrado['project_country'].value_counts().head(3)
        
        st.write("**Top 3 Países por Contagem de Transações:**")
        for i, (country, count) in enumerate(country_counts.items(), 1):
            share = (count / total_transactions) * 100
            st.write(f"{i}. {country}: {count:,} transações ({share:.1f}%)")
        
        # Geographic diversity
        if unique_countries > 0:
            diversity_score = min(unique_countries / 20, 1.0) * 100  # Normalize to max 20 countries
            st.info(f"🌐 Pontuação de Diversidade Geográfica: {diversity_score:.0f}%")


def get_tab_summary(df_filtrado: pd.DataFrame) -> dict:
    """
    Get a summary of key metrics for this tab.
    
    Args:
        df_filtrado: Filtered DataFrame
        
    Returns:
        Dictionary containing summary metrics
    """
    
    if df_filtrado.empty:
        return {}
    
    return {
        'total_transactions': len(df_filtrado),
        'total_volume': df_filtrado['credits_quantity'].sum(),
        'categories_count': len(df_filtrado['project_category'].unique()),
        'countries_count': len(df_filtrado['project_country'].unique()),
        'avg_transaction_size': df_filtrado['credits_quantity'].mean()
    } 