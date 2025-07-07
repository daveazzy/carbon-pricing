"""
Distribution Analysis Tab Module

This module contains all functionality for the Distribution Analysis tab,
including histograms, boxplots, and statistical distribution analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.plotting.distribution_charts import (
    plotar_histograma_distribuicao,
    plotar_boxplot_por_categoria
)


def render_distribution_analysis_tab(df_filtrado: pd.DataFrame, selected_categories: list) -> None:
    """
    Render the complete Distribution Analysis tab content.
    
    Args:
        df_filtrado: Filtered DataFrame to analyze
        selected_categories: List of selected categories for analysis
    """
    
    st.header("Análise de Distribuição do Volume de Transações")
    
    if df_filtrado.empty:
        st.warning("Nenhum dado disponível para os filtros selecionados.")
        return
    
    # Render distribution overview
    _render_distribution_overview(df_filtrado)
    
    # Render distribution charts
    _render_distribution_charts(df_filtrado, selected_categories)
    
    # Render statistical analysis
    _render_statistical_analysis(df_filtrado)
    
    # Render outlier analysis
    _render_outlier_analysis(df_filtrado)


def _render_distribution_overview(df_filtrado: pd.DataFrame) -> None:
    """
    Render overview statistics about the volume distribution.
    
    Args:
        df_filtrado: Filtered DataFrame to analyze
    """
    
    st.subheader("📊 Visão Geral da Distribuição")
    
    # Calculate distribution statistics
    volume_stats = df_filtrado['credits_quantity'].describe()
    
    # Display key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Volume Médio",
            f"{volume_stats['mean']:,.0f} tCO₂",
            help="Volume médio de transação"
        )
    
    with col2:
        st.metric(
            "Volume Mediano", 
            f"{volume_stats['50%']:,.0f} tCO₂",
            help="Valor do meio de todas as transações"
        )
    
    with col3:
        st.metric(
            "Desvio Padrão",
            f"{volume_stats['std']:,.0f} tCO₂",
            help="Medida de variabilidade do volume"
        )
    
    with col4:
        # Calculate skewness
        skewness = df_filtrado['credits_quantity'].skew()
        st.metric(
            "Assimetria",
            f"{skewness:.2f}",
            help="Medida de assimetria da distribuição"
        )
    
    # Additional distribution metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Mínimo",
            f"{volume_stats['min']:,.0f} tCO₂"
        )
    
    with col6:
        st.metric(
            "25º Percentil",
            f"{volume_stats['25%']:,.0f} tCO₂"
        )
    
    with col7:
        st.metric(
            "75º Percentil", 
            f"{volume_stats['75%']:,.0f} tCO₂"
        )
    
    with col8:
        st.metric(
            "Máximo",
            f"{volume_stats['max']:,.0f} tCO₂"
        )
    
    # Distribution shape analysis
    _render_distribution_shape_analysis(df_filtrado, skewness)


def _render_distribution_shape_analysis(df_filtrado: pd.DataFrame, skewness: float) -> None:
    """
    Analyze and display information about the distribution shape.
    
    Args:
        df_filtrado: Filtered DataFrame
        skewness: Calculated skewness value
    """
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📐 Formato da Distribuição")
        
        if abs(skewness) < 0.5:
            shape_desc = "aproximadamente simétrica"
            shape_icon = "⚖️"
        elif skewness > 0.5:
            shape_desc = "assimétrica à direita (assimetria positiva)"
            shape_icon = "↗️"
        else:
            shape_desc = "assimétrica à esquerda (assimetria negativa)"
            shape_icon = "↖️"
        
        st.info(f"{shape_icon} A distribuição é **{shape_desc}** (assimetria: {skewness:.2f})")
        
        # Interpretation
        if skewness > 1:
            st.warning("Alta assimetria positiva indica muitas transações pequenas com poucas muito grandes.")
        elif skewness < -1:
            st.warning("Alta assimetria negativa indica muitas transações grandes com poucas muito pequenas.")
        else:
            st.success("Assimetria moderada indica uma distribuição relativamente equilibrada.")
    
    with col2:
        st.markdown("#### 📏 Coeficiente de Variação")
        
        mean_vol = df_filtrado['credits_quantity'].mean()
        std_vol = df_filtrado['credits_quantity'].std()
        cv = (std_vol / mean_vol) * 100
        
        st.metric("CV", f"{cv:.1f}%", help="Desvio padrão como % da média")
        
        if cv < 50:
            st.success("Baixa variabilidade - transações são relativamente consistentes em tamanho")
        elif cv < 100:
            st.info("Variabilidade moderada - tamanhos de transação mistos")
        else:
            st.warning("Alta variabilidade - tamanhos de transação muito diversos")


def _render_distribution_charts(df_filtrado: pd.DataFrame, selected_categories: list) -> None:
    """
    Render distribution visualization charts.
    
    Args:
        df_filtrado: Filtered DataFrame
        selected_categories: Selected categories for analysis
    """
    
    st.subheader("📊 Visualizações da Distribuição")
    
    # Create tabs for different chart types
    chart_tab1, chart_tab2 = st.tabs(["📈 Histograma", "📦 Box Plots"])
    
    with chart_tab1:
        st.markdown("#### Histograma da Distribuição de Volume")
        
        # Options for histogram
        col1, col2 = st.columns(2)
        with col1:
            log_scale = st.checkbox("Usar escala logarítmica", help="Útil para distribuições altamente assimétricas")
        with col2:
            bins = st.slider("Número de barras", 10, 100, 30, help="Mais barras mostram detalhes mais finos")
        
        try:
            if log_scale:
                # Create log-scale histogram
                _render_log_histogram(df_filtrado, bins)
            else:
                # Regular histogram
                fig_hist = plotar_histograma_distribuicao(df_filtrado)
                st.pyplot(fig_hist)
            
            # Add interpretation
            _render_histogram_interpretation(df_filtrado)
            
        except Exception as e:
            st.error(f"Erro ao gerar histograma: {str(e)}")
    
    with chart_tab2:
        st.markdown("#### Distribuição de Volume por Categoria (Box Plots)")
        
        if len(selected_categories) < 2:
            st.warning("Selecione pelo menos 2 categorias para ver comparações significativas de box plot.")
        else:
            try:
                fig_box = plotar_boxplot_por_categoria(df_filtrado, selected_categories)
                st.pyplot(fig_box)
                
                # Add box plot interpretation
                _render_boxplot_interpretation(df_filtrado, selected_categories)
                
            except Exception as e:
                st.error(f"Erro ao gerar box plots: {str(e)}")


def _render_log_histogram(df_filtrado: pd.DataFrame, bins: int) -> None:
    """
    Render a logarithmic scale histogram.
    
    Args:
        df_filtrado: Filtered DataFrame
        bins: Number of histogram bins
    """
    
    import matplotlib.pyplot as plt
    
    # Filter out zero values for log scale
    positive_volumes = df_filtrado[df_filtrado['credits_quantity'] > 0]['credits_quantity']
    
    if positive_volumes.empty:
        st.warning("Nenhum volume positivo disponível para histograma de escala logarítmica.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(positive_volumes, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xscale('log')
    ax.set_xlabel('Volume de Transação (tCO₂) - Escala Logarítmica')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Volumes de Transação (Escala Logarítmica)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def _render_histogram_interpretation(df_filtrado: pd.DataFrame) -> None:
    """
    Provide interpretation for the histogram.
    
    Args:
        df_filtrado: Filtered DataFrame
    """
    
    with st.expander("📖 Como Interpretar o Histograma"):
        st.markdown("""
        **Lendo o Histograma:**
        - **Eixo X**: Faixas de volume (tCO₂)
        - **Eixo Y**: Número de transações em cada faixa
        - **Barras altas**: Tamanhos de transação comuns
        - **Barras curtas**: Tamanhos de transação raros
        
        **Padrões de Distribuição:**
        - **Um único pico**: Maioria das transações concentra-se em um tamanho
        - **Múltiplos picos**: Diferentes tipos ou segmentos de mercado
        - **Cauda longa**: Poucas transações muito grandes
        - **Uniforme**: Frequência similar em todos os tamanhos
        """)
        
        # Calculate and display mode
        volume_counts = df_filtrado['credits_quantity'].value_counts()
        if not volume_counts.empty:
            modal_volume = volume_counts.index[0]
            modal_count = volume_counts.iloc[0]
            st.info(f"📊 **Tamanho de transação mais comum**: {modal_volume:,.0f} tCO₂ ({modal_count} transações)")


def _render_boxplot_interpretation(df_filtrado: pd.DataFrame, selected_categories: list) -> None:
    """
    Provide interpretation for the box plots.
    
    Args:
        df_filtrado: Filtered DataFrame
        selected_categories: Selected categories
    """
    
    with st.expander("📖 Como Interpretar Box Plots"):
        st.markdown("""
        **Elementos do Box Plot:**
        - **Fundo da Caixa**: 25º percentil (Q1)
        - **Linha na Caixa**: Mediana (50º percentil)
        - **Topo da Caixa**: 75º percentil (Q3)
        - **Cisne**: Extende-se para pontos de dados dentro de 1.5×IQR
        - **Pontos**: Outliers além dos cisnes
        
        **Comparando Categorias:**
        - **Posição da Caixa**: Tamanhos de transação típicos
        - **Altura da Caixa**: Variabilidade dentro da categoria
        - **Comprimento dos Cisnes**: Faixa de valores típicos
        - **Outliers**: Transações incomuns grandes/pequenas
        """)
    
    # Category comparison insights
    if len(selected_categories) >= 2:
        category_stats = []
        for category in selected_categories:
            cat_data = df_filtrado[df_filtrado['project_category'] == category]['credits_quantity']
            if not cat_data.empty:
                category_stats.append({
                    'category': category,
                    'median': cat_data.median(),
                    'iqr': cat_data.quantile(0.75) - cat_data.quantile(0.25),
                    'outliers': len(cat_data[cat_data > (cat_data.quantile(0.75) + 1.5 * (cat_data.quantile(0.75) - cat_data.quantile(0.25)))])
                })
        
        if category_stats:
            stats_df = pd.DataFrame(category_stats)
            
            # Find most/least variable categories
            most_variable = stats_df.loc[stats_df['iqr'].idxmax(), 'category']
            least_variable = stats_df.loc[stats_df['iqr'].idxmin(), 'category']
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📊 **Categoria mais variável**: {most_variable}")
            with col2:
                st.info(f"📊 **Categoria menos variável**: {least_variable}")


def _render_statistical_analysis(df_filtrado: pd.DataFrame) -> None:
    """
    Render advanced statistical analysis of the distribution.
    
    Args:
        df_filtrado: Filtered DataFrame
    """
    
    st.subheader("🔬 Análise Estatística")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Análise de Percentis")
        
        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = df_filtrado['credits_quantity'].quantile([p/100 for p in percentiles])
        
        percentile_df = pd.DataFrame({
            'Percentil': [f"{p}º" for p in percentiles],
            'Volume (tCO₂)': [f"{v:,.0f}" for v in percentile_values]
        })
        
        st.dataframe(percentile_df, hide_index=True)
    
    with col2:
        st.markdown("#### 📈 Testes de Distribuição")
        
        # Normality indicators
        from scipy import stats
        
        volumes = df_filtrado['credits_quantity'].dropna()
        
        # Shapiro-Wilk test (for smaller samples)
        if len(volumes) <= 5000:
            try:
                stat, p_value = stats.shapiro(volumes.sample(min(len(volumes), 5000)))
                st.write(f"**Teste de Shapiro-Wilk**")
                st.write(f"Estatística: {stat:.4f}")
                st.write(f"P-valor: {p_value:.4e}")
                
                if p_value < 0.05:
                    st.info("❌ Não distribuído normalmente")
                else:
                    st.success("✅ Pode ser distribuído normalmente")
            except:
                st.write("Teste de normalidade não disponível")
        else:
            st.info("Dataset muito grande para teste de normalidade")
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_p = stats.jarque_bera(volumes)
            st.write(f"**Teste de Jarque-Bera**")
            st.write(f"Estatística: {jb_stat:.2f}")
            st.write(f"P-valor: {jb_p:.4e}")
        except:
            st.write("Teste de Jarque-Bera não disponível")


def _render_outlier_analysis(df_filtrado: pd.DataFrame) -> None:
    """
    Analyze and display outliers in the data.
    
    Args:
        df_filtrado: Filtered DataFrame
    """
    
    st.subheader("🎯 Análise de Outliers")
    
    volumes = df_filtrado['credits_quantity']
    
    # IQR method for outlier detection
    Q1 = volumes.quantile(0.25)
    Q3 = volumes.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_filtrado[(volumes < lower_bound) | (volumes > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total de Outliers",
            len(outliers),
            help="Transações fora do intervalo de 1.5×IQR"
        )
    
    with col2:
        outlier_percent = (len(outliers) / len(df_filtrado)) * 100
        st.metric(
            "Percentual de Outliers",
            f"{outlier_percent:.1f}%"
        )
    
    with col3:
        if len(outliers) > 0:
            max_outlier = outliers['credits_quantity'].max()
            st.metric(
                "Maior Outlier",
                f"{max_outlier:,.0f} tCO₂"
            )
    
    # Show outlier details
    if len(outliers) > 0:
        with st.expander(f"📋 Ver Detalhes dos Outliers ({len(outliers)} transações)"):
            outlier_display = outliers[['project_category', 'project_country', 'credits_quantity']].copy()
            outlier_display = outlier_display.sort_values('credits_quantity', ascending=False)
            st.dataframe(outlier_display, use_container_width=True)
    
    # Outlier impact analysis
    if len(outliers) > 0:
        outlier_volume = outliers['credits_quantity'].sum()
        total_volume = df_filtrado['credits_quantity'].sum()
        volume_impact = (outlier_volume / total_volume) * 100
        
        st.info(f"💡 **Impacto dos Outliers**: {len(outliers)} outliers ({outlier_percent:.1f}% das transações) representam {volume_impact:.1f}% do volume total")


def get_distribution_summary(df_filtrado: pd.DataFrame) -> dict:
    """
    Get a summary of distribution metrics.
    
    Args:
        df_filtrado: Filtered DataFrame
        
    Returns:
        Dictionary containing distribution summary
    """
    
    if df_filtrado.empty:
        return {}
    
    volumes = df_filtrado['credits_quantity']
    
    return {
        'mean': volumes.mean(),
        'median': volumes.median(),
        'std': volumes.std(),
        'skewness': volumes.skew(),
        'cv': (volumes.std() / volumes.mean()) * 100,
        'min': volumes.min(),
        'max': volumes.max()
    } 