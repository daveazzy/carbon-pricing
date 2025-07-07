# src/modules/plotting/distribution_charts.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def plotar_histograma_distribuicao(df):
    """
    Cria e retorna um histograma da distribuição dos tamanhos das transações,
    usando uma escala logarítmica para melhor visualização.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if df.empty or df['credits_quantity'].isnull().all():
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.histplot(data=df, x='credits_quantity', log_scale=True, bins=50, kde=True, ax=ax)
    
    ax.set_title('Distribuição dos Tamanhos das Transações (Escala Logarítmica)', fontsize=16)
    ax.set_xlabel('Quantidade de Créditos (Escala Logarítmica)', fontsize=12)
    ax.set_ylabel('Frequência (Nº de Transações)', fontsize=12)
    plt.tight_layout()
    
    return fig

def plotar_boxplot_por_categoria(df, categorias_selecionadas):
    """
    Cria e retorna boxplots para comparar as distribuições de volume para as
    categorias selecionadas.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if df.empty or not categorias_selecionadas:
        return None

    # Usa as categorias que foram passadas como argumento
    df_para_plot = df[df['project_category'].isin(categorias_selecionadas)]

    fig, ax = plt.subplots(figsize=(16, 9))
    
    sns.boxplot(data=df_para_plot, x='project_category', y='credits_quantity', palette='magma', ax=ax, hue='project_category', legend=False)
    
    ax.set_yscale('log')
    ax.set_title('Comparação das Distribuições por Categoria (Escala Logarítmica)', fontsize=16)
    ax.set_xlabel('Categoria do Projeto', fontsize=12)
    ax.set_ylabel('Quantidade de Créditos (Escala Logarítmica)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def render_distribution_charts_tab(df: pd.DataFrame):
    """
    Render the distribution analysis tab with various distribution visualizations.
    """
    st.header("📊 Distribution Analysis")
    st.markdown("""
    **Explore the statistical distributions of carbon credit transactions.**
    
    This section provides:
    - **Distribution Overview** - Histograms and density plots
    - **Category Comparisons** - Box plots and violin plots
    - **Statistical Summaries** - Key distribution metrics
    """)
    
    if df.empty:
        st.warning("No data available for distribution analysis.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 Overall Distribution", "📦 Category Distributions", "📋 Statistical Summary"])
    
    with tab1:
        st.subheader("Transaction Volume Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Interactive Histogram")
            
            # Interactive histogram with Plotly
            fig_hist = px.histogram(
                df, 
                x='credits_quantity',
                nbins=50,
                title="Distribution of Transaction Volumes",
                labels={'credits_quantity': 'Credits Quantity', 'count': 'Frequency'},
                template='plotly_white'
            )
            fig_hist.update_xaxis(type="log", title="Credits Quantity (Log Scale)")
            fig_hist.update_layout(height=500)
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Key Statistics")
            
            # Distribution statistics
            stats_data = df['credits_quantity'].describe()
            
            st.metric("Total Transactions", f"{len(df):,}")
            st.metric("Mean Volume", f"{stats_data['mean']:,.0f}")
            st.metric("Median Volume", f"{stats_data['50%']:,.0f}")
            st.metric("Max Volume", f"{stats_data['max']:,.0f}")
            
            # Skewness and Kurtosis
            from scipy.stats import skew, kurtosis
            skewness = skew(df['credits_quantity'].dropna())
            kurt = kurtosis(df['credits_quantity'].dropna())
            
            st.metric("Skewness", f"{skewness:.2f}")
            st.metric("Kurtosis", f"{kurt:.2f}")
        
        # Density plot
        st.markdown("### 🌊 Density Distribution")
        
        # Create density plot
        fig_density = go.Figure()
        
        # Calculate kernel density estimation
        from scipy.stats import gaussian_kde
        log_data = np.log1p(df['credits_quantity'].dropna())
        kde = gaussian_kde(log_data)
        x_range = np.linspace(log_data.min(), log_data.max(), 100)
        density = kde(x_range)
        
        fig_density.add_trace(go.Scatter(
            x=np.expm1(x_range),
            y=density,
            mode='lines',
            fill='tonexty',
            name='Density',
            line=dict(color='blue', width=2)
        ))
        
        fig_density.update_layout(
            title="Transaction Volume Density Distribution",
            xaxis_title="Credits Quantity",
            yaxis_title="Density",
            xaxis_type="log",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        # Distribution interpretation
        st.markdown("### 💡 Distribution Insights")
        
        if skewness > 1:
            skew_interpretation = "**Highly right-skewed** - Most transactions are small with few very large ones"
        elif skewness > 0.5:
            skew_interpretation = "**Moderately right-skewed** - Some concentration in smaller transactions"
        else:
            skew_interpretation = "**Relatively symmetric** - Balanced distribution of transaction sizes"
        
        st.info(f"""
        **Distribution Characteristics:**
        - {skew_interpretation}
        - **Range:** {stats_data['min']:,.0f} to {stats_data['max']:,.0f} credits
        - **Interquartile Range:** {stats_data['25%']:,.0f} to {stats_data['75%']:,.0f} credits
        - **Coefficient of Variation:** {(stats_data['std']/stats_data['mean']*100):.1f}%
        """)
    
    with tab2:
        st.subheader("Distribution by Category")
        
        # Category selection
        categories = df['project_category'].unique()
        selected_categories = st.multiselect(
            "Select categories to compare:",
            categories,
            default=list(categories[:5]) if len(categories) > 5 else list(categories),
            help="Choose categories for distribution comparison"
        )
        
        if selected_categories:
            filtered_df = df[df['project_category'].isin(selected_categories)]
            
            # Box plots
            st.markdown("### 📦 Box Plot Comparison")
            
            fig_box = px.box(
                filtered_df,
                x='project_category',
                y='credits_quantity',
                title="Transaction Volume Distribution by Category",
                labels={'project_category': 'Project Category', 'credits_quantity': 'Credits Quantity'},
                template='plotly_white'
            )
            fig_box.update_yaxis(type="log", title="Credits Quantity (Log Scale)")
            fig_box.update_xaxis(tickangle=45)
            fig_box.update_layout(height=500)
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Violin plots
            st.markdown("### 🎻 Violin Plot Comparison")
            
            fig_violin = px.violin(
                filtered_df,
                x='project_category',
                y='credits_quantity',
                title="Detailed Distribution Shapes by Category",
                labels={'project_category': 'Project Category', 'credits_quantity': 'Credits Quantity'},
                template='plotly_white',
                box=True
            )
            fig_violin.update_yaxis(type="log", title="Credits Quantity (Log Scale)")
            fig_violin.update_xaxis(tickangle=45)
            fig_violin.update_layout(height=500)
            
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # Category statistics table
            st.markdown("### 📊 Category Statistics")
            
            category_stats = []
            for category in selected_categories:
                cat_data = filtered_df[filtered_df['project_category'] == category]['credits_quantity']
                
                if not cat_data.empty:
                    category_stats.append({
                        'Category': category,
                        'Count': len(cat_data),
                        'Mean': f"{cat_data.mean():,.0f}",
                        'Median': f"{cat_data.median():,.0f}",
                        'Std Dev': f"{cat_data.std():,.0f}",
                        'Min': f"{cat_data.min():,.0f}",
                        'Max': f"{cat_data.max():,.0f}",
                        'CV (%)': f"{(cat_data.std()/cat_data.mean()*100):.1f}"
                    })
            
            if category_stats:
                stats_df = pd.DataFrame(category_stats)
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
        else:
            st.info("Please select at least one category to display distributions.")
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Comprehensive statistics
        st.markdown("### 📋 Descriptive Statistics")
        
        # Overall statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Central Tendencies")
            central_stats = df['credits_quantity'].agg(['mean', 'median', 'mode']).to_frame('Value')
            central_stats.index = ['Mean', 'Median', 'Mode']
            central_stats['Value'] = central_stats['Value'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(central_stats)
        
        with col2:
            st.markdown("#### 📏 Variability Measures")
            variability_stats = df['credits_quantity'].agg(['std', 'var', 'min', 'max']).to_frame('Value')
            variability_stats.index = ['Standard Deviation', 'Variance', 'Minimum', 'Maximum']
            variability_stats['Value'] = variability_stats['Value'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(variability_stats)
        
        # Percentiles
        st.markdown("### 📊 Percentile Analysis")
        
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [df['credits_quantity'].quantile(p/100) for p in percentiles]
        
        fig_percentiles = go.Figure()
        fig_percentiles.add_trace(go.Bar(
            x=[f"{p}th" for p in percentiles],
            y=percentile_values,
            text=[f"{v:,.0f}" for v in percentile_values],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig_percentiles.update_layout(
            title="Transaction Volume Percentiles",
            xaxis_title="Percentile",
            yaxis_title="Credits Quantity",
            yaxis_type="log",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_percentiles, use_container_width=True)
        
        # Distribution tests
        st.markdown("### 🧪 Distribution Tests")
        
        test_col1, test_col2 = st.columns(2)
        
        with test_col1:
            # Normality test
            from scipy.stats import shapiro, normaltest
            
            # Sample for testing (Shapiro-Wilk has sample size limitations)
            sample_size = min(5000, len(df))
            sample_data = df['credits_quantity'].sample(sample_size, random_state=42)
            
            try:
                stat, p_value = normaltest(sample_data)
                st.metric(
                    "D'Agostino Normality Test",
                    f"Statistic: {stat:.4f}",
                    f"p-value: {p_value:.2e}"
                )
                
                if p_value > 0.05:
                    st.success("✅ Data appears normally distributed")
                else:
                    st.warning("⚠️ Data is not normally distributed")
            except Exception:
                st.info("Normality test not available")
        
        with test_col2:
            # Anderson-Darling test
            try:
                from scipy.stats import anderson
                result = anderson(np.log1p(sample_data), dist='norm')
                
                st.metric(
                    "Anderson-Darling Test (Log-transformed)",
                    f"Statistic: {result.statistic:.4f}",
                    f"Critical Value (5%): {result.critical_values[2]:.4f}"
                )
                
                if result.statistic < result.critical_values[2]:
                    st.success("✅ Log-transformed data fits normal distribution")
                else:
                    st.warning("⚠️ Log-transformed data doesn't fit normal distribution")
            except Exception:
                st.info("Anderson-Darling test not available")
        
        # Summary insights
        st.markdown("### 💡 Key Insights")
        
        total_volume = df['credits_quantity'].sum()
        mean_volume = df['credits_quantity'].mean()
        median_volume = df['credits_quantity'].median()
        
        insights = f"""
        **Transaction Volume Analysis:**
        
        - **Total Volume Traded:** {total_volume:,.0f} carbon credits
        - **Average Transaction:** {mean_volume:,.0f} credits
        - **Typical Transaction (Median):** {median_volume:,.0f} credits
        - **Market Concentration:** Top 10% of transactions represent {((df['credits_quantity'].quantile(0.9) * len(df) * 0.1) / total_volume * 100):.1f}% of total volume
        - **Distribution Type:** {'Heavy-tailed' if skewness > 1 else 'Moderately skewed' if skewness > 0.5 else 'Symmetric'}
        
        **Recommendations:**
        - Use log-scale visualizations for better data interpretation
        - Consider median values for typical transaction analysis
        - Apply appropriate statistical tests for skewed distributions
        """
        
        st.markdown(insights)