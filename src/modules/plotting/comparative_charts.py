# src/modules/plotting/comparative_charts.py (Corrigido)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from ..utils.helpers import configurar_estilo_grafico

def plotar_volume_por_categoria(analise_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    if analise_df.empty: return None
    fig, ax = plt.subplots(figsize=(14, 8))
    # Corrigido: hue=analise_df.index e legend=False
    sns.barplot(x=analise_df.index, y=analise_df['total_credits_volume'], palette='viridis', ax=ax, hue=analise_df.index, legend=False)
    ax.set_title('Volume Total de Créditos por Categoria de Projeto', fontsize=16)
    ax.set_xlabel('Categoria do Projeto', fontsize=12)
    ax.set_ylabel('Volume Total de Créditos', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plotar_volume_por_pais(df_filtrado):
    plt.style.use('seaborn-v0_8-whitegrid')
    if df_filtrado.empty: return None
    volume_por_pais = df_filtrado.groupby('project_country')['credits_quantity'].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(14, 8))
    # Corrigido: hue=volume_por_pais.index e legend=False
    sns.barplot(x=volume_por_pais.index, y=volume_por_pais.values, palette='plasma', ax=ax, hue=volume_por_pais.index, legend=False)
    ax.set_title('Top 15 Países por Volume Total de Créditos', fontsize=16)
    ax.set_xlabel('País', fontsize=12)
    ax.set_ylabel('Volume Total de Créditos', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plotar_evolucao_por_safra(df_filtrado):
    plt.style.use('seaborn-v0_8-whitegrid')
    if df_filtrado.empty: return None
    volume_por_safra = df_filtrado.groupby('credit_vintage_year')['credits_quantity'].sum().sort_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x=volume_por_safra.index, y=volume_por_safra.values, marker='o', linestyle='-', ax=ax)
    ax.set_title('Evolução do Volume de Créditos por Ano de Safra', fontsize=16)
    ax.set_xlabel('Ano de Safra do Crédito', fontsize=12)
    ax.set_ylabel('Volume Total de Créditos', fontsize=12)
    if not volume_por_safra.empty:
        ax.set_xlim(volume_por_safra.index.min(), volume_por_safra.index.max())
    plt.tight_layout()
    return fig


def render_comparative_charts_tab(df: pd.DataFrame):
    """
    Render the comparative analysis tab with various comparison visualizations.
    """
    st.header("📊 Comparative Analysis")
    st.markdown("""
    **Compare carbon credit transactions across different dimensions.**
    
    This section provides:
    - **Category Comparisons** - Volume and transaction patterns by project type
    - **Geographic Analysis** - Country and regional performance
    - **Temporal Trends** - Evolution over time and vintages
    """)
    
    if df.empty:
        st.warning("No data available for comparative analysis.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Category Analysis", 
        "🌍 Geographic Analysis", 
        "📅 Temporal Analysis",
        "📈 Cross-Dimensional"
    ])
    
    with tab1:
        st.subheader("Project Category Comparison")
        
        # Category analysis
        category_analysis = df.groupby('project_category').agg({
            'credits_quantity': ['sum', 'mean', 'count', 'std'],
            'transaction_date': ['min', 'max']
        }).round(2)
        
        category_analysis.columns = [
            'total_volume', 'avg_volume', 'transaction_count', 'std_volume',
            'first_transaction', 'last_transaction'
        ]
        
        category_analysis = category_analysis.sort_values('total_volume', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Volume by Category")
            
            # Interactive bar chart
            top_categories = category_analysis.head(15)
            
            fig_category = px.bar(
                x=top_categories.index,
                y=top_categories['total_volume'],
                title="Total Volume by Project Category (Top 15)",
                labels={'x': 'Project Category', 'y': 'Total Volume'},
                template='plotly_white'
            )
            fig_category.update_xaxis(tickangle=45)
            fig_category.update_layout(height=500)
            
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Top Categories")
            
            # Top categories metrics
            for i, (category, data) in enumerate(top_categories.head(5).iterrows(), 1):
                st.metric(
                    f"{i}. {category[:20]}{'...' if len(category) > 20 else ''}",
                    f"{data['total_volume']:,.0f}",
                    f"{data['transaction_count']:,.0f} transactions"
                )
        
        # Category comparison table
        st.markdown("### 📋 Category Statistics")
        
        display_categories = category_analysis.head(10).copy()
        display_categories['total_volume'] = display_categories['total_volume'].apply(lambda x: f"{x:,.0f}")
        display_categories['avg_volume'] = display_categories['avg_volume'].apply(lambda x: f"{x:,.0f}")
        display_categories['transaction_count'] = display_categories['transaction_count'].apply(lambda x: f"{x:,.0f}")
        display_categories['std_volume'] = display_categories['std_volume'].apply(lambda x: f"{x:,.0f}")
        
        display_categories.columns = [
            'Total Volume', 'Avg Volume', 'Transactions', 'Std Deviation',
            'First Transaction', 'Last Transaction'
        ]
        
        st.dataframe(display_categories, use_container_width=True)
        
        # Market share analysis
        st.markdown("### 🥧 Market Share")
        
        total_market_volume = category_analysis['total_volume'].sum()
        category_analysis['market_share'] = (category_analysis['total_volume'] / total_market_volume * 100).round(2)
        
        # Pie chart for top categories
        top_10_categories = category_analysis.head(10)
        others_volume = category_analysis.iloc[10:]['total_volume'].sum() if len(category_analysis) > 10 else 0
        
        if others_volume > 0:
            pie_data = pd.concat([
                top_10_categories[['total_volume']],
                pd.DataFrame({'total_volume': [others_volume]}, index=['Others'])
            ])
        else:
            pie_data = top_10_categories[['total_volume']]
        
        fig_pie = px.pie(
            values=pie_data['total_volume'],
            names=pie_data.index,
            title="Market Share by Category (Top 10 + Others)",
            template='plotly_white'
        )
        fig_pie.update_layout(height=500)
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.subheader("Geographic Distribution Analysis")
        
        # Country analysis
        country_analysis = df.groupby('project_country').agg({
            'credits_quantity': ['sum', 'mean', 'count'],
            'project_category': 'nunique'
        }).round(2)
        
        country_analysis.columns = ['total_volume', 'avg_volume', 'transaction_count', 'category_count']
        country_analysis = country_analysis.sort_values('total_volume', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🌍 Volume by Country")
            
            # Top countries bar chart
            top_countries = country_analysis.head(15)
            
            fig_country = px.bar(
                x=top_countries.index,
                y=top_countries['total_volume'],
                title="Total Volume by Country (Top 15)",
                labels={'x': 'Country', 'y': 'Total Volume'},
                template='plotly_white'
            )
            fig_country.update_xaxis(tickangle=45)
            fig_country.update_layout(height=500)
            
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Top Countries")
            
            for i, (country, data) in enumerate(top_countries.head(5).iterrows(), 1):
                st.metric(
                    f"{i}. {country}",
                    f"{data['total_volume']:,.0f}",
                    f"{data['category_count']} categories"
                )
        
        # Geographic heatmap (if coordinates were available, we'd use a map)
        st.markdown("### 📊 Country Performance Matrix")
        
        # Scatter plot: Volume vs Diversity
        fig_scatter = px.scatter(
            country_analysis.head(20),
            x='category_count',
            y='total_volume',
            size='transaction_count',
            hover_name=country_analysis.head(20).index,
            title="Country Performance: Volume vs Category Diversity",
            labels={
                'category_count': 'Number of Categories',
                'total_volume': 'Total Volume',
                'transaction_count': 'Transaction Count'
            },
            template='plotly_white'
        )
        fig_scatter.update_layout(height=500)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Country statistics table
        st.markdown("### 📋 Country Statistics")
        
        display_countries = country_analysis.head(10).copy()
        display_countries['total_volume'] = display_countries['total_volume'].apply(lambda x: f"{x:,.0f}")
        display_countries['avg_volume'] = display_countries['avg_volume'].apply(lambda x: f"{x:,.0f}")
        display_countries['transaction_count'] = display_countries['transaction_count'].apply(lambda x: f"{x:,.0f}")
        
        display_countries.columns = ['Total Volume', 'Avg Volume', 'Transactions', 'Categories']
        
        st.dataframe(display_countries, use_container_width=True)
    
    with tab3:
        st.subheader("Temporal Analysis")
        
        # Time-based analysis
        df['transaction_year'] = pd.to_datetime(df['transaction_date']).dt.year
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Volume Evolution by Transaction Year")
            
            yearly_volume = df.groupby('transaction_year')['credits_quantity'].sum().reset_index()
            
            fig_yearly = px.line(
                yearly_volume,
                x='transaction_year',
                y='credits_quantity',
                title="Total Volume by Transaction Year",
                labels={'transaction_year': 'Year', 'credits_quantity': 'Total Volume'},
                template='plotly_white',
                markers=True
            )
            fig_yearly.update_layout(height=400)
            
            st.plotly_chart(fig_yearly, use_container_width=True)
        
        with col2:
            st.markdown("### 🍷 Volume by Credit Vintage Year")
            
            vintage_volume = df.groupby('credit_vintage_year')['credits_quantity'].sum().reset_index()
            
            fig_vintage = px.line(
                vintage_volume,
                x='credit_vintage_year',
                y='credits_quantity',
                title="Total Volume by Credit Vintage Year",
                labels={'credit_vintage_year': 'Vintage Year', 'credits_quantity': 'Total Volume'},
                template='plotly_white',
                markers=True
            )
            fig_vintage.update_layout(height=400)
            
            st.plotly_chart(fig_vintage, use_container_width=True)
        
        # Age analysis
        st.markdown("### ⏰ Credit Age Analysis")
        
        age_bins = [0, 365, 730, 1095, 1460, float('inf')]
        age_labels = ['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4+ years']
        
        df['age_category'] = pd.cut(df['credit_age_at_transaction'], bins=age_bins, labels=age_labels, right=False)
        age_analysis = df.groupby('age_category')['credits_quantity'].agg(['sum', 'count', 'mean']).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age_volume = px.bar(
                x=age_analysis.index,
                y=age_analysis['sum'],
                title="Total Volume by Credit Age",
                labels={'x': 'Credit Age Category', 'y': 'Total Volume'},
                template='plotly_white'
            )
            st.plotly_chart(fig_age_volume, use_container_width=True)
        
        with col2:
            fig_age_count = px.bar(
                x=age_analysis.index,
                y=age_analysis['count'],
                title="Transaction Count by Credit Age",
                labels={'x': 'Credit Age Category', 'y': 'Transaction Count'},
                template='plotly_white'
            )
            st.plotly_chart(fig_age_count, use_container_width=True)
    
    with tab4:
        st.subheader("Cross-Dimensional Analysis")
        
        st.markdown("### 🔄 Interactive Multi-Dimensional Explorer")
        
        # Multi-dimensional analysis controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_dimension = st.selectbox(
                "X-Axis Dimension",
                ['project_category', 'project_country', 'transaction_year', 'credit_vintage_year'],
                index=0
            )
        
        with col2:
            y_dimension = st.selectbox(
                "Y-Axis Metric",
                ['credits_quantity', 'credit_age_at_transaction'],
                index=0
            )
        
        with col3:
            aggregation = st.selectbox(
                "Aggregation Method",
                ['sum', 'mean', 'median', 'count'],
                index=0
            )
        
        # Create cross-dimensional analysis
        if x_dimension and y_dimension:
            cross_analysis = df.groupby(x_dimension)[y_dimension].agg(aggregation).sort_values(ascending=False).head(15)
            
            fig_cross = px.bar(
                x=cross_analysis.index,
                y=cross_analysis.values,
                title=f"{aggregation.title()} of {y_dimension} by {x_dimension}",
                labels={'x': x_dimension.replace('_', ' ').title(), 'y': f"{aggregation.title()} {y_dimension}"},
                template='plotly_white'
            )
            fig_cross.update_xaxis(tickangle=45)
            fig_cross.update_layout(height=500)
            
            st.plotly_chart(fig_cross, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### 🔗 Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Numeric Variables",
                template='plotly_white'
            )
            fig_corr.update_layout(height=500)
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Summary insights
        st.markdown("### 💡 Key Insights")
        
        total_volume = df['credits_quantity'].sum()
        total_transactions = len(df)
        unique_categories = df['project_category'].nunique()
        unique_countries = df['project_country'].nunique()
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.metric("Total Market Volume", f"{total_volume:,.0f} credits")
            st.metric("Total Transactions", f"{total_transactions:,}")
        
        with insights_col2:
            st.metric("Project Categories", f"{unique_categories}")
            st.metric("Countries Involved", f"{unique_countries}")
        
        # Market concentration
        top_5_categories_share = (category_analysis.head(5)['total_volume'].sum() / total_volume * 100)
        top_5_countries_share = (country_analysis.head(5)['total_volume'].sum() / total_volume * 100)
        
        st.info(f"""
        **Market Concentration:**
        - Top 5 categories represent {top_5_categories_share:.1f}% of total volume
        - Top 5 countries represent {top_5_countries_share:.1f}% of total volume
        - Average transaction size: {df['credits_quantity'].mean():,.0f} credits
        - Market activity span: {df['transaction_year'].min()} to {df['transaction_year'].max()}
        """)