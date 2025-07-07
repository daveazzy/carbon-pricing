import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go


def render_inferential_analysis_tab(df):
    
    st.header("🔬 Análise Inferencial")
    
    tab1, tab2, tab3 = st.tabs(["Testes de Hipóteses", "Correlações", "ANOVA"])
    
    with tab1:
        render_hypothesis_tests(df)
    
    with tab2:
        render_correlation_analysis(df)
    
    with tab3:
        render_anova_analysis(df)


def render_hypothesis_tests(df):
    
    st.subheader("Testes de Hipóteses")
    
    test_type = st.selectbox(
        "Selecione o tipo de teste:",
        ["Mann-Whitney U", "Kolmogorov-Smirnov", "T-Test"]
    )
    
    categories = df['project_category'].unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        cat1 = st.selectbox("Categoria 1:", categories, key="cat1")
    
    with col2:
        cat2 = st.selectbox("Categoria 2:", categories, key="cat2", index=1)
    
    if st.button("Executar Teste"):
        data1 = df[df['project_category'] == cat1]['credits_quantity']
        data2 = df[df['project_category'] == cat2]['credits_quantity']
        
        if test_type == "Mann-Whitney U":
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        elif test_type == "Kolmogorov-Smirnov":
            statistic, p_value = stats.ks_2samp(data1, data2)
            test_name = "Kolmogorov-Smirnov"
        
        else:
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_name = "T-Test"
        
        st.success(f"""
        **Resultado do {test_name}:**
        - Estatística: {statistic:.4f}
        - P-valor: {p_value:.4f}
        - Significativo (α=0.05): {'Sim' if p_value < 0.05 else 'Não'}
        """)


def render_correlation_analysis(df):
    
    st.subheader("Análise de Correlações")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlações Mais Fortes")
        
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                corr_pairs.append((col1, col2, corr_value))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for col1, col2, corr in corr_pairs[:5]:
            st.write(f"**{col1}** vs **{col2}**: {corr:.3f}")


def render_anova_analysis(df):
    
    st.subheader("Análise ANOVA")
    
    selected_categories = st.multiselect(
        "Selecione categorias para comparação:",
        df['project_category'].unique(),
        default=df['project_category'].value_counts().head(3).index.tolist()
    )
    
    if len(selected_categories) >= 2 and st.button("Executar ANOVA"):
        
        groups = []
        for cat in selected_categories:
            group_data = df[df['project_category'] == cat]['credits_quantity']
            groups.append(group_data)
        
        f_statistic, p_value = stats.f_oneway(*groups)
        
        st.success(f"""
        **Resultado da ANOVA:**
        - F-estatística: {f_statistic:.4f}
        - P-valor: {p_value:.4f}
        - Significativo (α=0.05): {'Sim' if p_value < 0.05 else 'Não'}
        """)
        
        if p_value < 0.05:
            st.info("As médias dos grupos são significativamente diferentes.")
        else:
            st.warning("Não há diferença significativa entre as médias dos grupos.")
        
        st.subheader("Estatísticas Descritivas por Grupo")
        
        for cat in selected_categories:
            group_data = df[df['project_category'] == cat]['credits_quantity']
            st.write(f"""
            **{cat}:**
            - Média: {group_data.mean():.2f}
            - Desvio Padrão: {group_data.std():.2f}
            - Mediana: {group_data.median():.2f}
            - N: {len(group_data)}
            """)


def realizar_teste_mann_whitney(grupo1, grupo2):
    
    statistic, p_value = stats.mannwhitneyu(grupo1, grupo2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': 'Diferença significativa' if p_value < 0.05 else 'Sem diferença significativa'
    }


def calcular_tabela_contingencia(df, var1, var2):
    
    contingency_table = pd.crosstab(df[var1], df[var2])
    return contingency_table


def realizar_teste_qui_quadrado(contingency_table):
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'significant': p_value < 0.05
    }