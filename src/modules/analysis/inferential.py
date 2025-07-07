# src/modules/analysis/inferential.py

import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

def realizar_teste_mann_whitney(df, categoria1, categoria2, sample_size=5000):
    """
    Realiza o Teste de Mann-Whitney U para comparar as distribuições de duas categorias.

    Retorna:
        tuple: (Estatística U, p-valor) ou (None, None) se não for possível realizar o teste.
    """
    if df.empty or categoria1 not in df['project_category'].unique() or categoria2 not in df['project_category'].unique():
        return None, None

    cat1_quantities = df[df['project_category'] == categoria1]['credits_quantity'].dropna()
    cat2_quantities = df[df['project_category'] == categoria2]['credits_quantity'].dropna()

    if len(cat1_quantities) < 10 or len(cat2_quantities) < 10:
        return None, None # Dados insuficientes para um teste minimamente razoável

    # Se as amostras forem grandes, usa uma sub-amostra para eficiência. Caso contrário, usa os dados completos.
    if len(cat1_quantities) > sample_size:
        cat1_sample = cat1_quantities.sample(sample_size, random_state=42)
    else:
        cat1_sample = cat1_quantities

    if len(cat2_quantities) > sample_size:
        cat2_sample = cat2_quantities.sample(sample_size, random_state=42)
    else:
        cat2_sample = cat2_quantities
    
    stat, p_value = mannwhitneyu(cat1_sample, cat2_sample, alternative='two-sided')
    
    return stat, p_value

def calcular_tabela_contingencia(df, top_n_countries=5, top_n_categories=5):
    """
    Cria uma tabela de contingência entre os principais países e categorias.
    """
    if df.empty:
        return pd.DataFrame()

    top_countries = df.groupby('project_country')['credits_quantity'].sum().nlargest(top_n_countries).index
    top_categories = df.groupby('project_category')['credits_quantity'].sum().nlargest(top_n_categories).index

    test_df = df[
        df['project_country'].isin(top_countries) & 
        df['project_category'].isin(top_categories)
    ]
    
    if test_df.empty:
        return pd.DataFrame()

    contingency_table = pd.crosstab(test_df['project_country'], test_df['project_category'])
    return contingency_table

def realizar_teste_qui_quadrado(contingency_table):
    """
    Realiza o Teste Qui-Quadrado de Independência numa tabela de contingência.
    """
    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None, None
        
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return chi2, p_value
    except ValueError:
        return None, None # Ocorre se a tabela for inadequada