# src/modules/plotting/distribution_charts.py

import matplotlib.pyplot as plt
import seaborn as sns

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