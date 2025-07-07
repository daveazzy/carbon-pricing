# src/modules/plotting/comparative_charts.py (Corrigido)
import matplotlib.pyplot as plt
import seaborn as sns
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