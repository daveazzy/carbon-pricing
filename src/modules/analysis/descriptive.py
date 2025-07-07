import pandas as pd
import numpy as np

def calcular_analise_enriquecida_por_categoria(df):
    """
    Agrupa o dataframe por categoria de projeto e calcula um conjunto
    rico de estatísticas descritivas, incluindo o intervalo de confiança.
    """
    if df.empty:
        return pd.DataFrame()

    analise = df.groupby('project_category').agg(
        total_credits_volume=('credits_quantity', 'sum'),
        number_of_transactions=('credits_quantity', 'count'),
        mean_transaction_volume=('credits_quantity', 'mean'),
        median_transaction_volume=('credits_quantity', 'median'),
        std_dev_of_volume=('credits_quantity', 'std'),
        min_transaction_volume=('credits_quantity', 'min'),
        max_transaction_volume=('credits_quantity', 'max')
    ).sort_values(by='total_credits_volume', ascending=False)
    
    # Cálculo do Intervalo de Confiança 95%
    z_score = 1.96
    margin_of_error = z_score * (analise['std_dev_of_volume'] / np.sqrt(analise['number_of_transactions']))
    analise['ci_95_lower_bound'] = analise['mean_transaction_volume'] - margin_of_error
    analise['ci_95_upper_bound'] = analise['mean_transaction_volume'] + margin_of_error

    return analise