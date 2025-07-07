# src/modules/analysis/modeling.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

def treinar_modelo_regressao(df):
    """
    Prepara os dados e treina um modelo de regressão linear (OLS) para
    explicar o logaritmo do volume das transações.

    Retorna:
        statsmodels.results.regression.RegressionResultsWrapper: O objeto de resultados do modelo treinado.
    """
    if df.empty:
        return None

    # 1. Preparar os dados para o modelo
    # Para um modelo robusto, vamos usar as 5 categorias com maior volume
    top_5_categories = df.groupby('project_category')['credits_quantity'].sum().nlargest(5).index
    model_df = df[df['project_category'].isin(top_5_categories)].copy()

    # Aplicar a transformação logarítmica para lidar com a assimetria
    model_df['log_credits_quantity'] = np.log1p(model_df['credits_quantity'])
    
    # Substituir valores infinitos (caso a transformação log gere algum) por NaN
    model_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remover linhas com dados faltantes nas colunas essenciais para o modelo
    model_df.dropna(subset=['log_credits_quantity', 'credit_age_at_transaction', 'project_category'], inplace=True)

    if model_df.empty:
        return None

    # 2. Definir as variáveis Y (dependente) e X (independentes)
    Y = model_df['log_credits_quantity']
    X = model_df[['credit_age_at_transaction', 'project_category']]

    # Converter a variável categórica 'project_category' em variáveis dummy
    # O dtype=float é importante para evitar erros de tipo no statsmodels
    X = pd.get_dummies(X, columns=['project_category'], drop_first=True, dtype=float)

    # Adicionar uma constante (o intercepto) ao modelo
    X = sm.add_constant(X)

    # 3. Construir e treinar o modelo
    model = sm.OLS(Y, X).fit()

    return model