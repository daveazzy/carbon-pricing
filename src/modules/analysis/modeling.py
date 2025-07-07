# src/modules/analysis/modeling.py

import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def render_modeling_tab(df: pd.DataFrame):
    """
    Render the modeling analysis tab with regression analysis and statistical modeling.
    """
    st.header("🔬 Statistical Modeling Analysis")
    st.markdown("""
    **Advanced statistical modeling to understand relationships in carbon credit transactions.**
    
    This section includes:
    - **Regression Analysis** - OLS modeling to understand transaction volume drivers
    - **Model Diagnostics** - Statistical tests and residual analysis  
    - **Predictive Insights** - How variables affect transaction volumes
    """)
    
    if df.empty:
        st.warning("No data available for modeling analysis.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Regression Model", "🔍 Model Diagnostics", "📈 Predictions"])
    
    with tab1:
        st.subheader("Linear Regression Analysis")
        st.markdown("**Modeling log-transformed transaction volumes using OLS regression**")
        
        with st.spinner("Training regression model..."):
            model = treinar_modelo_regressao(df)
        
        if model is None:
            st.error("Unable to train model. Insufficient data or data quality issues.")
            return
        
        # Model Summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Model Summary")
            
            # Key metrics
            summary_data = {
                'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'P-value (F-stat)', 'AIC', 'BIC'],
                'Value': [
                    f"{model.rsquared:.4f}",
                    f"{model.rsquared_adj:.4f}", 
                    f"{model.fvalue:.2f}",
                    f"{model.f_pvalue:.2e}",
                    f"{model.aic:.2f}",
                    f"{model.bic:.2f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(summary_data), hide_index=True)
            
            # Interpretation
            r2 = model.rsquared
            if r2 > 0.7:
                interpretation = "🟢 **Strong model** - Explains most variance in transaction volumes"
            elif r2 > 0.4:
                interpretation = "🟡 **Moderate model** - Explains some variance, room for improvement"
            else:
                interpretation = "🔴 **Weak model** - Low explanatory power, consider additional variables"
            
            st.info(interpretation)
        
        with col2:
            st.markdown("### 🎯 Model Performance")
            
            # Performance gauge
            performance_score = int(model.rsquared * 100)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = performance_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "R² Score (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Coefficients Analysis
        st.markdown("### 🔢 Regression Coefficients")
        
        # Extract coefficient information
        coef_data = []
        for var, coef in model.params.items():
            p_value = model.pvalues[var]
            conf_int = model.conf_int().loc[var]
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            coef_data.append({
                'Variable': var,
                'Coefficient': f"{coef:.4f}",
                'P-value': f"{p_value:.4f}",
                'Significance': significance,
                '95% CI Lower': f"{conf_int[0]:.4f}",
                '95% CI Upper': f"{conf_int[1]:.4f}"
            })
        
        coef_df = pd.DataFrame(coef_data)
        st.dataframe(coef_df, hide_index=True)
        
        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Coefficient visualization
        st.markdown("### 📊 Coefficient Plot")
        
        # Create coefficient plot
        variables = [var for var in model.params.index if var != 'const']
        coefficients = [model.params[var] for var in variables]
        conf_intervals = [model.conf_int().loc[var] for var in variables]
        
        fig_coef = go.Figure()
        
        for i, (var, coef) in enumerate(zip(variables, coefficients)):
            ci = conf_intervals[i]
            fig_coef.add_trace(go.Scatter(
                x=[ci[0], ci[1]],
                y=[var, var],
                mode='lines',
                line=dict(color='lightblue', width=8),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_coef.add_trace(go.Scatter(
                x=[coef],
                y=[var],
                mode='markers',
                marker=dict(color='blue', size=12),
                showlegend=False,
                hovertemplate=f'{var}<br>Coefficient: {coef:.4f}<extra></extra>'
            ))
        
        fig_coef.add_vline(x=0, line_dash="dash", line_color="red")
        fig_coef.update_layout(
            title="Regression Coefficients with 95% Confidence Intervals",
            xaxis_title="Coefficient Value",
            yaxis_title="Variables",
            height=400
        )
        
        st.plotly_chart(fig_coef, use_container_width=True)
    
    with tab2:
        st.subheader("Model Diagnostics")
        
        if model is None:
            st.warning("No model available for diagnostics.")
            return
        
        # Residual Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Residuals vs Fitted")
            
            fitted_values = model.fittedvalues
            residuals = model.resid
            
            fig_resid = px.scatter(
                x=fitted_values, 
                y=residuals,
                title="Residuals vs Fitted Values",
                labels={'x': 'Fitted Values', 'y': 'Residuals'},
                template='plotly_white'
            )
            
            # Add horizontal line at y=0
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_resid, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Q-Q Plot")
            
            # Calculate theoretical quantiles for normal distribution
            from scipy import stats
            theoretical_quantiles = stats.probplot(residuals, dist="norm")[0][0]
            sample_quantiles = stats.probplot(residuals, dist="norm")[0][1]
            
            fig_qq = px.scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                title="Q-Q Plot: Residuals vs Normal Distribution",
                labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                template='plotly_white'
            )
            
            # Add diagonal line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ))
            
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Statistical Tests
        st.markdown("### 🧪 Statistical Tests")
        
        test_col1, test_col2 = st.columns(2)
        
        with test_col1:
            # Jarque-Bera test for normality
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(residuals)
            
            st.metric(
                "Jarque-Bera Test (Normality)",
                f"Statistic: {jb_stat:.4f}",
                f"p-value: {jb_pvalue:.4f}"
            )
            
            if jb_pvalue > 0.05:
                st.success("✅ Residuals appear normally distributed")
            else:
                st.warning("⚠️ Residuals may not be normally distributed")
        
        with test_col2:
            # Breusch-Pagan test for heteroscedasticity
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, model.model.exog)
                
                st.metric(
                    "Breusch-Pagan Test (Heteroscedasticity)",
                    f"LM Statistic: {lm_stat:.4f}",
                    f"p-value: {lm_pvalue:.4f}"
                )
                
                if lm_pvalue > 0.05:
                    st.success("✅ Homoscedasticity assumption met")
                else:
                    st.warning("⚠️ Heteroscedasticity detected")
            except Exception as e:
                st.info("Heteroscedasticity test not available")
        
        # Model Summary Text
        st.markdown("### 📝 Detailed Model Summary")
        with st.expander("Show Full Statistical Summary"):
            st.text(str(model.summary()))
    
    with tab3:
        st.subheader("Model Predictions & Insights")
        
        if model is None:
            st.warning("No model available for predictions.")
            return
        
        st.markdown("### 🔮 Interactive Prediction Tool")
        st.markdown("Estimate log-transaction volume based on model inputs:")
        
        # Get available categories from the model
        available_categories = []
        for col in model.model.exog_names:
            if col.startswith('project_category_'):
                category_name = col.replace('project_category_', '')
                available_categories.append(category_name)
        
        # Prediction interface
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            age_input = st.slider(
                "Credit Age at Transaction (days)",
                min_value=0,
                max_value=int(df['credit_age_at_transaction'].max()) if 'credit_age_at_transaction' in df.columns else 1000,
                value=365,
                help="Age of carbon credits when transaction occurred"
            )
        
        with pred_col2:
            if available_categories:
                category_input = st.selectbox(
                    "Project Category",
                    available_categories,
                    help="Carbon credit project category"
                )
            else:
                category_input = None
                st.info("No categories available for prediction")
        
        if st.button("🔍 Calculate Prediction", type="primary"):
            try:
                # Prepare prediction data
                pred_data = {'const': 1, 'credit_age_at_transaction': age_input}
                
                # Set all category dummies to 0
                for col in model.model.exog_names:
                    if col.startswith('project_category_'):
                        pred_data[col] = 0
                
                # Set selected category to 1
                if category_input:
                    selected_col = f'project_category_{category_input}'
                    if selected_col in pred_data:
                        pred_data[selected_col] = 1
                
                # Make prediction
                pred_input = pd.Series(pred_data)[model.model.exog_names]
                log_prediction = model.predict(pred_input)[0]
                prediction = np.expm1(log_prediction)  # Transform back from log
                
                # Display results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric(
                        "Predicted Transaction Volume",
                        f"{prediction:,.0f} credits",
                        help="Estimated number of carbon credits in transaction"
                    )
                
                with result_col2:
                    st.metric(
                        "Log-transformed Value",
                        f"{log_prediction:.4f}",
                        help="Raw model output (log-transformed)"
                    )
                
                # Confidence interval (approximate)
                prediction_se = np.sqrt(model.mse_resid)
                ci_lower = np.expm1(log_prediction - 1.96 * prediction_se)
                ci_upper = np.expm1(log_prediction + 1.96 * prediction_se)
                
                st.info(f"📊 **95% Confidence Interval:** {ci_lower:,.0f} - {ci_upper:,.0f} credits")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Variable Importance
        st.markdown("### 📊 Variable Importance")
        
        # Calculate standardized coefficients for importance
        if len(model.params) > 1:
            importance_data = []
            for var in model.params.index:
                if var != 'const':
                    coef = abs(model.params[var])
                    p_val = model.pvalues[var]
                    importance_score = coef * (1 - p_val) if p_val < 1 else 0
                    
                    importance_data.append({
                        'Variable': var.replace('project_category_', ''),
                        'Absolute Coefficient': coef,
                        'P-value': p_val,
                        'Importance Score': importance_score
                    })
            
            importance_df = pd.DataFrame(importance_data).sort_values('Importance Score', ascending=False)
            
            if not importance_df.empty:
                fig_importance = px.bar(
                    importance_df.head(10),
                    x='Importance Score',
                    y='Variable',
                    orientation='h',
                    title="Top 10 Most Important Variables",
                    template='plotly_white'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model Interpretation
        st.markdown("### 💡 Model Interpretation")
        
        interpretation_text = f"""
        **Key Insights from the Regression Model:**
        
        - **Model Fit:** The model explains {model.rsquared:.1%} of the variance in log-transaction volumes
        - **Statistical Significance:** F-statistic p-value = {model.f_pvalue:.2e}
        - **Sample Size:** {model.nobs:,.0f} observations used in modeling
        - **Variables:** {len(model.params)-1} predictor variables included
        
        **Interpretation Notes:**
        - Coefficients represent the change in log-volume for a one-unit change in the predictor
        - Positive coefficients indicate higher transaction volumes
        - Statistical significance (p < 0.05) indicates reliable relationships
        - The model uses log-transformation to handle skewed transaction volume data
        """
        
        st.markdown(interpretation_text)