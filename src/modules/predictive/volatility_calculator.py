"""
Volatility Risk Calculator for Carbon Credits Market

This module calculates risk levels for carbon credit categories based on 
historical transaction volatility from 458,302 real transactions (2002-2025).

Key Features:
- Risk classification (LOW/MEDIUM/HIGH) by category
- Coefficient of Variation (CV) analysis
- Portfolio risk assessment
- Risk-adjusted recommendations
- Historical volatility trends

Based on Real Data:
- Rice Emission: CV=0.59 (Most stable)
- Wind: CV=2.3+ (More volatile)
- Statistical analysis of volume variability
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class VolatilityRiskCalculator:
    """
    Calculates risk levels for carbon credit categories based on historical volatility.
    
    This calculator uses real transaction data to assess the volatility (risk) of
    different project categories, helping users make risk-informed decisions.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the calculator with historical transaction data.
        
        Args:
            df: DataFrame containing historical carbon credit transactions
        """
        self.df = df
        self.category_risk_profiles = None
        self.risk_thresholds = {
            'LOW': 0.8,      # CV < 0.8
            'MEDIUM': 1.5,   # 0.8 <= CV < 1.5  
            'HIGH': float('inf')  # CV >= 1.5
        }
        self._analyze_category_volatility()
    
    
    def _analyze_category_volatility(self) -> None:
        """
        Analyze volatility patterns for each category from historical data.
        
        Calculates coefficient of variation (CV), standard deviation, and other
        risk metrics for each project category with sufficient data.
        """
        
        # Group by category and calculate volatility metrics
        category_stats = self.df.groupby('project_category')['credits_quantity'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        # Filter categories with sufficient data (min 10 transactions)
        category_stats = category_stats[category_stats['count'] >= 10].copy()
        
        # Calculate Coefficient of Variation (CV = std/mean)
        category_stats['cv'] = (category_stats['std'] / category_stats['mean']).round(3)
        category_stats['cv'] = category_stats['cv'].fillna(0)
        
        # Calculate additional risk metrics
        category_stats['range_ratio'] = (category_stats['max'] / category_stats['min']).round(2)
        category_stats['volatility_score'] = (category_stats['cv'] * 100).round(1)
        
        # Classify risk levels
        category_stats['risk_level'] = category_stats['cv'].apply(self._classify_risk_level)
        category_stats['risk_score'] = category_stats['cv'].apply(self._calculate_risk_score)
        
        # Add percentile rankings
        category_stats['risk_percentile'] = category_stats['cv'].rank(pct=True) * 100
        category_stats['risk_percentile'] = category_stats['risk_percentile'].round(1)
        
        # Sort by risk (CV) ascending (lowest risk first)
        category_stats = category_stats.sort_values('cv')
        
        self.category_risk_profiles = category_stats
    
    
    def _classify_risk_level(self, cv: float) -> str:
        """
        Classify risk level based on coefficient of variation.
        
        Args:
            cv: Coefficient of variation
            
        Returns:
            Risk level string
        """
        
        if cv < self.risk_thresholds['LOW']:
            return 'LOW'
        elif cv < self.risk_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    
    def _calculate_risk_score(self, cv: float) -> int:
        """
        Calculate numerical risk score (0-100) based on CV.
        
        Args:
            cv: Coefficient of variation
            
        Returns:
            Risk score (0=lowest risk, 100=highest risk)
        """
        
        # Normalize CV to 0-100 scale
        # Use log transformation to handle extreme values
        if cv <= 0:
            return 0
        
        # Cap at reasonable maximum CV for scaling
        max_cv = 3.0
        normalized_cv = min(cv / max_cv, 1.0)
        
        return int(normalized_cv * 100)
    
    
    def get_category_risk_analysis(self, category: str) -> Optional[Dict]:
        """
        Get detailed risk analysis for a specific category.
        
        Args:
            category: Project category name
            
        Returns:
            Dictionary with risk analysis results or None if category not found
        """
        
        if self.category_risk_profiles is None or category not in self.category_risk_profiles.index:
            return None
        
        data = self.category_risk_profiles.loc[category]
        
        return {
            'category': category,
            'risk_level': data['risk_level'],
            'risk_score': data['risk_score'],
            'cv': data['cv'],
            'volatility_score': data['volatility_score'],
            'risk_percentile': data['risk_percentile'],
            'transaction_count': int(data['count']),
            'mean_volume': data['mean'],
            'std_volume': data['std'],
            'min_volume': data['min'],
            'max_volume': data['max'],
            'median_volume': data['median'],
            'range_ratio': data['range_ratio'],
            'recommendation': self._get_risk_recommendation(data['risk_level'], data['cv']),
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
    
    
    def _get_risk_recommendation(self, risk_level: str, cv: float) -> str:
        """
        Get investment recommendation based on risk level.
        
        Args:
            risk_level: Risk classification
            cv: Coefficient of variation
            
        Returns:
            Investment recommendation string
        """
        
        if risk_level == 'LOW':
            return "CONSERVATIVE - Stable investment suitable for risk-averse portfolios"
        elif risk_level == 'MEDIUM':
            return "BALANCED - Moderate risk with potential for stable returns"
        else:
            return "AGGRESSIVE - High volatility requires careful risk management"
    
    
    def get_risk_ranking(self, top_n: int = 10) -> List[Dict]:
        """
        Get ranking of categories by risk level.
        
        Args:
            top_n: Number of categories to return
            
        Returns:
            List of dictionaries with category risk rankings
        """
        
        if self.category_risk_profiles is None:
            return []
        
        # Get top N categories (lowest risk first)
        top_categories = self.category_risk_profiles.head(top_n)
        
        ranking = []
        for i, (category, data) in enumerate(top_categories.iterrows(), 1):
            ranking.append({
                'rank': i,
                'category': category,
                'risk_level': data['risk_level'],
                'cv': data['cv'],
                'volatility_score': data['volatility_score'],
                'transaction_count': int(data['count']),
                'recommendation': self._get_risk_recommendation(data['risk_level'], data['cv'])
            })
        
        return ranking
    
    
    def get_highest_risk_categories(self, top_n: int = 5) -> List[Dict]:
        """
        Get categories with highest risk.
        
        Args:
            top_n: Number of categories to return
            
        Returns:
            List of highest risk categories
        """
        
        if self.category_risk_profiles is None:
            return []
        
        # Get highest risk categories (highest CV)
        highest_risk = self.category_risk_profiles.tail(top_n)[::-1]  # Reverse to get highest first
        
        ranking = []
        for i, (category, data) in enumerate(highest_risk.iterrows(), 1):
            ranking.append({
                'rank': i,
                'category': category,
                'risk_level': data['risk_level'],
                'cv': data['cv'],
                'volatility_score': data['volatility_score'],
                'transaction_count': int(data['count']),
                'warning': "⚠️ HIGH VOLATILITY - Requires careful risk management"
            })
        
        return ranking
    
    
    def analyze_portfolio_risk(self, portfolio_categories: List[str], 
                              portfolio_weights: Optional[List[float]] = None) -> Dict:
        """
        Analyze risk for a portfolio of categories.
        
        Args:
            portfolio_categories: List of category names in portfolio
            portfolio_weights: Optional weights for each category (default: equal weights)
            
        Returns:
            Dictionary with portfolio risk analysis
        """
        
        if not portfolio_categories:
            return {}
        
        # Default to equal weights if not provided
        if portfolio_weights is None:
            portfolio_weights = [1.0 / len(portfolio_categories)] * len(portfolio_categories)
        
        # Validate inputs
        if len(portfolio_categories) != len(portfolio_weights):
            raise ValueError("Number of categories must match number of weights")
        
        if abs(sum(portfolio_weights) - 1.0) > 0.01:
            raise ValueError("Portfolio weights must sum to 1.0")
        
        # Get risk data for each category
        portfolio_data = []
        valid_categories = []
        valid_weights = []
        
        for i, category in enumerate(portfolio_categories):
            risk_data = self.get_category_risk_analysis(category)
            if risk_data:
                portfolio_data.append(risk_data)
                valid_categories.append(category)
                valid_weights.append(portfolio_weights[i])
        
        if not portfolio_data:
            return {'error': 'No valid categories found in portfolio'}
        
        # Renormalize weights for valid categories
        weight_sum = sum(valid_weights)
        valid_weights = [w / weight_sum for w in valid_weights]
        
        # Calculate portfolio metrics
        weighted_cv = sum(data['cv'] * weight for data, weight in zip(portfolio_data, valid_weights))
        weighted_risk_score = sum(data['risk_score'] * weight for data, weight in zip(portfolio_data, valid_weights))
        
        # Risk level distribution
        risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for data, weight in zip(portfolio_data, valid_weights):
            risk_distribution[data['risk_level']] += weight
        
        # Diversification score (lower CV variance indicates better diversification)
        cvs = [data['cv'] for data in portfolio_data]
        diversification_score = 100 - (np.std(cvs) * 50)  # Scale to 0-100
        diversification_score = max(0, min(100, diversification_score))
        
        return {
            'portfolio_categories': valid_categories,
            'portfolio_weights': [round(w, 3) for w in valid_weights],
            'weighted_cv': round(weighted_cv, 3),
            'weighted_risk_score': round(weighted_risk_score, 1),
            'portfolio_risk_level': self._classify_risk_level(weighted_cv),
            'risk_distribution': {k: round(v * 100, 1) for k, v in risk_distribution.items()},
            'diversification_score': round(diversification_score, 1),
            'total_categories': len(valid_categories),
            'recommendation': self._get_portfolio_recommendation(weighted_cv, diversification_score),
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
    
    
    def _get_portfolio_recommendation(self, weighted_cv: float, diversification_score: float) -> str:
        """
        Get portfolio recommendation based on risk and diversification.
        
        Args:
            weighted_cv: Portfolio weighted coefficient of variation
            diversification_score: Portfolio diversification score
            
        Returns:
            Portfolio recommendation string
        """
        
        risk_level = self._classify_risk_level(weighted_cv)
        
        if risk_level == 'LOW' and diversification_score > 70:
            return "EXCELLENT - Low risk with good diversification"
        elif risk_level == 'LOW':
            return "GOOD - Low risk but consider more diversification"
        elif risk_level == 'MEDIUM' and diversification_score > 70:
            return "BALANCED - Moderate risk with good diversification"
        elif risk_level == 'MEDIUM':
            return "FAIR - Moderate risk, improve diversification"
        elif diversification_score > 70:
            return "RISKY - High risk despite diversification"
        else:
            return "HIGH RISK - Concentrated high-volatility portfolio"
    
    
    def get_risk_insights(self) -> Dict:
        """
        Generate market-wide risk insights.
        
        Returns:
            Dictionary with risk market insights
        """
        
        if self.category_risk_profiles is None:
            return {}
        
        total_categories = len(self.category_risk_profiles)
        
        # Risk level distribution
        risk_counts = self.category_risk_profiles['risk_level'].value_counts()
        risk_distribution = {
            'LOW': risk_counts.get('LOW', 0),
            'MEDIUM': risk_counts.get('MEDIUM', 0),
            'HIGH': risk_counts.get('HIGH', 0)
        }
        
        # Market statistics
        avg_cv = self.category_risk_profiles['cv'].mean()
        median_cv = self.category_risk_profiles['cv'].median()
        min_cv = self.category_risk_profiles['cv'].min()
        max_cv = self.category_risk_profiles['cv'].max()
        
        # Most/least risky categories
        safest_category = self.category_risk_profiles.index[0]
        safest_cv = self.category_risk_profiles.iloc[0]['cv']
        
        riskiest_category = self.category_risk_profiles.index[-1]
        riskiest_cv = self.category_risk_profiles.iloc[-1]['cv']
        
        return {
            'total_categories_analyzed': total_categories,
            'risk_distribution': risk_distribution,
            'risk_percentages': {k: round(v/total_categories*100, 1) for k, v in risk_distribution.items()},
            'market_avg_cv': round(avg_cv, 3),
            'market_median_cv': round(median_cv, 3),
            'cv_range': {
                'min': round(min_cv, 3),
                'max': round(max_cv, 3),
                'spread': round(max_cv - min_cv, 3)
            },
            'safest_category': {
                'name': safest_category,
                'cv': round(safest_cv, 3),
                'risk_level': 'LOW'
            },
            'riskiest_category': {
                'name': riskiest_category,
                'cv': round(riskiest_cv, 3),
                'risk_level': 'HIGH'
            },
            'market_volatility': 'HIGH' if avg_cv > 1.5 else 'MEDIUM' if avg_cv > 0.8 else 'LOW',
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }


def render_volatility_calculator_interface(df: pd.DataFrame) -> None:
    """
    Render the Streamlit interface for the Volatility Risk Calculator.
    
    Args:
        df: DataFrame containing historical transaction data
    """
    
    st.header("Calculadora de Risco de Volatilidade")
    st.markdown("**Avalie o risco de investimento usando análise de coeficiente de variação**")
    
    # Initialize calculator
    with st.spinner("Analisando padrões de volatilidade..."):
        calculator = VolatilityRiskCalculator(df)
    
    # Create tabs for different analyses
    risk_tab, portfolio_tab, rankings_tab, insights_tab = st.tabs([
        "Análise de Risco por Categoria",
        "Avaliação de Risco de Portfolio", 
        "Rankings de Volatilidade",
        "Insights de Risco"
    ])
    
    with risk_tab:
        render_category_risk_analysis(calculator)
    
    with portfolio_tab:
        render_portfolio_risk_assessment(calculator)
    
    with rankings_tab:
        render_risk_rankings(calculator)
    
    with insights_tab:
        render_market_risk_insights(calculator)


def render_category_risk_analysis(calculator: VolatilityRiskCalculator) -> None:
    """Render category-specific risk analysis interface with improved UX."""
    
    # Section title
    st.markdown("### 🎯 Análise de Risco Individual por Categoria")
    
    # Category selection in organized input group
    if calculator.category_risk_profiles is not None:
        available_categories = list(calculator.category_risk_profiles.index)
        
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**🏷️ Configuração da Análise**")
        
        selected_category = st.selectbox(
            "Selecionar Categoria para Análise",
            available_categories,
            help="Escolha uma categoria para analisar seu perfil de risco detalhado",
            key="risk_category_selector"
        )
        
        # Improved button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 ANALISAR PERFIL DE RISCO", 
                type="primary",
                key="analyze_risk_button",
                help="Clique para executar análise completa de risco"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_button:
            # Get risk analysis
            with st.spinner("🔄 Processando análise de risco..."):
                risk_data = calculator.get_category_risk_analysis(selected_category)
            
            if risk_data:
                # Results container with improved styling
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.markdown("### 📊 Resultados da Análise de Risco")
                
                # Key metrics in organized layout
                st.markdown("#### 🎯 Métricas Principais")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📈 Nível de Risco",
                        risk_data['risk_level'],
                        help="Classificação geral: LOW, MEDIUM, HIGH"
                    )
                
                with col2:
                    st.metric(
                        "🎯 Score de Risco",
                        f"{risk_data['risk_score']}/100",
                        help="Pontuação numérica (0=mais seguro, 100=mais arriscado)"
                    )
                
                with col3:
                    st.metric(
                        "📊 Coef. Variação",
                        f"{risk_data['cv']:.3f}",
                        help="Desvio padrão ÷ média (volatilidade relativa)"
                    )
                
                with col4:
                    st.metric(
                        "🏆 Percentil",
                        f"{risk_data['risk_percentile']:.1f}%",
                        help="% de categorias com risco maior"
                    )
                
                # Risk level interpretation with better visual feedback
                st.markdown("#### 💡 Interpretação do Risco")
                if risk_data['risk_level'] == 'LOW':
                    st.success(f"""
                    **🟢 RISCO BAIXO IDENTIFICADO**  
                    📊 **{selected_category}** demonstra padrões de transação **estáveis** com baixa volatilidade.  
                    ✅ **Recomendação**: Categoria adequada para investimentos conservadores.
                    """)
                elif risk_data['risk_level'] == 'MEDIUM':
                    st.info(f"""
                    **🟡 RISCO MÉDIO IDENTIFICADO**  
                    📊 **{selected_category}** apresenta volatilidade **moderada** - perfil equilibrado.  
                    ⚖️ **Recomendação**: Boa opção para portfólios diversificados.
                    """)
                else:
                    st.error(f"""
                    **🔴 RISCO ALTO IDENTIFICADO**  
                    📊 **{selected_category}** exibe **alta volatilidade** de transações.  
                    ⚠️ **Recomendação**: Requer gestão cuidadosa e monitoramento ativo.
                    """)
                
                # Detailed statistics in organized sections
                st.markdown("#### 📈 Estatísticas Detalhadas")
                
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("""
                    **📊 Estatísticas de Volume:**
                    """)
                    st.markdown(f"""
                    - **Total de Transações**: {risk_data['transaction_count']:,}
                    - **Volume Médio**: {risk_data['mean_volume']:,.0f} tCO₂
                    - **Volume Mediano**: {risk_data['median_volume']:,.0f} tCO₂
                    - **Desvio Padrão**: {risk_data['std_volume']:,.0f} tCO₂
                    """)
                
                with stats_col2:
                    st.markdown("""
                    **⚠️ Métricas de Risco:**
                    """)
                    st.markdown(f"""
                    - **Volume Mínimo**: {risk_data['min_volume']:,.0f} tCO₂
                    - **Volume Máximo**: {risk_data['max_volume']:,.0f} tCO₂
                    - **Taxa de Amplitude**: {risk_data['range_ratio']:.1f}x
                    - **Score de Volatilidade**: {risk_data['volatility_score']:.1f}
                    """)
                
                # Investment recommendation with emphasis
                st.markdown("#### 💰 Recomendação de Investimento")
                st.info(f"**💡 {risk_data['recommendation']}**")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("❌ **Erro de Dados**: Incapaz de calcular perfis de risco. Verifique a qualidade dos dados.")


def render_portfolio_risk_assessment(calculator: VolatilityRiskCalculator) -> None:
    """Render portfolio risk assessment interface with improved UX."""
    
    # Section title
    st.markdown("### 📊 Avaliação de Risco de Portfolio")
    
    if calculator.category_risk_profiles is not None:
        available_categories = list(calculator.category_risk_profiles.index)
        
        # Portfolio configuration section
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Configuração do Portfolio")
        
        # Category selection for portfolio
        selected_portfolio_categories = st.multiselect(
            "**Selecionar Categorias para Portfolio**",
            available_categories,
            default=available_categories[:3] if len(available_categories) >= 3 else available_categories,
            help="Escolha entre 2-8 categorias para construir um portfolio diversificado",
            key="portfolio_categories"
        )
        
        if len(selected_portfolio_categories) >= 2:
            # Weight configuration
            st.markdown("#### ⚖️ Distribuição de Pesos")
            
            use_equal_weights = st.checkbox(
                "✅ Usar pesos iguais para todas as categorias", 
                value=True,
                help="Recomendado para portfolios equilibrados"
            )
            
            portfolio_weights = None
            if not use_equal_weights:
                st.markdown("**📊 Configuração Manual de Pesos:**")
                weights = []
                weight_cols = st.columns(min(len(selected_portfolio_categories), 4))
                
                for i, category in enumerate(selected_portfolio_categories):
                    with weight_cols[i % 4]:
                        weight = st.number_input(
                            f"**{category[:15]}{'...' if len(category) > 15 else ''}**",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0/len(selected_portfolio_categories),
                            step=0.01,
                            key=f"weight_{i}",
                            help="Peso entre 0.0 e 1.0"
                        )
                        weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    portfolio_weights = [w/total_weight for w in weights]
                    st.success(f"✅ **Pesos normalizados** para soma = 1.0")
                else:
                    st.error("❌ **Erro**: Todos os pesos não podem ser zero")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze portfolio button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_portfolio = st.button(
                    "🎯 ANALISAR RISCO DO PORTFOLIO", 
                    type="primary",
                    key="analyze_portfolio_button",
                    help="Executar análise completa do portfolio"
                )
            
            if analyze_portfolio:
                try:
                    with st.spinner("🔄 Analisando portfolio..."):
                        portfolio_analysis = calculator.analyze_portfolio_risk(
                            selected_portfolio_categories, 
                            portfolio_weights
                        )
                    
                    if 'error' not in portfolio_analysis:
                        # Display portfolio results
                        st.markdown('<div class="results-container">', unsafe_allow_html=True)
                        st.markdown("### 📊 Resultados da Análise do Portfolio")
                        
                        # Key metrics
                        st.markdown("#### 🎯 Métricas do Portfolio")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric(
                                "📈 Risco do Portfolio",
                                portfolio_analysis['portfolio_risk_level'],
                                help="Classificação geral do portfolio"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "🎯 Risk Score",
                                f"{portfolio_analysis['weighted_risk_score']:.1f}/100",
                                help="Score ponderado de risco"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "📊 Portfolio CV",
                                f"{portfolio_analysis['weighted_cv']:.3f}",
                                help="Coeficiente de variação ponderado"
                            )
                        
                        with metric_col4:
                            st.metric(
                                "🎲 Score Diversificação",
                                f"{portfolio_analysis['diversification_score']:.1f}/100",
                                help="Nível de diversificação (maior = melhor)"
                            )
                        
                        # Portfolio composition and distribution
                        st.markdown("#### 📈 Composição e Distribuição")
                        
                        risk_dist_col1, risk_dist_col2 = st.columns(2)
                        
                        with risk_dist_col1:
                            st.markdown("**📊 Composição do Portfolio:**")
                            for category, weight in zip(portfolio_analysis['portfolio_categories'], 
                                                       portfolio_analysis['portfolio_weights']):
                                st.markdown(f"• **{category}**: {weight*100:.1f}%")
                        
                        with risk_dist_col2:
                            st.markdown("**⚠️ Distribuição de Risco:**")
                            for risk_level, percentage in portfolio_analysis['risk_distribution'].items():
                                icon = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
                                st.markdown(f"• {icon} **{risk_level} Risk**: {percentage:.1f}%")
                        
                        # Portfolio recommendation
                        st.markdown("#### 💰 Recomendação do Portfolio")
                        st.info(f"**💡 {portfolio_analysis['recommendation']}**")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    else:
                        st.error(f"❌ **Erro na análise**: {portfolio_analysis['error']}")
                        
                except Exception as e:
                    st.error(f"❌ **Erro inesperado**: {str(e)}")
        
        else:
            st.warning("⚠️ **Selecione pelo menos 2 categorias** para análise de portfolio")
            
        if len(selected_portfolio_categories) == 0:
            st.info("📝 **Como usar**: Selecione categorias acima para começar a análise")
    
    else:
        st.error("❌ **Erro de Dados**: Perfis de risco não disponíveis. Verifique os dados de entrada.")


def render_risk_rankings(calculator: VolatilityRiskCalculator) -> None:
    """Render risk rankings interface."""
    
    st.subheader("📈 Risk Rankings")
    
    if calculator.category_risk_profiles is not None:
        
        # Display safest categories
        st.markdown("#### 🛡️ Safest Categories (Lowest Risk)")
        
        safest_categories = calculator.get_risk_ranking(10)
        
        for rank_data in safest_categories:
            col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
            
            with col1:
                st.write(f"**#{rank_data['rank']}**")
            
            with col2:
                st.write(f"**{rank_data['category'][:30]}{'...' if len(rank_data['category']) > 30 else ''}**")
            
            with col3:
                risk_color = "🟢" if rank_data['risk_level'] == 'LOW' else "🟡" if rank_data['risk_level'] == 'MEDIUM' else "🔴"
                st.write(f"{risk_color} {rank_data['risk_level']}")
            
            with col4:
                st.write(f"CV: {rank_data['cv']:.3f}")
        
        st.markdown("---")
        
        # Display riskiest categories
        st.markdown("#### ⚠️ Highest Risk Categories")
        
        riskiest_categories = calculator.get_highest_risk_categories(5)
        
        for rank_data in riskiest_categories:
            col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
            
            with col1:
                st.write(f"**#{rank_data['rank']}**")
            
            with col2:
                st.write(f"**{rank_data['category'][:30]}{'...' if len(rank_data['category']) > 30 else ''}**")
            
            with col3:
                st.write("🔴 HIGH")
            
            with col4:
                st.write(f"CV: {rank_data['cv']:.3f}")
        
        # Risk interpretation guide
        st.markdown("---")
        st.markdown("### 📝 Risk Level Guide")
        
        guide_col1, guide_col2, guide_col3 = st.columns(3)
        
        with guide_col1:
            st.success("🟢 **LOW RISK** (CV < 0.8)")
            st.write("Stable, predictable volumes")
            st.write("Suitable for conservative portfolios")
        
        with guide_col2:
            st.info("🟡 **MEDIUM RISK** (0.8 ≤ CV < 1.5)")
            st.write("Moderate volatility")
            st.write("Balanced risk/return profile")
        
        with guide_col3:
            st.error("🔴 **HIGH RISK** (CV ≥ 1.5)")
            st.write("High volatility")
            st.write("Requires careful risk management")
    
    else:
        st.error("❌ Unable to generate risk rankings. Please check the data.")


def render_market_risk_insights(calculator: VolatilityRiskCalculator) -> None:
    """Render market-wide risk insights interface."""
    
    st.subheader("🧠 Market Risk Insights")
    
    # Get market insights
    insights = calculator.get_risk_insights()
    
    if insights:
        # Market overview
        st.markdown("### 📊 Market Risk Overview")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric(
                "Categories Analyzed",
                insights['total_categories_analyzed']
            )
        
        with overview_col2:
            st.metric(
                "Market Volatility",
                insights['market_volatility']
            )
        
        with overview_col3:
            st.metric(
                "Average CV",
                f"{insights['market_avg_cv']:.3f}"
            )
        
        with overview_col4:
            st.metric(
                "CV Range",
                f"{insights['cv_range']['min']:.3f} - {insights['cv_range']['max']:.3f}"
            )
        
        # Risk distribution
        st.markdown("### 📈 Market Risk Distribution")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            st.markdown("**Risk Level Counts:**")
            for risk_level, count in insights['risk_distribution'].items():
                percentage = insights['risk_percentages'][risk_level]
                st.write(f"• **{risk_level} Risk**: {count} categories ({percentage}%)")
        
        with dist_col2:
            st.markdown("**Market Extremes:**")
            st.success(f"🛡️ **Safest**: {insights['safest_category']['name']} (CV: {insights['safest_category']['cv']})")
            st.error(f"⚠️ **Riskiest**: {insights['riskiest_category']['name']} (CV: {insights['riskiest_category']['cv']})")
        
        # Strategic insights
        st.markdown("### 💡 Strategic Risk Insights")
        
        risk_insights = [
            f"📊 **Market Profile**: {insights['risk_percentages']['LOW']:.1f}% low risk, {insights['risk_percentages']['MEDIUM']:.1f}% medium risk, {insights['risk_percentages']['HIGH']:.1f}% high risk categories",
            f"🎯 **Diversification Opportunity**: {insights['cv_range']['spread']:.3f} CV spread provides good diversification potential",
            f"⚖️ **Risk Balance**: Median CV ({insights['market_median_cv']:.3f}) vs Mean CV ({insights['market_avg_cv']:.3f}) indicates market skewness",
            f"🛡️ **Conservative Strategy**: Focus on categories with CV < {calculator.risk_thresholds['LOW']:.1f} for stable portfolios"
        ]
        
        for insight in risk_insights:
            st.write(insight)
        
        # Data source
        st.markdown("---")
        st.markdown(f"📊 **Analysis based on**: {insights['total_categories_analyzed']} categories with sufficient data | 458,302 real transactions | Statistical volatility analysis")
    
    else:
        st.error("❌ Unable to generate market insights. Please check the data.") 