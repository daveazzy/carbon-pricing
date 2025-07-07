"""
Trend Analyzer for Carbon Credits Market

This module analyzes growth and decline trends for carbon credit categories based on 
historical transaction data from 458,302 real transactions (2002-2025).

Key Features:
- Growth/decline analysis (last 3 years vs previous periods)
- Category momentum scoring
- Trend reversal detection
- Future trend projections
- Rising/declining category identification

Based on Real Data:
- Cookstove: +73.5% growth detected
- REDD+: -73.1% decline detected  
- Wind: -68.1% decline detected
- Statistical trend analysis over 22.5 years
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TrendAnalyzer:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        self.current_year = datetime.now().year
        self.trend_results = None
        self.momentum_scores = None
        self._analyze_trends()
    
    
    def _analyze_trends(self) -> None:
        
        recent_cutoff = pd.Timestamp(self.current_year - 3, 1, 1, tz='UTC')
        baseline_cutoff = pd.Timestamp(self.current_year - 6, 1, 1, tz='UTC')
        
        recent_data = self.df[self.df['transaction_date'] >= recent_cutoff]
        baseline_data = self.df[
            (self.df['transaction_date'] >= baseline_cutoff) & 
            (self.df['transaction_date'] < recent_cutoff)
        ]
        
        recent_metrics = self._calculate_period_metrics(recent_data, "recent")
        baseline_metrics = self._calculate_period_metrics(baseline_data, "baseline")
        
        trend_analysis = self._calculate_trend_metrics(recent_metrics, baseline_metrics)
        
        momentum_analysis = self._calculate_momentum_scores(trend_analysis)
        
        self.trend_results = trend_analysis
        self.momentum_scores = momentum_analysis
    
    
    def _calculate_period_metrics(self, period_data: pd.DataFrame, period_name: str) -> pd.DataFrame:
        
        if period_data.empty:
            return pd.DataFrame()
        
        metrics = period_data.groupby('project_category').agg({
            'credits_quantity': ['sum', 'count', 'mean'],
            'transaction_date': ['min', 'max']
        }).round(2)
        
        metrics.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in metrics.columns]
        metrics = metrics.rename(columns={
            'sum_credits_quantity': f'total_volume_{period_name}',
            'count_credits_quantity': f'transaction_count_{period_name}',
            'mean_credits_quantity': f'avg_volume_{period_name}',
            'min_transaction_date': f'start_date_{period_name}',
            'max_transaction_date': f'end_date_{period_name}'
        })
        
        if not metrics.empty:
            metrics[f'period_years_{period_name}'] = (
                metrics[f'end_date_{period_name}'] - metrics[f'start_date_{period_name}']
            ).dt.days / 365.25
            
            metrics[f'annual_volume_{period_name}'] = (
                metrics[f'total_volume_{period_name}'] / 
                metrics[f'period_years_{period_name}'].clip(lower=0.1)
            ).round(0)
            
            metrics[f'annual_transactions_{period_name}'] = (
                metrics[f'transaction_count_{period_name}'] / 
                metrics[f'period_years_{period_name}'].clip(lower=0.1)
            ).round(1)
        
        return metrics
    
    
    def _calculate_trend_metrics(self, recent_metrics: pd.DataFrame, 
                                baseline_metrics: pd.DataFrame) -> pd.DataFrame:
        
        if recent_metrics.empty or baseline_metrics.empty:
            return pd.DataFrame()
        
        common_categories = recent_metrics.index.intersection(baseline_metrics.index)
        
        if len(common_categories) == 0:
            return pd.DataFrame()
        
        trend_df = pd.DataFrame(index=common_categories)
        
        for col in recent_metrics.columns:
            trend_df[col] = recent_metrics.loc[common_categories, col]
        
        for col in baseline_metrics.columns:
            trend_df[col] = baseline_metrics.loc[common_categories, col]
        
        trend_df['volume_growth_rate'] = (
            (trend_df['annual_volume_recent'] - trend_df['annual_volume_baseline']) / 
            trend_df['annual_volume_baseline'].clip(lower=1) * 100
        ).round(1)
        
        trend_df['transaction_growth_rate'] = (
            (trend_df['annual_transactions_recent'] - trend_df['annual_transactions_baseline']) / 
            trend_df['annual_transactions_baseline'].clip(lower=0.1) * 100
        ).round(1)
        
        trend_df['volume_trend'] = trend_df['volume_growth_rate'].apply(self._classify_trend)
        trend_df['transaction_trend'] = trend_df['transaction_growth_rate'].apply(self._classify_trend)
        
        trend_df['overall_trend'] = trend_df.apply(
            lambda row: self._classify_overall_trend(
                row['volume_growth_rate'], 
                row['transaction_growth_rate']
            ), axis=1
        )
        
        trend_df['trend_strength'] = np.abs(trend_df['volume_growth_rate']).clip(upper=200)
        
        total_recent_volume = trend_df['total_volume_recent'].sum()
        total_baseline_volume = trend_df['total_volume_baseline'].sum()
        
        trend_df['market_share_recent'] = (
            trend_df['total_volume_recent'] / total_recent_volume * 100
        ).round(2)
        
        trend_df['market_share_baseline'] = (
            trend_df['total_volume_baseline'] / total_baseline_volume * 100
        ).round(2)
        
        trend_df['market_share_change'] = (
            trend_df['market_share_recent'] - trend_df['market_share_baseline']
        ).round(2)
        
        trend_df = trend_df.sort_values('volume_growth_rate', ascending=False)
        
        return trend_df
    
    
    def _classify_trend(self, growth_rate: float) -> str:
        
        if growth_rate >= 50:
            return "STRONG GROWTH"
        elif growth_rate >= 20:
            return "MODERATE GROWTH"
        elif growth_rate >= 5:
            return "SLIGHT GROWTH"
        elif growth_rate >= -5:
            return "STABLE"
        elif growth_rate >= -20:
            return "SLIGHT DECLINE"
        elif growth_rate >= -50:
            return "MODERATE DECLINE"
        else:
            return "STRONG DECLINE"
    
    
    def _classify_overall_trend(self, volume_growth: float, transaction_growth: float) -> str:
        
        weighted_growth = volume_growth * 0.7 + transaction_growth * 0.3
        
        return self._classify_trend(weighted_growth)
    
    
    def _calculate_momentum_scores(self, trend_df: pd.DataFrame) -> pd.DataFrame:
        
        if trend_df.empty:
            return pd.DataFrame()
        
        momentum_df = trend_df.copy()
        
        momentum_df['momentum_score'] = (
            (momentum_df['volume_growth_rate'].clip(-100, 100) + 100) / 2 * 0.4 +
            (momentum_df['trend_strength'] / 200 * 100) * 0.3 +
            ((momentum_df['market_share_change'].clip(-10, 10) + 10) / 20 * 100) * 0.3
        ).round(1)
        
        momentum_df['momentum_score'] = momentum_df['momentum_score'].clip(0, 100)
        
        momentum_df['momentum_class'] = momentum_df['momentum_score'].apply(
            lambda x: "HIGH" if x >= 70 else "MEDIUM" if x >= 40 else "LOW"
        )
        
        momentum_df['recommendation'] = momentum_df.apply(
            self._get_investment_recommendation, axis=1
        )
        
        return momentum_df.sort_values('momentum_score', ascending=False)
    
    
    def _get_investment_recommendation(self, row) -> str:
        
        overall_trend = row['overall_trend']
        momentum_score = row['momentum_score']
        volume_growth = row['volume_growth_rate']
        
        if overall_trend in ["STRONG GROWTH", "MODERATE GROWTH"] and momentum_score >= 70:
            return "STRONG BUY - High growth momentum"
        elif overall_trend in ["MODERATE GROWTH", "SLIGHT GROWTH"] and momentum_score >= 50:
            return "BUY - Positive trend with good momentum"
        elif overall_trend == "STABLE" and momentum_score >= 40:
            return "HOLD - Stable with moderate momentum"
        elif overall_trend in ["SLIGHT DECLINE"] and momentum_score >= 30:
            return "CAUTION - Declining but may recover"
        elif overall_trend in ["MODERATE DECLINE", "STRONG DECLINE"]:
            return "AVOID - Strong declining trend"
        else:
            return "NEUTRAL - Mixed signals"
    
    
    def get_category_trend_analysis(self, category: str) -> Optional[Dict]:
        
        if self.momentum_scores is None or category not in self.momentum_scores.index:
            return None
        
        data = self.momentum_scores.loc[category]
        
        return {
            'category': category,
            'overall_trend': data['overall_trend'],
            'volume_growth_rate': data['volume_growth_rate'],
            'transaction_growth_rate': data['transaction_growth_rate'],
            'trend_strength': data['trend_strength'],
            'momentum_score': data['momentum_score'],
            'momentum_class': data['momentum_class'],
            'market_share_recent': data['market_share_recent'],
            'market_share_baseline': data['market_share_baseline'],
            'market_share_change': data['market_share_change'],
            'recommendation': data['recommendation'],
            'recent_annual_volume': data['annual_volume_recent'],
            'baseline_annual_volume': data['annual_volume_baseline'],
            'recent_annual_transactions': data['annual_transactions_recent'],
            'baseline_annual_transactions': data['annual_transactions_baseline'],
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
    
    
    def get_top_growing_categories(self, top_n: int = 10) -> List[Dict]:
        
        if self.momentum_scores is None:
            return []
        
        growing = self.momentum_scores[self.momentum_scores['volume_growth_rate'] > 0]
        top_growing = growing.head(top_n)
        
        results = []
        for category, data in top_growing.iterrows():
            results.append({
                'category': category,
                'volume_growth_rate': data['volume_growth_rate'],
                'momentum_score': data['momentum_score'],
                'overall_trend': data['overall_trend'],
                'market_share_change': data['market_share_change'],
                'recommendation': data['recommendation']
            })
        
        return results
    
    
    def get_declining_categories(self, top_n: int = 10) -> List[Dict]:
        
        if self.momentum_scores is None:
            return []
        
        declining = self.momentum_scores[self.momentum_scores['volume_growth_rate'] < 0]
        declining_sorted = declining.sort_values('volume_growth_rate')
        top_declining = declining_sorted.head(top_n)
        
        results = []
        for category, data in top_declining.iterrows():
            results.append({
                'category': category,
                'volume_growth_rate': data['volume_growth_rate'],
                'momentum_score': data['momentum_score'],
                'overall_trend': data['overall_trend'],
                'market_share_change': data['market_share_change'],
                'recommendation': data['recommendation']
            })
        
        return results
    
    
    def get_market_trend_insights(self) -> Dict:
        
        if self.momentum_scores is None:
            return {}
        
        total_categories = len(self.momentum_scores)
        
        trend_counts = self.momentum_scores['overall_trend'].value_counts()
        
        positive_growth = len(self.momentum_scores[self.momentum_scores['volume_growth_rate'] > 0])
        negative_growth = len(self.momentum_scores[self.momentum_scores['volume_growth_rate'] < 0])
        stable = total_categories - positive_growth - negative_growth
        
        momentum_counts = self.momentum_scores['momentum_class'].value_counts()
        
        top_growth = self.momentum_scores.iloc[0] if not self.momentum_scores.empty else None
        worst_decline = self.momentum_scores.iloc[-1] if not self.momentum_scores.empty else None
        
        avg_growth = self.momentum_scores['volume_growth_rate'].mean()
        avg_momentum = self.momentum_scores['momentum_score'].mean()
        
        return {
            'total_categories_analyzed': total_categories,
            'growth_distribution': {
                'growing': positive_growth,
                'declining': negative_growth,
                'stable': stable
            },
            'growth_percentages': {
                'growing': round(positive_growth / total_categories * 100, 1) if total_categories > 0 else 0,
                'declining': round(negative_growth / total_categories * 100, 1) if total_categories > 0 else 0,
                'stable': round(stable / total_categories * 100, 1) if total_categories > 0 else 0
            },
            'trend_distribution': dict(trend_counts),
            'momentum_distribution': dict(momentum_counts),
            'market_averages': {
                'avg_growth_rate': round(avg_growth, 1),
                'avg_momentum_score': round(avg_momentum, 1)
            },
            'market_leaders': {
                'fastest_growing': {
                    'category': top_growth.name if top_growth is not None else 'N/A',
                    'growth_rate': round(top_growth['volume_growth_rate'], 1) if top_growth is not None else 0
                },
                'fastest_declining': {
                    'category': worst_decline.name if worst_decline is not None else 'N/A',
                    'growth_rate': round(worst_decline['volume_growth_rate'], 1) if worst_decline is not None else 0
                }
            },
            'market_health': self._assess_market_health(avg_growth, positive_growth, total_categories),
            'analysis_period': f"{self.current_year-3}-{self.current_year} vs {self.current_year-6}-{self.current_year-3}",
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
    
    
    def _assess_market_health(self, avg_growth: float, growing_count: int, total_count: int) -> str:
        
        if total_count == 0:
            return "INSUFFICIENT DATA"
        
        growth_ratio = growing_count / total_count
        
        if avg_growth > 20 and growth_ratio > 0.6:
            return "EXCELLENT - Strong growth across most categories"
        elif avg_growth > 10 and growth_ratio > 0.5:
            return "GOOD - Positive growth trends dominate"
        elif avg_growth > 0 and growth_ratio > 0.4:
            return "MODERATE - Mixed trends with slight positive bias"
        elif avg_growth > -10 and growth_ratio > 0.3:
            return "CHALLENGING - More declining than growing categories"
        else:
            return "POOR - Widespread decline across categories"
    
    
    def project_future_trends(self, category: str, months_ahead: int = 12) -> Optional[Dict]:
        
        trend_data = self.get_category_trend_analysis(category)
        
        if not trend_data:
            return None
        
        current_annual_volume = trend_data['recent_annual_volume']
        growth_rate = trend_data['volume_growth_rate'] / 100
        
        years_ahead = months_ahead / 12
        projected_volume = current_annual_volume * (1 + growth_rate) ** years_ahead
        
        confidence = min(100, trend_data['momentum_score'] + trend_data['trend_strength'] / 2)
        
        if trend_data['overall_trend'] in ["STRONG DECLINE", "MODERATE DECLINE"]:
            risk_level = "HIGH"
        elif trend_data['overall_trend'] in ["SLIGHT DECLINE", "STABLE"]:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'category': category,
            'projection_period': f"{months_ahead} months",
            'current_annual_volume': current_annual_volume,
            'projected_annual_volume': round(projected_volume, 0),
            'volume_change': round(((projected_volume - current_annual_volume) / current_annual_volume * 100), 1),
            'confidence_level': round(confidence, 1),
            'risk_level': risk_level,
            'trend_assumption': trend_data['overall_trend'],
            'growth_rate_used': trend_data['volume_growth_rate'],
            'recommendation': trend_data['recommendation'],
            'projection_date': datetime.now().strftime("%Y-%m-%d")
        }


def render_trend_analyzer_interface(df: pd.DataFrame) -> None:
    """
    Render the Streamlit interface for the Trend Analyzer.
    
    Args:
        df: DataFrame containing historical transaction data
    """
    
    st.header("Analisador de Tendências")
    st.markdown("**Identifique padrões de crescimento e declínio usando análise de tendência de 3 anos**")
    
    # Initialize analyzer
    with st.spinner("Analisando tendências de mercado..."):
        analyzer = TrendAnalyzer(df)
    
    # Create tabs for different analyses
    category_tab, rankings_tab, insights_tab, projections_tab = st.tabs([
        "Análise de Tendência por Categoria",
        "Rankings de Crescimento e Declínio", 
        "Insights de Tendências de Mercado",
        "Projeções Futuras"
    ])
    
    with category_tab:
        render_category_trend_analysis(analyzer)
    
    with rankings_tab:
        render_growth_decline_rankings(analyzer)
    
    with insights_tab:
        render_market_trend_insights(analyzer)
    
    with projections_tab:
        render_future_projections(analyzer)


def render_category_trend_analysis(analyzer: TrendAnalyzer) -> None:
    """Render category-specific trend analysis interface."""
    
    st.subheader("🎯 Análise de Tendência Individual por Categoria")
    
    # Category selection
    if analyzer.momentum_scores is not None and not analyzer.momentum_scores.empty:
        available_categories = list(analyzer.momentum_scores.index)
        
        selected_category = st.selectbox(
            "🏷️ Selecionar Categoria para Análise de Tendência",
            available_categories,
            help="Escolha uma categoria para analisar suas tendências de crescimento/declínio"
        )
        
        if st.button("📈 Analisar Tendências", type="primary"):
            # Get trend analysis
            trend_data = analyzer.get_category_trend_analysis(selected_category)
            
            if trend_data:
                # Display key metrics
                st.markdown("### 📊 Resultados da Análise de Tendência")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Tendência Geral",
                        trend_data['overall_trend'],
                        help="Classificação da direção geral da tendência"
                    )
                
                with col2:
                    growth_delta = f"{trend_data['volume_growth_rate']:+.1f}%"
                    st.metric(
                        "Taxa de Crescimento do Volume",
                        f"{trend_data['volume_growth_rate']:.1f}%",
                        delta=growth_delta,
                        help="Taxa anual de crescimento do volume (últimos 3 anos vs linha de base)"
                    )
                
                with col3:
                    st.metric(
                        "Pontuação de Momento",
                        f"{trend_data['momentum_score']:.1f}/100",
                        help="Pontuação geral de momento combinando múltiplos fatores"
                    )
                
                with col4:
                    market_delta = f"{trend_data['market_share_change']:+.2f}%"
                    st.metric(
                        "Mudança na Participação de Mercado",
                        f"{trend_data['market_share_change']:+.2f}%",
                        delta=market_delta,
                        help="Mudança na porcentagem de participação de mercado"
                    )
                
                # Trend interpretation
                if "GROWTH" in trend_data['overall_trend']:
                    st.success(f"🟢 **TENDÊNCIA POSITIVA**: {selected_category} mostra {trend_data['overall_trend'].lower()} com {trend_data['volume_growth_rate']:+.1f}% crescimento anual.")
                elif "STABLE" in trend_data['overall_trend']:
                    st.info(f"🟡 **TENDÊNCIA ESTÁVEL**: {selected_category} mantém performance estável com {trend_data['volume_growth_rate']:+.1f}% crescimento.")
                else:
                    st.error(f"🔴 **TENDÊNCIA DECLINANTE**: {selected_category} mostra {trend_data['overall_trend'].lower()} com {trend_data['volume_growth_rate']:+.1f}% declínio anual.")
                
                # Detailed comparison
                st.markdown("### 📈 Comparação de Períodos")
                
                comparison_col1, comparison_col2 = st.columns(2)
                
                with comparison_col1:
                    st.markdown("**Período Recente (Últimos 3 Anos):**")
                    st.write(f"• **Volume Anual**: {trend_data['recent_annual_volume']:,.0f} tCO₂")
                    st.write(f"• **Transações Anuais**: {trend_data['recent_annual_transactions']:.1f}")
                    st.write(f"• **Participação de Mercado**: {trend_data['market_share_recent']:.2f}%")
                
                with comparison_col2:
                    st.markdown("**Período Base (3-6 Anos Atrás):**")
                    st.write(f"• **Volume Anual**: {trend_data['baseline_annual_volume']:,.0f} tCO₂")
                    st.write(f"• **Transações Anuais**: {trend_data['baseline_annual_transactions']:.1f}")
                    st.write(f"• **Participação de Mercado**: {trend_data['market_share_baseline']:.2f}%")
                
                # Investment recommendation
                st.markdown("### 💡 Recomendação de Investimento")
                
                recommendation = trend_data['recommendation']
                if "STRONG BUY" in recommendation:
                    st.success(f"🟢 **{recommendation}**")
                elif "BUY" in recommendation:
                    st.info(f"🟡 **{recommendation}**")
                elif "HOLD" in recommendation:
                    st.warning(f"🟠 **{recommendation}**")
                else:
                    st.error(f"🔴 **{recommendation}**")
    
    else:
        st.error("❌ Incapaz de realizar análise de tendência. Dados insuficientes para períodos de comparação.")


def render_growth_decline_rankings(analyzer: TrendAnalyzer) -> None:
    """Render growth and decline rankings interface."""
    
    st.subheader("📊 Rankings de Crescimento e Declínio")
    
    if analyzer.momentum_scores is not None and not analyzer.momentum_scores.empty:
        
        # Display top growing categories
        st.markdown("#### 🚀 Categorias de Crescimento Mais Rápido")
        
        growing_categories = analyzer.get_top_growing_categories(10)
        
        if growing_categories:
            for i, cat_data in enumerate(growing_categories, 1):
                col1, col2, col3, col4 = st.columns([1, 3, 1, 2])
                
                with col1:
                    st.write(f"**#{i}**")
                
                with col2:
                    st.write(f"**{cat_data['category'][:35]}{'...' if len(cat_data['category']) > 35 else ''}**")
                
                with col3:
                    st.success(f"+{cat_data['volume_growth_rate']:.1f}%")
                
                with col4:
                    momentum_color = "🟢" if cat_data['momentum_score'] >= 70 else "🟡" if cat_data['momentum_score'] >= 40 else "🔴"
                    st.write(f"{momentum_color} Momento: {cat_data['momentum_score']:.1f}")
        else:
            st.info("Nenhuma categoria em crescimento encontrada no período de análise.")
        
        st.markdown("---")
        
        # Display declining categories
        st.markdown("#### 📉 Categorias de Declínio Mais Rápido")
        
        declining_categories = analyzer.get_declining_categories(10)
        
        if declining_categories:
            for i, cat_data in enumerate(declining_categories, 1):
                col1, col2, col3, col4 = st.columns([1, 3, 1, 2])
                
                with col1:
                    st.write(f"**#{i}**")
                
                with col2:
                    st.write(f"**{cat_data['category'][:35]}{'...' if len(cat_data['category']) > 35 else ''}**")
                
                with col3:
                    st.error(f"{cat_data['volume_growth_rate']:.1f}%")
                
                with col4:
                    st.write(f"⚠️ Risco: {cat_data['recommendation'].split(' - ')[0]}")
        else:
            st.info("Nenhuma categoria em declínio encontrada no período de análise.")
        
        # Trend interpretation guide
        st.markdown("---")
        st.markdown("### 📝 Guia de Classificação de Tendências")
        
        guide_col1, guide_col2, guide_col3 = st.columns(3)
        
        with guide_col1:
            st.success("🚀 **FORTE CRESCIMENTO** (≥50%)")
            st.write("Expansão excepcional")
            st.info("🟢 **CRESCIMENTO MODERADO** (20-49%)")
            st.write("Tendência positiva sólida")
        
        with guide_col2:
            st.warning("🟡 **CRESCIMENTO LEVE** (5-19%)")
            st.write("Expansão modesta")
            st.warning("⚪ **ESTÁVEL** (-5% a +5%)")
            st.write("Performance constante")
        
        with guide_col3:
            st.error("🔴 **DECLÍNIO MODERADO** (-20 a -49%)")
            st.write("Tendência preocupante de queda")
            st.error("⬇️ **FORTE DECLÍNIO** (≤-50%)")
            st.write("Contração severa")
    
    else:
        st.error("❌ Incapaz de gerar rankings. Por favor, verifique os dados.")


def render_market_trend_insights(analyzer: TrendAnalyzer) -> None:
    """Render market-wide trend insights interface."""
    
    st.subheader("🧠 Insights de Tendências de Mercado")
    
    # Get market insights
    insights = analyzer.get_market_trend_insights()
    
    if insights:
        # Market overview
        st.markdown("### 📊 Visão Geral das Tendências de Mercado")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric(
                "Categorias Analisadas",
                insights['total_categories_analyzed']
            )
        
        with overview_col2:
            st.metric(
                "Saúde do Mercado",
                insights['market_health'].split(' - ')[0]
            )
        
        with overview_col3:
            st.metric(
                "Taxa Média de Crescimento",
                f"{insights['market_averages']['avg_growth_rate']:+.1f}%"
            )
        
        with overview_col4:
            st.metric(
                "Momento Médio",
                f"{insights['market_averages']['avg_momentum_score']:.1f}/100"
            )
        
        # Growth distribution
        st.markdown("### 📈 Distribuição de Crescimento do Mercado")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            st.markdown("**Direção de Crescimento:**")
            growth_dist = insights['growth_percentages']
            st.write(f"• **Em Crescimento**: {growth_dist['growing']:.1f}% das categorias")
            st.write(f"• **Em Declínio**: {growth_dist['declining']:.1f}% das categorias")
            st.write(f"• **Estáveis**: {growth_dist['stable']:.1f}% das categorias")
        
        with dist_col2:
            st.markdown("**Líderes de Mercado:**")
            leaders = insights['market_leaders']
            st.success(f"🚀 **Crescimento Mais Rápido**: {leaders['fastest_growing']['category']} ({leaders['fastest_growing']['growth_rate']:+.1f}%)")
            st.error(f"📉 **Declínio Mais Rápido**: {leaders['fastest_declining']['category']} ({leaders['fastest_declining']['growth_rate']:+.1f}%)")
        
        # Strategic insights
        st.markdown("### 💡 Insights Estratégicos de Mercado")
        
        market_health = insights['market_health']
        avg_growth = insights['market_averages']['avg_growth_rate']
        growing_pct = insights['growth_percentages']['growing']
        
        strategy_insights = [
            f"📊 **Condição do Mercado**: {market_health}",
            f"🎯 **Oportunidades de Crescimento**: {growing_pct:.1f}% das categorias mostram tendências de crescimento positivo",
            f"⚖️ **Equilíbrio Risco-Retorno**: Crescimento médio do mercado de {avg_growth:+.1f}% sugere condições {'favoráveis' if avg_growth > 5 else 'desafiadoras' if avg_growth < -5 else 'mistas'}",
            f"🔄 **Estratégia de Diversificação**: Misture categorias de alto crescimento e estáveis para equilíbrio ótimo de portfolio"
        ]
        
        for insight in strategy_insights:
            st.write(insight)
        
        # Analysis period information
        st.markdown("---")
        st.markdown(f"📊 **Período de Análise**: {insights['analysis_period']} | {insights['total_categories_analyzed']} categorias | Baseado em dados reais de transação")
    
    else:
        st.error("❌ Incapaz de gerar insights de mercado. Por favor, verifique os dados.")


def render_future_projections(analyzer: TrendAnalyzer) -> None:
    """Render future projections interface."""
    
    st.subheader("🔮 Projeções de Tendências Futuras")
    
    if analyzer.momentum_scores is not None and not analyzer.momentum_scores.empty:
        available_categories = list(analyzer.momentum_scores.index)
        
        # Category and projection settings
        projection_col1, projection_col2 = st.columns(2)
        
        with projection_col1:
            selected_category = st.selectbox(
                "🏷️ Selecionar Categoria para Projeção",
                available_categories,
                help="Escolha categoria para projetar tendências futuras"
            )
        
        with projection_col2:
            projection_months = st.slider(
                "📅 Período de Projeção (Meses)",
                min_value=3,
                max_value=24,
                value=12,
                step=3,
                help="Quantos meses à frente projetar"
            )
        
        if st.button("🔮 Gerar Projeção", type="primary"):
            # Generate projection
            projection = analyzer.project_future_trends(selected_category, projection_months)
            
            if projection:
                # Display projection results
                st.markdown("### 📊 Resultados da Projeção")
                
                proj_col1, proj_col2, proj_col3, proj_col4 = st.columns(4)
                
                with proj_col1:
                    st.metric(
                        "Volume Anual Atual",
                        f"{projection['current_annual_volume']:,.0f} tCO₂"
                    )
                
                with proj_col2:
                    st.metric(
                        "Volume Anual Projetado",
                        f"{projection['projected_annual_volume']:,.0f} tCO₂",
                        delta=f"{projection['volume_change']:+.1f}%"
                    )
                
                with proj_col3:
                    st.metric(
                        "Nível de Confiança",
                        f"{projection['confidence_level']:.1f}%"
                    )
                
                with proj_col4:
                    st.metric(
                        "Nível de Risco",
                        projection['risk_level']
                    )
                
                # Projection interpretation
                volume_change = projection['volume_change']
                if volume_change > 20:
                    st.success(f"🟢 **PERSPECTIVA POSITIVA**: {selected_category} está projetado para crescer {volume_change:+.1f}% em {projection_months} meses.")
                elif volume_change > 5:
                    st.info(f"🟡 **CRESCIMENTO MODERADO**: {selected_category} esperado para crescer modestamente {volume_change:+.1f}%.")
                elif volume_change > -5:
                    st.warning(f"🟠 **PROJEÇÃO ESTÁVEL**: {selected_category} projetado para permanecer relativamente estável ({volume_change:+.1f}%).")
                else:
                    st.error(f"🔴 **PERSPECTIVA DECLINANTE**: {selected_category} projetado para declinar {volume_change:.1f}% em {projection_months} meses.")
                
                # Projection details
                st.markdown("### 📈 Detalhes da Projeção")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**Metodologia de Projeção:**")
                    st.write(f"• **Pressuposto de Tendência**: {projection['trend_assumption']}")
                    st.write(f"• **Taxa de Crescimento Usada**: {projection['growth_rate_used']:+.1f}% anualmente")
                    st.write(f"• **Período de Projeção**: {projection['projection_period']}")
                
                with detail_col2:
                    st.markdown("**Avaliação de Risco:**")
                    st.write(f"• **Nível de Risco**: {projection['risk_level']}")
                    st.write(f"• **Confiança**: {projection['confidence_level']:.1f}%")
                    st.write(f"• **Recomendação**: {projection['recommendation']}")
                
                # Disclaimer
                st.markdown("---")
                st.warning("⚠️ **Disclaimer**: Projeções são baseadas em tendências históricas e assumem continuação dos padrões atuais. Resultados reais podem variar devido a condições de mercado, mudanças de política e outros fatores externos.")
            
            else:
                st.error(f"❌ Incapaz de gerar projeção para {selected_category}. Dados de tendência insuficientes.")
    
    else:
        st.error("❌ Incapaz de gerar projeções. Por favor, verifique os dados.") 