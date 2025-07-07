"""
Seasonal Activity Predictor for Carbon Credits Market

This module predicts optimal timing for carbon credit transactions based on 
22.5 years of real historical data (458,302 transactions from 2002-2025).

Key Features:
- Monthly activity index (0-100 scale)
- Category-specific seasonal patterns
- Best/worst months identification
- Annual activity calendar
- Strategic timing recommendations

Based on Real Data:
- April: Peak activity month (index=100)
- August: Lowest activity month (index=35)
- Clear seasonal patterns detected in actual market data
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import calendar


class SeasonalActivityPredictor:
    """
    Predicts optimal timing for carbon credit transactions based on historical seasonal patterns.
    
    This calculator uses 22.5 years of real transaction data to identify the best
    and worst months for market activity, helping users optimize transaction timing.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the predictor with historical transaction data.
        
        Args:
            df: DataFrame containing historical carbon credit transactions
        """
        self.df = df
        self.monthly_patterns = None
        self.category_patterns = None
        self._analyze_seasonal_patterns()
    
    
    def _analyze_seasonal_patterns(self) -> None:
        """
        Analyze seasonal patterns from historical data.
        
        This method calculates real monthly activity indices and category-specific
        patterns based on actual transaction volumes and frequencies.
        """
        
        # Overall monthly patterns (all categories combined)
        monthly_volumes = self.df.groupby(self.df['transaction_date'].dt.month)['credits_quantity'].agg([
            'sum', 'count', 'mean'
        ]).round(0)
        
        # Normalize to 0-100 scale (April = 100, others relative)
        max_volume = monthly_volumes['sum'].max()
        monthly_volumes['activity_index'] = (monthly_volumes['sum'] / max_volume * 100).round(1)
        
        # Add month names
        monthly_volumes['month_name'] = [calendar.month_name[i] for i in monthly_volumes.index]
        
        self.monthly_patterns = monthly_volumes
        
        # Category-specific patterns (top categories only for performance)
        top_categories = self.df['project_category'].value_counts().head(10).index
        
        category_seasonal = {}
        for category in top_categories:
            cat_data = self.df[self.df['project_category'] == category]
            cat_monthly = cat_data.groupby(cat_data['transaction_date'].dt.month)['credits_quantity'].agg([
                'sum', 'count'
            ])
            
            # Normalize to 0-100 scale
            if not cat_monthly.empty:
                max_cat_volume = cat_monthly['sum'].max()
                cat_monthly['activity_index'] = (cat_monthly['sum'] / max_cat_volume * 100).round(1)
                cat_monthly['month_name'] = [calendar.month_name[i] for i in cat_monthly.index]
                category_seasonal[category] = cat_monthly
        
        self.category_patterns = category_seasonal
    
    
    def get_monthly_activity_index(self, month: int, category: Optional[str] = None) -> float:
        """
        Get activity index for a specific month.
        
        Args:
            month: Month number (1-12)
            category: Optional specific project category
            
        Returns:
            Activity index (0-100 scale)
        """
        
        if category and category in self.category_patterns:
            if month in self.category_patterns[category].index:
                return self.category_patterns[category].loc[month, 'activity_index']
        
        # Use overall patterns as fallback
        if month in self.monthly_patterns.index:
            return self.monthly_patterns.loc[month, 'activity_index']
        
        return 50.0  # Default neutral activity
    
    
    def get_best_months(self, category: Optional[str] = None, top_n: int = 3) -> List[Dict]:
        """
        Identify the best months for activity.
        
        Args:
            category: Optional specific project category
            top_n: Number of top months to return
            
        Returns:
            List of dictionaries with month info and activity indices
        """
        
        if category and category in self.category_patterns:
            patterns = self.category_patterns[category].copy()
        else:
            patterns = self.monthly_patterns.copy()
        
        # Sort by activity index descending
        top_months = patterns.nlargest(top_n, 'activity_index')
        
        result = []
        for month_num, data in top_months.iterrows():
            result.append({
                'month': int(month_num),
                'month_name': data['month_name'],
                'activity_index': data['activity_index'],
                'recommendation': self._get_timing_recommendation(data['activity_index'])
            })
        
        return result
    
    
    def get_worst_months(self, category: Optional[str] = None, top_n: int = 3) -> List[Dict]:
        """
        Identify the worst months for activity.
        
        Args:
            category: Optional specific project category
            top_n: Number of worst months to return
            
        Returns:
            List of dictionaries with month info and activity indices
        """
        
        if category and category in self.category_patterns:
            patterns = self.category_patterns[category].copy()
        else:
            patterns = self.monthly_patterns.copy()
        
        # Sort by activity index ascending
        worst_months = patterns.nsmallest(top_n, 'activity_index')
        
        result = []
        for month_num, data in worst_months.iterrows():
            result.append({
                'month': int(month_num),
                'month_name': data['month_name'],
                'activity_index': data['activity_index'],
                'recommendation': self._get_timing_recommendation(data['activity_index'])
            })
        
        return result
    
    
    def _get_timing_recommendation(self, activity_index: float) -> str:
        """
        Get timing recommendation based on activity index.
        
        Args:
            activity_index: Activity index (0-100)
            
        Returns:
            Timing recommendation string
        """
        
        if activity_index >= 80:
            return "OPTIMAL"
        elif activity_index >= 60:
            return "GOOD"
        elif activity_index >= 40:
            return "FAIR"
        else:
            return "AVOID"
    
    
    def calculate_timing_score(self, target_month: int, category: Optional[str] = None) -> Dict:
        """
        Calculate comprehensive timing score for a specific month.
        
        Args:
            target_month: Target month (1-12)
            category: Optional project category
            
        Returns:
            Dictionary with timing analysis results
        """
        
        activity_index = self.get_monthly_activity_index(target_month, category)
        month_name = calendar.month_name[target_month]
        
        # Get comparative context
        best_months = self.get_best_months(category, 3)
        worst_months = self.get_worst_months(category, 3)
        
        # Calculate relative position
        all_indices = [self.get_monthly_activity_index(m, category) for m in range(1, 13)]
        percentile_rank = (sum(1 for x in all_indices if x <= activity_index) / len(all_indices)) * 100
        
        return {
            'target_month': target_month,
            'month_name': month_name,
            'activity_index': activity_index,
            'recommendation': self._get_timing_recommendation(activity_index),
            'percentile_rank': round(percentile_rank, 1),
            'best_months': best_months,
            'worst_months': worst_months,
            'category_analyzed': category or "Overall Market",
            'analysis_date': datetime.now().strftime("%Y-%m-%d"),
            'data_source': f"Based on {len(self.df):,} real transactions (2002-2025)"
        }
    
    
    def generate_annual_calendar(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        Generate annual activity calendar with monthly recommendations.
        
        Args:
            category: Optional project category
            
        Returns:
            DataFrame with monthly activity calendar
        """
        
        calendar_data = []
        
        for month in range(1, 13):
            activity_index = self.get_monthly_activity_index(month, category)
            month_name = calendar.month_name[month]
            recommendation = self._get_timing_recommendation(activity_index)
            
            calendar_data.append({
                'month': month,
                'month_name': month_name,
                'activity_index': activity_index,
                'recommendation': recommendation,
                'status': self._get_status_emoji(recommendation)
            })
        
        return pd.DataFrame(calendar_data)
    
    
    def _get_status_emoji(self, recommendation: str) -> str:
        """Get emoji for recommendation status."""
        
        emoji_map = {
            'OPTIMAL': '🟢',
            'GOOD': '🟡', 
            'FAIR': '🟠',
            'AVOID': '🔴'
        }
        return emoji_map.get(recommendation, '⚪')
    
    
    def get_market_insights(self, category: Optional[str] = None) -> Dict:
        """
        Generate strategic market timing insights.
        
        Args:
            category: Optional project category
            
        Returns:
            Dictionary with market insights
        """
        
        patterns = self.category_patterns.get(category, self.monthly_patterns) if category else self.monthly_patterns
        
        # Calculate seasonal metrics
        peak_month = patterns['activity_index'].idxmax()
        low_month = patterns['activity_index'].idxmin()
        peak_value = patterns['activity_index'].max()
        low_value = patterns['activity_index'].min()
        
        seasonal_variation = peak_value - low_value
        avg_activity = patterns['activity_index'].mean()
        
        # Q1, Q2, Q3, Q4 analysis
        quarterly_analysis = {
            'Q1 (Jan-Mar)': patterns.loc[patterns.index.isin([1,2,3]), 'activity_index'].mean(),
            'Q2 (Apr-Jun)': patterns.loc[patterns.index.isin([4,5,6]), 'activity_index'].mean(),
            'Q3 (Jul-Sep)': patterns.loc[patterns.index.isin([7,8,9]), 'activity_index'].mean(),
            'Q4 (Oct-Dec)': patterns.loc[patterns.index.isin([10,11,12]), 'activity_index'].mean()
        }
        
        best_quarter = max(quarterly_analysis, key=quarterly_analysis.get)
        worst_quarter = min(quarterly_analysis, key=quarterly_analysis.get)
        
        return {
            'peak_month': calendar.month_name[peak_month],
            'peak_month_num': int(peak_month),
            'peak_activity': round(peak_value, 1),
            'low_month': calendar.month_name[low_month],
            'low_month_num': int(low_month),
            'low_activity': round(low_value, 1),
            'seasonal_variation': round(seasonal_variation, 1),
            'average_activity': round(avg_activity, 1),
            'best_quarter': best_quarter,
            'worst_quarter': worst_quarter,
            'quarterly_analysis': {k: round(v, 1) for k, v in quarterly_analysis.items()},
            'market_volatility': 'HIGH' if seasonal_variation > 40 else 'MEDIUM' if seasonal_variation > 20 else 'LOW',
            'category_analyzed': category or "Overall Market"
        }


def render_seasonal_predictor_interface(df: pd.DataFrame) -> None:
    """
    Render the main seasonal predictor interface with enhanced UX/UI.
    
    Args:
        df: DataFrame containing historical transaction data
    """
    
    # Section title
    st.markdown("### 📅 Preditor de Atividade Sazonal")
    st.markdown("*Otimize o timing de suas transações usando análise sazonal histórica baseada em dados reais*")
    
    # Initialize predictor
    with st.spinner("🔄 Analisando padrões sazonais históricos..."):
        predictor = SeasonalActivityPredictor(df)
    
    # Configuration section
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown("**⚙️ Configuração da Análise**")
    
    # Category selection
    available_categories = ["Mercado Geral"] + list(predictor.category_patterns.keys())
    selected_category = st.selectbox(
        "Categoria para Análise",
        available_categories,
        help="Escolha categoria específica para insights detalhados ou mercado geral para visão ampla",
        key="seasonal_category"
    )
    
    # Analysis type
    analysis_type = st.radio(
        "**Tipo de Análise Desejada**",
        ["📊 Análise Completa", "🎯 Timing Ótimo Apenas"],
        help="Escolha o nível de detalhamento: completa inclui insights estratégicos",
        horizontal=True,
        key="seasonal_analysis_type"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Execute analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        execute_analysis = st.button(
            "🚀 EXECUTAR ANÁLISE SAZONAL", 
            type="primary",
            key="execute_seasonal_analysis",
            help="Clique para processar a análise sazonal completa"
        )
    
    if execute_analysis:
        # Render analysis based on selection
        with st.spinner("🔄 Processando análise sazonal..."):
            if selected_category == "Mercado Geral":
                render_market_seasonal_analysis(predictor, analysis_type)
            else:
                render_category_seasonal_analysis(predictor, selected_category, analysis_type)


def render_market_seasonal_analysis(predictor: SeasonalActivityPredictor, analysis_type: str) -> None:
    """Render market-wide seasonal analysis."""
    
    if analysis_type == "📊 Análise Completa":
        # Full analysis with all components
        render_timing_analysis(predictor, None)
        st.markdown("---")
        render_annual_calendar(predictor, None)
        st.markdown("---") 
        render_market_insights(predictor, None)
    else:
        # Just timing analysis
        render_timing_analysis(predictor, None)


def render_category_seasonal_analysis(predictor: SeasonalActivityPredictor, category: str, analysis_type: str) -> None:
    """Render category-specific seasonal analysis."""
    
    if analysis_type == "📊 Análise Completa":
        # Full analysis with all components
        render_timing_analysis(predictor, category)
        st.markdown("---")
        render_annual_calendar(predictor, category)
        st.markdown("---")
        render_market_insights(predictor, category)
    else:
        # Just timing analysis
        render_timing_analysis(predictor, category)


def render_timing_analysis(predictor: SeasonalActivityPredictor, category: Optional[str]) -> None:
    """Render timing analysis interface with improved UX."""
    
    # Section title
    st.markdown("### 🎯 Análise de Timing por Mês")
    
    # Month selection section
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown("**📅 Configuração do Timing**")
    
    month_names_pt = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto", 
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_month = st.selectbox(
            "**Selecionar Mês para Análise**",
            range(1, 13),
            format_func=lambda x: f"{month_names_pt[x]} ({x:02d})",
            index=datetime.now().month - 1,
            help="Escolha o mês que deseja analisar para timing ótimo",
            key="timing_month"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        analyze_timing = st.button(
            "🔍 ANALISAR TIMING", 
            type="primary",
            key="analyze_timing_button",
            help="Executar análise detalhada de timing para o mês selecionado"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if analyze_timing:
        # Calculate timing score
        with st.spinner("🔄 Calculando índices de timing..."):
            result = predictor.calculate_timing_score(target_month, category)
        
        # Results container
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Resultados da Análise de Timing")
        
        # Main metrics with enhanced display
        st.markdown("#### 🎯 Métricas de Performance")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "📈 Índice de Atividade",
                f"{result['activity_index']}/100",
                help="Valores maiores indicam melhor timing (0-100)"
            )
        
        with metric_col2:
            st.metric(
                "🎯 Recomendação",
                result['recommendation'],
                help="Classificação estratégica de timing"
            )
        
        with metric_col3:
            st.metric(
                "🏆 Percentil",
                f"{result['percentile_rank']}%",
                help="% de meses com atividade menor"
            )
        
        # Enhanced timing recommendations
        month_name_pt = month_names_pt[target_month]
        st.markdown("#### 💡 Interpretação dos Resultados")
        
        if result['recommendation'] == 'OPTIMAL':
            st.success(f"""
            **🟢 TIMING ÓTIMO IDENTIFICADO**  
            📅 **{month_name_pt}** é um **excelente mês** para transações de carbono.  
            📊 **Vantagem**: Alta atividade de mercado e liquidez superior.  
            ✅ **Ação Recomendada**: Priorize transações neste período.
            """)
        elif result['recommendation'] == 'GOOD':
            st.info(f"""
            **🟡 BOM TIMING IDENTIFICADO**  
            📅 **{month_name_pt}** apresenta **níveis sólidos** de atividade.  
            📊 **Vantagem**: Boa liquidez e condições favoráveis.  
            ⚖️ **Ação Recomendada**: Período adequado para a maioria das transações.
            """)
        elif result['recommendation'] == 'FAIR':
            st.warning(f"""
            **🟠 TIMING RAZOÁVEL**  
            📅 **{month_name_pt}** tem atividade **moderada** no histórico.  
            📊 **Consideração**: Liquidez limitada pode afetar preços.  
            🤔 **Ação Recomendada**: Considere meses alternativos se possível.
            """)
        else:
            st.error(f"""
            **🔴 TIMING INADEQUADO**  
            📅 **{month_name_pt}** historicamente mostra **baixa atividade**.  
            📊 **Risco**: Liquidez reduzida e possível impacto nos preços.  
            ⚠️ **Ação Recomendada**: Evite este período para transações críticas.
            """)
        
        # Comparative context with improved layout
        st.markdown("#### 📈 Contexto Comparativo do Mercado")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("**🏆 Melhores Meses do Ano:**")
            for i, month_info in enumerate(result['best_months'], 1):
                st.markdown(f"**{i}.** {month_info['month_name']} • Índice: **{month_info['activity_index']}**")
        
        with comp_col2:
            st.markdown("**⚠️ Meses de Menor Atividade:**")
            for i, month_info in enumerate(result['worst_months'], 1):
                st.markdown(f"**{i}.** {month_info['month_name']} • Índice: **{month_info['activity_index']}**")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_annual_calendar(predictor: SeasonalActivityPredictor, category: Optional[str]) -> None:
    """Render annual calendar interface with improved UX."""
    
    # Section title
    st.markdown("### 📅 Calendário de Atividade Anual")
    
    # Generate calendar
    with st.spinner("🔄 Gerando calendário sazonal..."):
        calendar_df = predictor.generate_annual_calendar(category)
    
    # Results container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Calendário Sazonal Interativo")
    
    # Display calendar with enhanced styling
    st.dataframe(
        calendar_df.style.format({
            'activity_index': '{:.1f}'
        }).background_gradient(subset=['activity_index'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Enhanced legend with better organization
    st.markdown("#### 📝 Guia de Interpretação")
    
    legend_col1, legend_col2 = st.columns(2)
    
    with legend_col1:
        st.markdown("""
        **🎯 Classificações de Timing:**
        - 🟢 **ÓTIMO** (80-100): Períodos de pico - máxima atividade
        - 🟡 **BOM** (60-79): Períodos favoráveis - alta atividade
        """)
    
    with legend_col2:
        st.markdown("""
        **⚠️ Períodos de Cautela:**
        - 🟠 **RAZOÁVEL** (40-59): Atividade moderada - avaliar contexto
        - 🔴 **EVITAR** (0-39): Baixa atividade - aguardar melhor timing
        """)
    
    # Strategic insights
    st.markdown("#### 💡 Insights Estratégicos")
    st.info("""
    **📈 Como usar este calendário:**
    - **Planejamento Anual**: Use para definir cronogramas de transações
    - **Timing Estratégico**: Concentre atividades nos meses verdes
    - **Gestão de Risco**: Evite períodos vermelhos para transações críticas
    - **Oportunidades**: Monitore transições entre períodos para arbitragem
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_market_insights(predictor: SeasonalActivityPredictor, category: Optional[str]) -> None:
    """Render market insights interface with improved UX."""
    
    # Section title
    st.markdown("### 🧠 Insights Estratégicos de Mercado")
    
    # Get insights
    with st.spinner("🔄 Gerando insights estratégicos..."):
        insights = predictor.get_market_insights(category)
    
    # Results container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Análise Estratégica Sazonal")
    
    # Key metrics with enhanced presentation
    st.markdown("#### 📊 Métricas Sazonais Principais")
    
    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
    
    with insight_col1:
        st.metric(
            "🏆 Mês de Pico", 
            insights['peak_month'], 
            f"Índice: {insights['peak_activity']}",
            help="Mês com maior atividade histórica"
        )
    
    with insight_col2:
        st.metric(
            "📉 Mês Baixo", 
            insights['low_month'], 
            f"Índice: {insights['low_activity']}",
            help="Mês com menor atividade histórica"
        )
    
    with insight_col3:
        st.metric(
            "📊 Variação Sazonal", 
            f"{insights['seasonal_variation']:.1f} pts",
            help="Amplitude da variação entre pico e baixa"
        )
    
    with insight_col4:
        st.metric(
            "⚖️ Volatilidade", 
            insights['market_volatility'],
            help="Classificação da volatilidade sazonal"
        )
    
    # Quarterly analysis with improved layout
    st.markdown("#### 📈 Análise Trimestral Detalhada")
    
    quarterly_col1, quarterly_col2 = st.columns(2)
    
    with quarterly_col1:
        st.markdown("**🌸 Primeiro Semestre:**")
        st.markdown(f"• **Q1 (Jan-Mar)**: {insights['quarterly_analysis']['Q1 (Jan-Mar)']} pts")
        st.markdown(f"• **Q2 (Abr-Jun)**: {insights['quarterly_analysis']['Q2 (Apr-Jun)']} pts")
    
    with quarterly_col2:
        st.markdown("**🍂 Segundo Semestre:**")
        st.markdown(f"• **Q3 (Jul-Set)**: {insights['quarterly_analysis']['Q3 (Jul-Sep)']} pts")
        st.markdown(f"• **Q4 (Out-Dez)**: {insights['quarterly_analysis']['Q4 (Oct-Dec)']} pts")
    
    # Strategic recommendations based on seasonality
    st.markdown("#### 💰 Recomendações Estratégicas")
    
    if insights['market_volatility'] == 'HIGH':
        st.warning(f"""
        **⚠️ ALTA VOLATILIDADE SAZONAL DETECTADA**  
        📊 **Variação de {insights['seasonal_variation']:.1f} pontos** entre pico e baixa.  
        🎯 **Estratégia**: Timing crítico - concentre transações no período **{insights['peak_month']}**.  
        📅 **Evitar**: Período de **{insights['low_month']}** apresenta riscos de liquidez.
        """)
    elif insights['market_volatility'] == 'MEDIUM':
        st.info(f"""
        **📊 VOLATILIDADE SAZONAL MODERADA**  
        📈 **Variação de {insights['seasonal_variation']:.1f} pontos** - padrões consistentes.  
        🎯 **Estratégia**: Prefira **{insights['peak_month']}** mas outros meses são viáveis.  
        ⚖️ **Flexibilidade**: Timing menos crítico, mais opções disponíveis.
        """)
    else:
        st.success(f"""
        **✅ BAIXA VOLATILIDADE SAZONAL**  
        📊 **Variação de apenas {insights['seasonal_variation']:.1f} pontos** - mercado estável.  
        🎯 **Estratégia**: Timing flexível - foque em outros fatores fundamentais.  
        📅 **Vantagem**: Menor dependência de sazonalidade para decisões.
        """)
    
    # Best quarter insights
    st.markdown("#### 🏆 Análise dos Melhores Períodos")
    st.success(f"""
    **🥇 MELHOR TRIMESTRE**: {insights['best_quarter']}  
    **🥉 PIOR TRIMESTRE**: {insights['worst_quarter']}  
    
    **💡 Insight Estratégico**: Concentre atividades no **{insights['best_quarter']}** 
    e planeje com cautela durante **{insights['worst_quarter']}**.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True) 