"""
Geographic Expansion Predictor for Carbon Credits Market

This module analyzes geographic expansion opportunities based on 109 countries
of historical transaction data from 458,302 real transactions (2002-2025).

Key Features:
- Emerging market identification (growth rates by country)
- Country-category correlation analysis
- Market entry timing recommendations
- Geographic diversification scoring
- Risk assessment by region/country

Based on Real Data:
- 109 countries analyzed
- Regional growth patterns detected
- Cross-border opportunity mapping
- Market maturity assessment by geography
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GeographicPredictor:
    """
    Analyzes geographic expansion opportunities for carbon credit markets.
    
    This predictor identifies emerging markets, assesses country-category
    correlations, and provides expansion timing recommendations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the predictor with historical transaction data.
        
        Args:
            df: DataFrame containing historical carbon credit transactions
        """
        self.df = df.copy()
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        self.current_year = datetime.now().year
        
        # Geographic analysis results
        self.country_analysis = None
        self.regional_analysis = None
        self.category_country_matrix = None
        self.expansion_opportunities = None
        
        self._prepare_geographic_data()
        self._analyze_geographic_patterns()
    
    
    def _prepare_geographic_data(self) -> None:
        """
        Prepare geographic data for analysis.
        
        Creates country-level aggregations and regional groupings.
        """
        
        # Country-level aggregations
        self.country_data = self.df.groupby('project_country').agg({
            'credits_quantity': ['sum', 'count', 'mean'],
            'project_category': 'nunique',
            'transaction_date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        self.country_data.columns = [
            'total_volume', 'transaction_count', 'avg_volume', 
            'categories_count', 'first_transaction', 'last_transaction'
        ]
        
        # Calculate market metrics
        self.country_data['market_age_years'] = (
            self.country_data['last_transaction'] - self.country_data['first_transaction']
        ).dt.days / 365.25
        
        self.country_data['annual_volume'] = (
            self.country_data['total_volume'] / 
            self.country_data['market_age_years'].clip(lower=0.1)
        ).round(0)
        
        self.country_data['market_share'] = (
            self.country_data['total_volume'] / self.country_data['total_volume'].sum() * 100
        ).round(3)
        
        # Regional groupings (simplified)
        self.country_data['region'] = self.country_data.index.map(self._assign_region)
        
        # Category-country matrix
        self.category_country_matrix = pd.crosstab(
            self.df['project_category'], 
            self.df['project_country'], 
            values=self.df['credits_quantity'], 
            aggfunc='sum'
        ).fillna(0)
    
    
    def _assign_region(self, country: str) -> str:
        """
        Assign region to country (simplified mapping).
        
        Args:
            country: Country name
            
        Returns:
            Region name
        """
        
        # Simplified regional mapping
        regions = {
            'AFRICA': ['Nigeria', 'South Africa', 'Kenya', 'Ghana', 'Egypt', 'Morocco', 'Ethiopia', 'Tanzania', 'Uganda', 'Zimbabwe'],
            'ASIA': ['China', 'India', 'Indonesia', 'Thailand', 'Malaysia', 'Philippines', 'Vietnam', 'Bangladesh', 'Pakistan', 'Sri Lanka'],
            'EUROPE': ['Germany', 'United Kingdom', 'France', 'Spain', 'Italy', 'Netherlands', 'Poland', 'Sweden', 'Norway', 'Denmark'],
            'NORTH_AMERICA': ['United States', 'Canada', 'Mexico'],
            'SOUTH_AMERICA': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Ecuador', 'Venezuela', 'Uruguay', 'Bolivia', 'Paraguay'],
            'OCEANIA': ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji']
        }
        
        for region, countries in regions.items():
            if any(c.lower() in country.lower() for c in countries):
                return region
        
        return 'OTHER'
    
    
    def _analyze_geographic_patterns(self) -> None:
        """
        Analyze geographic patterns and growth opportunities.
        
        Performs comprehensive analysis of country and regional trends.
        """
        
        # Time-based analysis
        recent_cutoff = pd.Timestamp(self.current_year - 3, 1, 1, tz='UTC')
        baseline_cutoff = pd.Timestamp(self.current_year - 6, 1, 1, tz='UTC')
        
        # Recent vs baseline analysis by country
        recent_data = self.df[self.df['transaction_date'] >= recent_cutoff]
        baseline_data = self.df[
            (self.df['transaction_date'] >= baseline_cutoff) & 
            (self.df['transaction_date'] < recent_cutoff)
        ]
        
        # Country growth analysis
        self.country_analysis = self._calculate_country_growth(recent_data, baseline_data)
        
        # Regional analysis
        self.regional_analysis = self._calculate_regional_patterns()
        
        # Expansion opportunities
        self.expansion_opportunities = self._identify_expansion_opportunities()
    
    
    def _calculate_country_growth(self, recent_data: pd.DataFrame, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth patterns by country.
        
        Args:
            recent_data: Recent period data
            baseline_data: Baseline period data
            
        Returns:
            DataFrame with country growth analysis
        """
        
        # Recent metrics
        recent_metrics = recent_data.groupby('project_country').agg({
            'credits_quantity': 'sum',
            'project_category': 'nunique'
        }).rename(columns={'credits_quantity': 'recent_volume', 'project_category': 'recent_categories'})
        
        # Baseline metrics
        baseline_metrics = baseline_data.groupby('project_country').agg({
            'credits_quantity': 'sum',
            'project_category': 'nunique'
        }).rename(columns={'credits_quantity': 'baseline_volume', 'project_category': 'baseline_categories'})
        
        # Merge and calculate growth
        growth_df = recent_metrics.join(baseline_metrics, how='outer').fillna(0)
        
        # Calculate growth rates
        growth_df['volume_growth_rate'] = np.where(
            growth_df['baseline_volume'] > 0,
            ((growth_df['recent_volume'] - growth_df['baseline_volume']) / growth_df['baseline_volume'] * 100),
            np.where(growth_df['recent_volume'] > 0, 999, 0)  # New markets get 999% (emerging)
        ).round(1)
        
        growth_df['category_expansion'] = (
            growth_df['recent_categories'] - growth_df['baseline_categories']
        ).astype(int)
        
        # Market classification
        growth_df['market_status'] = growth_df.apply(self._classify_market_status, axis=1)
        
        # Add country data
        growth_df = growth_df.join(self.country_data[['market_share', 'market_age_years', 'region']])
        
        # Calculate opportunity score
        growth_df['opportunity_score'] = self._calculate_opportunity_score(growth_df)
        
        return growth_df.sort_values('opportunity_score', ascending=False)
    
    
    def _classify_market_status(self, row) -> str:
        """
        Classify market status based on growth and volume patterns.
        
        Args:
            row: DataFrame row with growth metrics
            
        Returns:
            Market status classification
        """
        
        growth_rate = row['volume_growth_rate']
        recent_volume = row['recent_volume']
        baseline_volume = row['baseline_volume']
        
        if baseline_volume == 0 and recent_volume > 0:
            return "EMERGING"
        elif growth_rate >= 100:
            return "BOOMING"
        elif growth_rate >= 50:
            return "GROWING"
        elif growth_rate >= 10:
            return "DEVELOPING"
        elif growth_rate >= -10:
            return "STABLE"
        elif growth_rate >= -50:
            return "DECLINING"
        else:
            return "CONTRACTING"
    
    
    def _calculate_opportunity_score(self, growth_df: pd.DataFrame) -> pd.Series:
        """
        Calculate opportunity score for each country.
        
        Args:
            growth_df: DataFrame with growth metrics
            
        Returns:
            Series with opportunity scores (0-100)
        """
        
        # Normalize components to 0-100 scale
        volume_component = np.clip(growth_df['volume_growth_rate'] / 200 * 50 + 50, 0, 100)
        
        # Category expansion component
        max_categories = growth_df['category_expansion'].max()
        category_component = np.clip(growth_df['category_expansion'] / max(max_categories, 1) * 100, 0, 100)
        
        # Market share component (lower share = higher opportunity)
        market_share_component = np.clip((5 - growth_df['market_share'].fillna(0)) / 5 * 100, 0, 100)
        
        # Combined score (weighted)
        opportunity_score = (
            volume_component * 0.5 +       # Growth rate (50%)
            category_component * 0.3 +     # Category expansion (30%)
            market_share_component * 0.2   # Market share opportunity (20%)
        )
        
        return opportunity_score.round(1)
    
    
    def _calculate_regional_patterns(self) -> Dict:
        """
        Calculate regional growth patterns and opportunities.
        
        Returns:
            Dictionary with regional analysis
        """
        
        regional_data = self.country_analysis.groupby('region').agg({
            'recent_volume': 'sum',
            'baseline_volume': 'sum',
            'volume_growth_rate': 'mean',
            'opportunity_score': 'mean',
            'market_share': 'sum'
        }).round(2)
        
        regional_data['countries_count'] = self.country_analysis.groupby('region').size()
        
        # Regional growth calculation
        regional_data['regional_growth_rate'] = np.where(
            regional_data['baseline_volume'] > 0,
            ((regional_data['recent_volume'] - regional_data['baseline_volume']) / regional_data['baseline_volume'] * 100),
            999
        ).round(1)
        
        # Regional classification
        regional_data['regional_status'] = regional_data['regional_growth_rate'].apply(
            lambda x: "EMERGING" if x >= 100 else "GROWING" if x >= 25 else "STABLE" if x >= -10 else "DECLINING"
        )
        
        return regional_data.to_dict('index')
    
    
    def _identify_expansion_opportunities(self) -> Dict:
        """
        Identify top expansion opportunities.
        
        Returns:
            Dictionary with expansion opportunities
        """
        
        # Top opportunities by different criteria
        top_emerging = self.country_analysis[
            self.country_analysis['market_status'] == 'EMERGING'
        ].head(10)
        
        top_growing = self.country_analysis[
            (self.country_analysis['volume_growth_rate'] > 25) & 
            (self.country_analysis['market_status'] != 'EMERGING')
        ].head(10)
        
        top_underserved = self.country_analysis[
            (self.country_analysis['market_share'] < 1.0) & 
            (self.country_analysis['recent_volume'] > 0)
        ].sort_values('opportunity_score', ascending=False).head(10)
        
        # Category-specific opportunities
        category_opportunities = self._find_category_opportunities()
        
        return {
            'emerging_markets': top_emerging.index.tolist() if not top_emerging.empty else [],
            'high_growth_markets': top_growing.index.tolist() if not top_growing.empty else [],
            'underserved_markets': top_underserved.index.tolist() if not top_underserved.empty else [],
            'category_expansion': category_opportunities
        }
    
    
    def _find_category_opportunities(self) -> Dict:
        """
        Find category expansion opportunities by country.
        
        Returns:
            Dictionary with category opportunities
        """
        
        opportunities = {}
        
        # For each category, find countries with low presence but market potential
        for category in self.category_country_matrix.index[:20]:  # Top 20 categories
            category_data = self.category_country_matrix.loc[category]
            
            # Countries with this category
            active_countries = category_data[category_data > 0].index.tolist()
            
            # Countries without this category but with high opportunity scores
            potential_countries = self.country_analysis[
                ~self.country_analysis.index.isin(active_countries) &
                (self.country_analysis['opportunity_score'] > 60)
            ].head(5)
            
            if not potential_countries.empty:
                opportunities[category] = potential_countries.index.tolist()
        
        return opportunities
    
    
    def get_country_analysis(self, country: str) -> Optional[Dict]:
        """
        Get detailed analysis for a specific country.
        
        Args:
            country: Country name
            
        Returns:
            Dictionary with country analysis or None if not found
        """
        
        if country not in self.country_analysis.index:
            return None
        
        country_data = self.country_analysis.loc[country]
        base_data = self.country_data.loc[country]
        
        # Category presence in this country
        country_categories = self.category_country_matrix[country]
        active_categories = country_categories[country_categories > 0].sort_values(ascending=False)
        
        # Regional context
        region_data = self.regional_analysis.get(country_data['region'], {})
        
        return {
            'country': country,
            'region': country_data['region'],
            'market_status': country_data['market_status'],
            'volume_growth_rate': country_data['volume_growth_rate'],
            'opportunity_score': country_data['opportunity_score'],
            'market_share': country_data['market_share'],
            'market_age_years': base_data['market_age_years'],
            'total_volume': base_data['total_volume'],
            'annual_volume': base_data['annual_volume'],
            'categories_active': len(active_categories),
            'top_categories': active_categories.head(5).to_dict(),
            'recent_volume': country_data['recent_volume'],
            'baseline_volume': country_data['baseline_volume'],
            'category_expansion': country_data['category_expansion'],
            'regional_context': {
                'region': country_data['region'],
                'regional_growth': region_data.get('regional_growth_rate', 0),
                'regional_status': region_data.get('regional_status', 'UNKNOWN')
            },
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
    
    
    def get_regional_insights(self) -> Dict:
        """
        Get regional market insights.
        
        Returns:
            Dictionary with regional insights
        """
        
        regional_summary = {}
        
        for region, data in self.regional_analysis.items():
            regional_summary[region] = {
                'countries_count': data['countries_count'],
                'regional_growth_rate': data['regional_growth_rate'],
                'regional_status': data['regional_status'],
                'total_market_share': data['market_share'],
                'avg_opportunity_score': data['opportunity_score'],
                'recent_volume': data['recent_volume'],
                'baseline_volume': data['baseline_volume']
            }
        
        # Find regional leaders
        growth_leader = max(self.regional_analysis.items(), key=lambda x: x[1]['regional_growth_rate'])
        volume_leader = max(self.regional_analysis.items(), key=lambda x: x[1]['recent_volume'])
        opportunity_leader = max(self.regional_analysis.items(), key=lambda x: x[1]['opportunity_score'])
        
        return {
            'regional_data': regional_summary,
            'leaders': {
                'fastest_growing_region': growth_leader[0],
                'largest_volume_region': volume_leader[0],
                'highest_opportunity_region': opportunity_leader[0]
            },
            'total_regions': len(self.regional_analysis),
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
    
    
    def get_expansion_recommendations(self, category: str = None, investment_size: str = "MEDIUM") -> Dict:
        """
        Get expansion recommendations based on criteria.
        
        Args:
            category: Specific category to analyze (optional)
            investment_size: Investment size (SMALL/MEDIUM/LARGE)
            
        Returns:
            Dictionary with expansion recommendations
        """
        
        recommendations = {
            'priority_markets': [],
            'secondary_markets': [],
            'watch_list': [],
            'avoid_list': []
        }
        
        # Filter by investment size
        if investment_size == "SMALL":
            # Focus on smaller, emerging markets
            candidates = self.country_analysis[
                (self.country_analysis['market_share'] < 0.5) & 
                (self.country_analysis['opportunity_score'] > 50)
            ]
        elif investment_size == "LARGE":
            # Focus on established, high-volume markets
            candidates = self.country_analysis[
                (self.country_analysis['market_share'] > 1.0) | 
                (self.country_analysis['recent_volume'] > self.country_analysis['recent_volume'].quantile(0.75))
            ]
        else:  # MEDIUM
            # Balanced approach
            candidates = self.country_analysis[
                self.country_analysis['opportunity_score'] > 40
            ]
        
        # Category-specific filtering
        if category and category in self.category_country_matrix.index:
            category_presence = self.category_country_matrix.loc[category]
            
            # Countries without this category (greenfield opportunity)
            greenfield = candidates[
                ~candidates.index.isin(category_presence[category_presence > 0].index)
            ].head(5)
            
            # Countries with low presence (expansion opportunity)
            expansion = candidates[
                candidates.index.isin(category_presence[
                    (category_presence > 0) & (category_presence < category_presence.quantile(0.5))
                ].index)
            ].head(5)
            
            recommendations['greenfield_opportunities'] = greenfield.index.tolist()
            recommendations['expansion_opportunities'] = expansion.index.tolist()
        
        # General recommendations
        sorted_candidates = candidates.sort_values('opportunity_score', ascending=False)
        
        recommendations['priority_markets'] = sorted_candidates.head(5).index.tolist()
        recommendations['secondary_markets'] = sorted_candidates.iloc[5:10].index.tolist()
        
        # Watch list (good potential but some risks)
        watch_candidates = self.country_analysis[
            (self.country_analysis['opportunity_score'] > 30) & 
            (self.country_analysis['market_status'].isin(['STABLE', 'DEVELOPING']))
        ]
        recommendations['watch_list'] = watch_candidates.head(5).index.tolist()
        
        # Avoid list (declining markets)
        avoid_candidates = self.country_analysis[
            self.country_analysis['market_status'].isin(['DECLINING', 'CONTRACTING'])
        ]
        recommendations['avoid_list'] = avoid_candidates.head(5).index.tolist()
        
        return recommendations
    
    
    def calculate_diversification_score(self, current_countries: List[str]) -> Dict:
        """
        Calculate geographic diversification score for a portfolio.
        
        Args:
            current_countries: List of countries currently in portfolio
            
        Returns:
            Dictionary with diversification analysis
        """
        
        if not current_countries:
            return {'diversification_score': 0, 'recommendations': []}
        
        # Current portfolio analysis
        current_data = self.country_analysis.loc[
            self.country_analysis.index.intersection(current_countries)
        ]
        
        if current_data.empty:
            return {'diversification_score': 0, 'recommendations': []}
        
        # Regional diversification
        current_regions = current_data['region'].value_counts()
        total_regions = self.country_analysis['region'].nunique()
        regional_diversity = len(current_regions) / total_regions * 100
        
        # Market status diversification
        current_statuses = current_data['market_status'].value_counts()
        total_statuses = self.country_analysis['market_status'].nunique()
        status_diversity = len(current_statuses) / total_statuses * 100
        
        # Volume concentration (HHI)
        volume_shares = current_data['recent_volume'] / current_data['recent_volume'].sum()
        hhi = (volume_shares ** 2).sum()
        concentration_score = (1 - hhi) * 100  # Lower HHI = better diversification
        
        # Overall diversification score
        diversification_score = (
            regional_diversity * 0.4 +
            status_diversity * 0.3 +
            concentration_score * 0.3
        )
        
        # Recommendations for improvement
        missing_regions = set(self.country_analysis['region'].unique()) - set(current_regions.index)
        region_recommendations = []
        
        for region in missing_regions:
            region_countries = self.country_analysis[
                self.country_analysis['region'] == region
            ].sort_values('opportunity_score', ascending=False)
            
            if not region_countries.empty:
                region_recommendations.append({
                    'region': region,
                    'top_country': region_countries.index[0],
                    'opportunity_score': region_countries.iloc[0]['opportunity_score']
                })
        
        return {
            'diversification_score': round(diversification_score, 1),
            'regional_diversity': round(regional_diversity, 1),
            'status_diversity': round(status_diversity, 1),
            'concentration_score': round(concentration_score, 1),
            'current_regions': current_regions.to_dict(),
            'missing_regions': len(missing_regions),
            'region_recommendations': sorted(region_recommendations, key=lambda x: x['opportunity_score'], reverse=True)[:3],
            'portfolio_countries': len(current_countries),
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }


def render_geographic_predictor_interface(df: pd.DataFrame) -> None:
    """
    Render the Streamlit interface for the Geographic Expansion Predictor.
    
    Args:
        df: DataFrame containing historical transaction data
    """
    
    st.header("Preditor de Expansão Geográfica")
    st.markdown("**Identifique oportunidades de mercado global em 109 países**")
    
    # Initialize predictor
    with st.spinner("Analisando padrões de mercado global..."):
        predictor = GeographicPredictor(df)
    
    # Create tabs for different analyses
    country_tab, regional_tab, expansion_tab, portfolio_tab = st.tabs([
        "Análise por País",
        "Insights Regionais",
        "Oportunidades de Expansão",
        "Diversificação de Portfolio"
    ])
    
    with country_tab:
        render_country_analysis(predictor)
    
    with regional_tab:
        render_regional_insights(predictor)
    
    with expansion_tab:
        render_expansion_opportunities(predictor)
    
    with portfolio_tab:
        render_portfolio_diversification(predictor)


def render_country_analysis(predictor: GeographicPredictor) -> None:
    """Render country-specific analysis interface."""
    
    st.subheader("🎯 Análise Individual por País")
    
    # Country selection
    available_countries = list(predictor.country_analysis.index)
    
    if available_countries:
        selected_country = st.selectbox(
            "🌍 Selecionar País para Análise",
            available_countries,
            help="Escolha um país para analisar seu potencial de mercado"
        )
        
        if st.button("🎯 Analisar País", type="primary"):
            # Get country analysis
            country_data = predictor.get_country_analysis(selected_country)
            
            if country_data:
                # Display key metrics
                st.markdown("### 📊 Análise de Mercado do País")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    status_color = {
                        'EMERGING': '🟢', 'BOOMING': '🟢', 'GROWING': '🟢',
                        'DEVELOPING': '🟡', 'STABLE': '🟡',
                        'DECLINING': '🔴', 'CONTRACTING': '🔴'
                    }.get(country_data['market_status'], '⚪')
                    
                    st.metric(
                        "Status do Mercado",
                        f"{status_color} {country_data['market_status']}",
                        help="Estágio atual de desenvolvimento do mercado"
                    )
                
                with col2:
                    growth_delta = f"{country_data['volume_growth_rate']:+.1f}%"
                    st.metric(
                        "Taxa de Crescimento do Volume",
                        f"{country_data['volume_growth_rate']:.1f}%",
                        delta=growth_delta,
                        help="Taxa de crescimento do volume de 3 anos"
                    )
                
                with col3:
                    st.metric(
                        "Pontuação de Oportunidade",
                        f"{country_data['opportunity_score']:.1f}/100",
                        help="Pontuação geral de oportunidade de expansão"
                    )
                
                with col4:
                    st.metric(
                        "Participação de Mercado",
                        f"{country_data['market_share']:.3f}%",
                        help="Participação do volume global de créditos de carbono"
                    )
                
                # Market interpretation
                if country_data['market_status'] in ['EMERGING', 'BOOMING']:
                    st.success(f"🟢 **ALTO POTENCIAL**: {selected_country} mostra mercado {country_data['market_status'].lower()} com {country_data['volume_growth_rate']:+.1f}% crescimento.")
                elif country_data['market_status'] in ['GROWING', 'DEVELOPING']:
                    st.info(f"🟡 **POTENCIAL MODERADO**: {selected_country} é um mercado {country_data['market_status'].lower()} com progresso constante.")
                elif country_data['market_status'] == 'STABLE':
                    st.warning(f"🟠 **MERCADO ESTÁVEL**: {selected_country} mostra performance estável com crescimento limitado.")
                else:
                    st.error(f"🔴 **CAUTELA**: {selected_country} mostra tendências {country_data['market_status'].lower()}.")
                
                # Detailed market information
                st.markdown("### 📈 Detalhes do Mercado")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**Características do Mercado:**")
                    st.write(f"• **Região**: {country_data['region']}")
                    st.write(f"• **Idade do Mercado**: {country_data['market_age_years']:.1f} anos")
                    st.write(f"• **Volume Total**: {country_data['total_volume']:,.0f} tCO₂")
                    st.write(f"• **Volume Anual**: {country_data['annual_volume']:,.0f} tCO₂")
                    st.write(f"• **Categorias Ativas**: {country_data['categories_active']}")
                
                with detail_col2:
                    st.markdown("**Análise de Crescimento:**")
                    st.write(f"• **Volume Recente**: {country_data['recent_volume']:,.0f} tCO₂")
                    st.write(f"• **Volume Base**: {country_data['baseline_volume']:,.0f} tCO₂")
                    st.write(f"• **Expansão de Categorias**: {country_data['category_expansion']:+d} categorias")
                    
                    regional_ctx = country_data['regional_context']
                    st.write(f"• **Crescimento Regional**: {regional_ctx['regional_growth']:.1f}%")
                    st.write(f"• **Status Regional**: {regional_ctx['regional_status']}")
                
                # Top categories in this country
                if country_data['top_categories']:
                    st.markdown("### 🏷️ Principais Categorias no País")
                    
                    for i, (category, volume) in enumerate(list(country_data['top_categories'].items())[:5], 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{i}. {category[:40]}{'...' if len(category) > 40 else ''}**")
                        with col2:
                            st.write(f"{volume:,.0f} tCO₂")
                
                # Investment recommendation
                opportunity_score = country_data['opportunity_score']
                if opportunity_score >= 80:
                    st.success("🟢 **FORTE RECOMENDAÇÃO**: País de alta oportunidade para expansão imediata.")
                elif opportunity_score >= 60:
                    st.info("🟡 **BOA OPORTUNIDADE**: Condições favoráveis de mercado para expansão.")
                elif opportunity_score >= 40:
                    st.warning("🟠 **OPORTUNIDADE MODERADA**: Considere entrada no mercado com planejamento cuidadoso.")
                else:
                    st.error("🔴 **BAIXA PRIORIDADE**: Oportunidade limitada ou mercado de alto risco.")
    
    else:
        st.error("❌ Nenhum dado de país disponível para análise.")


def render_regional_insights(predictor: GeographicPredictor) -> None:
    """Render regional insights interface."""
    
    st.subheader("🌎 Insights de Mercado Regional")
    
    # Get regional insights
    regional_data = predictor.get_regional_insights()
    
    if regional_data:
        # Regional overview
        st.markdown("### 📊 Visão Geral Regional")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric(
                "Total de Regiões",
                regional_data['total_regions']
            )
        
        with overview_col2:
            st.metric(
                "Crescimento Mais Rápido",
                regional_data['leaders']['fastest_growing_region']
            )
        
        with overview_col3:
            st.metric(
                "Maior Volume",
                regional_data['leaders']['largest_volume_region']
            )
        
        with overview_col4:
            st.metric(
                "Maior Oportunidade",
                regional_data['leaders']['highest_opportunity_region']
            )
        
        # Detailed regional analysis
        st.markdown("### 🌍 Análise de Performance Regional")
        
        regional_metrics = []
        for region, data in regional_data['regional_data'].items():
            regional_metrics.append({
                'Região': region.replace('_', ' ').title(),
                'Países': data['countries_count'],
                'Taxa de Crescimento': f"{data['regional_growth_rate']:.1f}%",
                'Status': data['regional_status'],
                'Participação de Mercado': f"{data['total_market_share']:.2f}%",
                'Pontuação de Oportunidade': f"{data['avg_opportunity_score']:.1f}",
                'Volume Recente': f"{data['recent_volume']:,.0f}"
            })
        
        regional_df = pd.DataFrame(regional_metrics)
        st.dataframe(regional_df, use_container_width=True)
        
        # Regional strategy insights
        st.markdown("### 💡 Insights de Estratégia Regional")
        
        leaders = regional_data['leaders']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🚀 **Região de Crescimento Mais Rápido**: {leaders['fastest_growing_region'].replace('_', ' ').title()}")
            st.write("Considere expansão prioritária nesta região de alto crescimento")
            
            st.info(f"📊 **Maior Mercado**: {leaders['largest_volume_region'].replace('_', ' ').title()}")
            st.write("Mercado estabelecido com potencial de volume comprovado")
        
        with col2:
            st.warning(f"🎯 **Maior Oportunidade**: {leaders['highest_opportunity_region'].replace('_', ' ').title()}")
            st.write("Melhor combinação de fatores de crescimento e oportunidade")
            
            # Regional diversification advice
            high_opportunity_regions = [
                region for region, data in regional_data['regional_data'].items()
                if data['avg_opportunity_score'] > 50
            ]
            
            st.markdown("**Estratégia de Diversificação:**")
            st.write(f"• Foque em {len(high_opportunity_regions)} regiões de alta oportunidade")
            st.write(f"• Balance mercados estabelecidos com oportunidades emergentes")
        
        # Regional growth comparison
        st.markdown("### 📈 Comparação de Crescimento Regional")
        
        growth_insights = []
        for region, data in regional_data['regional_data'].items():
            growth_rate = data['regional_growth_rate']
            status = data['regional_status']
            
            if status == 'EMERGING':
                growth_insights.append(f"🟢 **{region.replace('_', ' ').title()}**: {growth_rate:.1f}% (Oportunidade de mercado emergente)")
            elif status == 'GROWING':
                growth_insights.append(f"🟡 **{region.replace('_', ' ').title()}**: {growth_rate:.1f}% (Trajetória sólida de crescimento)")
            elif status == 'STABLE':
                growth_insights.append(f"🟠 **{region.replace('_', ' ').title()}**: {growth_rate:.1f}% (Mercado estável e maduro)")
            else:
                growth_insights.append(f"🔴 **{region.replace('_', ' ').title()}**: {growth_rate:.1f}% (Mercado em declínio)")
        
        for insight in growth_insights:
            st.write(insight)
    
    else:
        st.error("❌ Incapaz de gerar insights regionais.")


def render_expansion_opportunities(predictor: GeographicPredictor) -> None:
    """Render expansion opportunities interface."""
    
    st.subheader("🚀 Oportunidades de Expansão de Mercado")
    
    # Expansion criteria
    expansion_col1, expansion_col2 = st.columns(2)
    
    with expansion_col1:
        investment_size = st.selectbox(
            "💰 Tamanho do Investimento",
            ["SMALL", "MEDIUM", "LARGE"],
            index=1,
            help="Escolha o tamanho do investimento para obter recomendações personalizadas"
        )
    
    with expansion_col2:
        # Get available categories
        available_categories = list(predictor.category_country_matrix.index)
        category_filter = st.selectbox(
            "🏷️ Foco em Categoria (Opcional)",
            ["Todas as Categorias"] + available_categories[:20],
            help="Foque em oportunidades de categoria específica"
        )
        
        if category_filter == "Todas as Categorias":
            category_filter = None
    
    if st.button("🚀 Gerar Plano de Expansão", type="primary"):
        # Get expansion recommendations
        recommendations = predictor.get_expansion_recommendations(category_filter, investment_size)
        
        if recommendations:
            st.markdown("### 🎯 Recomendações de Expansão")
            
            # Priority markets
            if recommendations['priority_markets']:
                st.success("🟢 **MERCADOS PRIORITÁRIOS** (Foco Imediato)")
                for i, country in enumerate(recommendations['priority_markets'], 1):
                    country_data = predictor.get_country_analysis(country)
                    if country_data:
                        st.write(f"**{i}. {country}** - Mercado {country_data['market_status']} ({country_data['opportunity_score']:.1f}/100)")
            
            # Secondary markets
            if recommendations['secondary_markets']:
                st.info("🟡 **MERCADOS SECUNDÁRIOS** (Alvos de Médio Prazo)")
                for i, country in enumerate(recommendations['secondary_markets'], 1):
                    country_data = predictor.get_country_analysis(country)
                    if country_data:
                        st.write(f"**{i}. {country}** - Mercado {country_data['market_status']} ({country_data['opportunity_score']:.1f}/100)")
            
            # Category-specific opportunities
            if category_filter and 'greenfield_opportunities' in recommendations:
                st.markdown("### 🆕 Oportunidades Específicas por Categoria")
                
                if recommendations['greenfield_opportunities']:
                    st.success(f"🟢 **MERCADOS GREENFIELD** para {category_filter}")
                    st.write("Países sem esta categoria (oportunidade de novo mercado):")
                    for country in recommendations['greenfield_opportunities']:
                        st.write(f"• **{country}**")
                
                if recommendations['expansion_opportunities']:
                    st.info(f"🟡 **MERCADOS DE EXPANSÃO** para {category_filter}")
                    st.write("Países com baixa presença (oportunidade de crescimento):")
                    for country in recommendations['expansion_opportunities']:
                        st.write(f"• **{country}**")
            
            # Watch list
            if recommendations['watch_list']:
                st.warning("🟠 **LISTA DE OBSERVAÇÃO** (Monitorar para o Futuro)")
                for country in recommendations['watch_list']:
                    country_data = predictor.get_country_analysis(country)
                    if country_data:
                        st.write(f"• **{country}** - {country_data['market_status']} ({country_data['opportunity_score']:.1f}/100)")
            
            # Avoid list
            if recommendations['avoid_list']:
                st.error("🔴 **LISTA DE EVITAR** (Mercados de Alto Risco)")
                for country in recommendations['avoid_list']:
                    country_data = predictor.get_country_analysis(country)
                    if country_data:
                        st.write(f"• **{country}** - Mercado {country_data['market_status']}")
            
            # Investment size specific advice
            st.markdown("### 💡 Conselho de Estratégia de Investimento")
            
            if investment_size == "SMALL":
                st.info("📊 **Estratégia de Investimento Pequeno**: Foque em mercados emergentes com menor competição e barreiras de entrada.")
            elif investment_size == "LARGE":
                st.info("📊 **Estratégia de Investimento Grande**: Mire em mercados estabelecidos com volume comprovado e potencial de crescimento.")
            else:
                st.info("📊 **Estratégia de Investimento Médio**: Abordagem equilibrada misturando oportunidades emergentes com mercados estáveis.")
        
        else:
            st.error("❌ Incapaz de gerar recomendações de expansão.")


def render_portfolio_diversification(predictor: GeographicPredictor) -> None:
    """Render portfolio diversification interface."""
    
    st.subheader("📊 Diversificação de Portfolio Geográfico")
    
    # Current portfolio input
    st.markdown("### 🗂️ Análise de Portfolio Atual")
    
    available_countries = list(predictor.country_analysis.index)
    
    current_portfolio = st.multiselect(
        "🌍 Selecionar Países do Portfolio Atual",
        available_countries,
        help="Escolha países atualmente em seu portfolio para análise de diversificação"
    )
    
    if len(current_portfolio) >= 2:
        if st.button("📊 Analisar Diversificação", type="primary"):
            # Calculate diversification score
            diversification = predictor.calculate_diversification_score(current_portfolio)
            
            if diversification and diversification['diversification_score'] > 0:
                st.markdown("### 📊 Resultados da Análise de Diversificação")
                
                # Main metrics
                div_col1, div_col2, div_col3, div_col4 = st.columns(4)
                
                with div_col1:
                    score = diversification['diversification_score']
                    score_color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                    st.metric(
                        "Pontuação de Diversificação",
                        f"{score_color} {score:.1f}/100"
                    )
                
                with div_col2:
                    st.metric(
                        "Diversidade Regional",
                        f"{diversification['regional_diversity']:.1f}%"
                    )
                
                with div_col3:
                    st.metric(
                        "Diversidade de Status",
                        f"{diversification['status_diversity']:.1f}%"
                    )
                
                with div_col4:
                    st.metric(
                        "Países do Portfolio",
                        diversification['portfolio_countries']
                    )
                
                # Diversification interpretation
                score = diversification['diversification_score']
                if score >= 70:
                    st.success("🟢 **EXCELENTE DIVERSIFICAÇÃO**: Seu portfolio tem forte diversificação geográfica.")
                elif score >= 50:
                    st.info("🟡 **BOA DIVERSIFICAÇÃO**: Diversificação sólida com espaço para melhoria.")
                elif score >= 30:
                    st.warning("🟠 **DIVERSIFICAÇÃO MODERADA**: Considere expandir para novas regiões/mercados.")
                else:
                    st.error("🔴 **BAIXA DIVERSIFICAÇÃO**: Alto risco de concentração - diversificação fortemente recomendada.")
                
                # Current portfolio breakdown
                st.markdown("### 🗂️ Composição do Portfolio")
                
                breakdown_col1, breakdown_col2 = st.columns(2)
                
                with breakdown_col1:
                    st.markdown("**Distribuição Regional:**")
                    for region, count in diversification['current_regions'].items():
                        st.write(f"• **{region.replace('_', ' ').title()}**: {count} países")
                
                with breakdown_col2:
                    st.markdown("**Métricas de Diversificação:**")
                    st.write(f"• **Pontuação de Concentração**: {diversification['concentration_score']:.1f}/100")
                    st.write(f"• **Regiões Ausentes**: {diversification['missing_regions']}")
                    st.write(f"• **Cobertura Regional**: {len(diversification['current_regions'])} regiões")
                
                # Improvement recommendations
                if diversification['region_recommendations']:
                    st.markdown("### 💡 Recomendações de Diversificação")
                    
                    st.info("🎯 **Adições Recomendadas** (por região):")
                    for rec in diversification['region_recommendations']:
                        st.write(f"• **{rec['region'].replace('_', ' ').title()}**: {rec['top_country']} (Pontuação: {rec['opportunity_score']:.1f})")
                
                # Portfolio optimization suggestions
                st.markdown("### 🔧 Otimização de Portfolio")
                
                optimization_suggestions = [
                    f"📈 **Meta de Pontuação**: Obtenha uma pontuação de diversificação de 70+ para distribuição de risco ótima",
                    f"🌍 **Equilíbrio Regional**: Considere adicionar países de {diversification['missing_regions']} regiões ausentes",
                    f"⚖️ **Equilíbrio de Risco**: Misture mercados emergentes com mercados estabelecidos",
                    f"📊 **Revisão Regular**: Reavalie a diversificação trimestralmente à medida que os mercados evoluem"
                ]
                
                for suggestion in optimization_suggestions:
                    st.write(suggestion)
            
            else:
                st.error("❌ Incapaz de calcular a pontuação de diversificação.")
    
    elif len(current_portfolio) == 1:
        st.warning("⚠️ Adicione pelo menos 2 países para analisar a diversificação.")
    
    else:
        st.info("📝 Selecione países de seu portfolio atual para analisar a diversificação geográfica.")
        
        # Show diversification benefits
        st.markdown("### 💡 Benefícios da Diversificação Geográfica")
        
        benefits = [
            "🛡️ **Redução de Risco**: Espalhe a exposição em diferentes condições de mercado",
            "📈 **Oportunidades de Crescimento**: Acesse diferentes mercados emergentes",
            "🔄 **Equilíbrio Sazonal**: Padrões sazonais diferentes em diferentes regiões",
            "💱 **Diversificação Cambial**: Reduza a exposição a uma única moeda",
            "🌍 **Acesso ao Mercado**: Acesso mais amplo a diferentes tipos de créditos de carbono",
            "📊 **Estabilidade**: Reduza a volatilidade do portfolio através da diversificação geográfica"
        ]
        
        for benefit in benefits:
            st.write(benefit) 