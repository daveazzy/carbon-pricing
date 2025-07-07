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
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.category_risk_profiles = None
        self.risk_thresholds = {
            'LOW': 0.8,
            'MEDIUM': 1.5,
            'HIGH': float('inf')
        }
        self._analyze_category_volatility()
    
    
    def _analyze_category_volatility(self) -> None:
        
        category_stats = self.df.groupby('project_category')['credits_quantity'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        category_stats = category_stats[category_stats['count'] >= 10].copy()
        
        category_stats['cv'] = (category_stats['std'] / category_stats['mean']).round(3)
        category_stats['cv'] = category_stats['cv'].fillna(0)
        
        category_stats['range_ratio'] = (category_stats['max'] / category_stats['min']).round(2)
        category_stats['volatility_score'] = (category_stats['cv'] * 100).round(1)
        
        category_stats['risk_level'] = category_stats['cv'].apply(self._classify_risk_level)
        category_stats['risk_score'] = category_stats['cv'].apply(self._calculate_risk_score)
        
        category_stats['risk_percentile'] = category_stats['cv'].rank(pct=True) * 100
        category_stats['risk_percentile'] = category_stats['risk_percentile'].round(1)
        
        category_stats = category_stats.sort_values('cv')
        
        self.category_risk_profiles = category_stats
    
    
    def _classify_risk_level(self, cv: float) -> str:
        
        if cv < self.risk_thresholds['LOW']:
            return 'LOW'
        elif cv < self.risk_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    
    def _calculate_risk_score(self, cv: float) -> int:
        
        if cv <= 0:
            return 0
        
        max_cv = 3.0
        normalized_cv = min(cv / max_cv, 1.0)
        
        return int(normalized_cv * 100)
    
    
    def get_category_risk_analysis(self, category: str) -> Optional[Dict]:
        
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
        
        if risk_level == 'LOW':
            return "CONSERVATIVE - Stable investment suitable for risk-averse portfolios"
        elif risk_level == 'MEDIUM':
            return "BALANCED - Moderate risk with potential for stable returns"
        else:
            return "AGGRESSIVE - High volatility requires careful risk management"
    
    
    def get_risk_ranking(self, top_n: int = 10) -> List[Dict]:
        
        if self.category_risk_profiles is None:
            return []
        
        top_categories = self.category_risk_profiles.head(top_n)
        
        result = []
        for idx, (category, data) in enumerate(top_categories.iterrows(), 1):
            result.append({
                'rank': idx,
                'category': category,
                'risk_level': data['risk_level'],
                'cv': data['cv'],
                'risk_score': data['risk_score'],
                'transaction_count': int(data['count']),
                'volatility_score': data['volatility_score']
            })
        
        return result
    
    
    def get_highest_risk_categories(self, top_n: int = 5) -> List[Dict]:
        
        if self.category_risk_profiles is None:
            return []
        
        highest_risk = self.category_risk_profiles.tail(top_n)
        
        result = []
        for idx, (category, data) in enumerate(highest_risk.iterrows(), 1):
            result.append({
                'rank': idx,
                'category': category,
                'risk_level': data['risk_level'],
                'cv': data['cv'],
                'risk_score': data['risk_score'],
                'transaction_count': int(data['count']),
                'volatility_score': data['volatility_score']
            })
        
        return result
    
    
    def analyze_portfolio_risk(self, portfolio_categories: List[str], 
                              portfolio_weights: Optional[List[float]] = None) -> Dict: 