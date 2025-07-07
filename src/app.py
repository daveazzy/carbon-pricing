"""
Carbon Credits Analytics Platform - Professional Interface

This is the main Streamlit application file for the Carbon Credits Analytics Platform.
Clean, modern interface with professional header navigation and optimized layout.
"""

import streamlit as st
from modules.data_loader import carregar_e_preparar_dados
from modules.ui.config import setup_page_config, display_error_message
from modules.ui.tabs.comparative_overview import render_comparative_overview_tab
from modules.ui.tabs.distribution_analysis import render_distribution_analysis_tab
from modules.ui.tabs.predictive_calculators import render_predictive_calculators_tab
from modules.analysis.descriptive import calcular_analise_enriquecida_por_categoria
from modules.analysis.inferential import realizar_teste_mann_whitney, calcular_tabela_contingencia, realizar_teste_qui_quadrado
from modules.analysis.modeling import treinar_modelo_regressao


def main():
    """
    Main function to run the Carbon Credits Analytics Platform.
    
    This function sets up the page configuration, loads data, and renders
    the professional interface with header navigation.
    """
    
    # Setup page configuration
    setup_page_config()
    
    # Load data first
    with st.spinner('Carregando dados de créditos de carbono...'):
        df = carregar_e_preparar_dados()

    if df is None:
        display_error_message("Falha ao carregar dados. Verifique os arquivos de dados e tente novamente.")
        return
    
    # Create professional header with navigation
    create_professional_header(df)
    
    # Handle navigation
    handle_navigation(df)


def create_professional_header(df):
    """Create a professional header with logo, title, and main navigation."""
    
    # Load custom CSS for the new layout
    from modules.ui.styles import load_custom_css
    load_custom_css()
    
    # Professional header container
    st.markdown("""
    <div class="professional-header">
        <div class="header-content">
            <div class="logo-section">
                <!-- Logo will be added here -->
                <div class="logo-placeholder">🌍</div>
                <div class="title-section">
                    <h1 class="main-title">Plataforma de Análise de Créditos de Carbono</h1>
                    <p class="subtitle">Inteligência de Mercado para Investimento e Negociação</p>
                </div>
            </div>
            <div class="metrics-section">
                <div class="metric-pill">
                    <span class="metric-label">Transações</span>
                    <span class="metric-value">{:,}</span>
                </div>
                <div class="metric-pill">
                    <span class="metric-label">Volume Total</span>
                    <span class="metric-value">{:,.0f} tCO₂</span>
                </div>
                <div class="metric-pill">
                    <span class="metric-label">Categorias</span>
                    <span class="metric-value">{}</span>
                </div>
                <div class="metric-pill">
                    <span class="metric-label">Países</span>
                    <span class="metric-value">{}</span>
                </div>
            </div>
        </div>
    </div>
    """.format(
        len(df),
        int(df['credits_quantity'].sum()),
        len(df['project_category'].unique()),
        len(df['project_country'].unique())
    ), unsafe_allow_html=True)
    
    # Navigation menu
    create_navigation_menu()


def create_navigation_menu():
    """Create the main navigation menu."""
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Visão Geral'
    
    # Navigation container
    st.markdown("""
    <div class="navigation-container">
        <div class="nav-content">
    """, unsafe_allow_html=True)
    
    # Navigation buttons in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("Visão Geral", key="nav_overview", 
                    type="primary" if st.session_state.current_page == 'Visão Geral' else "secondary"):
            st.session_state.current_page = 'Visão Geral'
    
    with col2:
        if st.button("Análise de Distribuição", key="nav_distribution",
                    type="primary" if st.session_state.current_page == 'Análise de Distribuição' else "secondary"):
            st.session_state.current_page = 'Análise de Distribuição'
    
    with col3:
        if st.button("Análise Estatística", key="nav_statistical",
                    type="primary" if st.session_state.current_page == 'Análise Estatística' else "secondary"):
            st.session_state.current_page = 'Análise Estatística'
    
    with col4:
        if st.button("Inteligência de Mercado", key="nav_intelligence",
                    type="primary" if st.session_state.current_page == 'Inteligência de Mercado' else "secondary"):
            st.session_state.current_page = 'Inteligência de Mercado'
    
    with col5:
        if st.button("Calculadoras Preditivas", key="nav_calculators",
                    type="primary" if st.session_state.current_page == 'Calculadoras Preditivas' else "secondary"):
            st.session_state.current_page = 'Calculadoras Preditivas'
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Add spacing after navigation
    st.markdown("<div class='nav-spacer'></div>", unsafe_allow_html=True)


def handle_navigation(df):
    """Handle navigation and render the appropriate page content."""
    
    current_page = st.session_state.current_page
    
    # Create main content container
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if current_page == 'Visão Geral':
        render_overview_page(df)
    
    elif current_page == 'Análise de Distribuição':
        render_distribution_page(df)
    
    elif current_page == 'Análise Estatística':
        render_statistical_page(df)
    
    elif current_page == 'Inteligência de Mercado':
        render_intelligence_page(df)
    
    elif current_page == 'Calculadoras Preditivas':
        render_calculators_page(df)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_overview_page(df):
    """Render the overview page with improved layout."""
    
    # Page header
    st.markdown("""
    <div class="page-header">
        <h2 class="page-title">Visão Geral do Mercado</h2>
        <p class="page-description">Análise comparativa abrangente dos dados de créditos de carbono</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Content sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main analysis content
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        
        # Optional filters in a clean expandable section
        with st.expander("🔧 Opções de Filtro", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                all_categories = sorted(df['project_category'].dropna().unique())
                selected_categories = st.multiselect(
                    "Categorias de Projeto", 
                    all_categories, 
                    default=all_categories[:8] if len(all_categories) >= 8 else all_categories,
                    help="Selecione categorias específicas para análise focada"
                )
            
            with filter_col2:
                all_countries = sorted(df['project_country'].dropna().unique())
                selected_countries = st.multiselect(
                    "Países de Origem", 
                    all_countries, 
                    default=all_countries[:12] if len(all_countries) >= 12 else all_countries,
                    help="Selecione países específicos para análise"
                )
            
            with filter_col3:
                min_year = int(df['credit_vintage_year'].min())
                max_year = int(df['credit_vintage_year'].max())
                year_range = st.slider(
                    "Período de Análise", 
                    min_year, max_year, 
                    (min_year, max_year),
                    help="Defina o período para análise temporal"
                )
        
        # Apply filters
        df_filtered = apply_filters(df, selected_categories, selected_countries, year_range)
        
        # Show filter impact
        if len(df_filtered) != len(df):
            reduction = ((len(df) - len(df_filtered)) / len(df)) * 100
            st.info(f"📊 Filtros aplicados: **{len(df_filtered):,}** transações selecionadas ({100-reduction:.1f}% do total)")
        
        # Render main analysis
        render_comparative_overview_tab(df_filtered)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar with key insights
        render_insights_sidebar(df_filtered if 'df_filtered' in locals() else df)


def render_insights_sidebar(df):
    """Render a sidebar with key insights and quick stats."""
    
    st.markdown('<div class="insights-sidebar">', unsafe_allow_html=True)
    
    st.markdown("### 📈 Insights Rápidos")
    
    # Top category
    top_category = df.groupby('project_category')['credits_quantity'].sum().idxmax()
    top_volume = df.groupby('project_category')['credits_quantity'].sum().max()
    
    st.markdown(f"""
    **🏆 Categoria Líder:**  
    {top_category}  
    *{top_volume:,.0f} tCO₂*
    """)
    
    # Top country
    top_country = df.groupby('project_country')['credits_quantity'].sum().idxmax()
    country_volume = df.groupby('project_country')['credits_quantity'].sum().max()
    
    st.markdown(f"""
    **🌍 País Líder:**  
    {top_country}  
    *{country_volume:,.0f} tCO₂*
    """)
    
    # Time span
    years_span = int(df['credit_vintage_year'].max() - df['credit_vintage_year'].min())
    st.markdown(f"""
    **📅 Período Analisado:**  
    {years_span} anos de dados  
    *({int(df['credit_vintage_year'].min())} - {int(df['credit_vintage_year'].max())})*
    """)
    
    # Average transaction size
    avg_transaction = df['credits_quantity'].mean()
    st.markdown(f"""
    **💼 Transação Média:**  
    {avg_transaction:,.0f} tCO₂  
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)


def apply_filters(df, categories, countries, year_range):
    """Apply selected filters to the dataframe."""
    
    df_filtered = df.copy()
    
    if categories:
        df_filtered = df_filtered[df_filtered['project_category'].isin(categories)]
    if countries:
        df_filtered = df_filtered[df_filtered['project_country'].isin(countries)]
    
    df_filtered = df_filtered[df_filtered['credit_vintage_year'].between(year_range[0], year_range[1])]
    
    if df_filtered.empty:
        return df  # Return original data if filters result in empty dataset
    
    return df_filtered


def render_distribution_page(df):
    """Render the distribution analysis page."""
    
    st.markdown("""
    <div class="page-header">
        <h2 class="page-title">Análise de Distribuição</h2>
        <p class="page-description">Distribuição de volumes de transação e padrões estatísticos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis scope selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Analisando padrões de distribuição de volumes de transação por categoria e período.")
    
    with col2:
        analysis_scope = st.selectbox(
            "Escopo de Análise", 
            ["Dados Completos", "Top 10 Categorias", "Seleção Personalizada"],
            help="Escolha o escopo para análise de distribuição"
        )
    
    # Apply scope
    df_analysis, selected_categories = apply_analysis_scope(df, analysis_scope)
    
    # Render analysis
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    render_distribution_analysis_tab(df_analysis, selected_categories)
    st.markdown('</div>', unsafe_allow_html=True)


def apply_analysis_scope(df, scope):
    """Apply the selected analysis scope."""
    
    if scope == "Top 10 Categorias":
        top_categories = df.groupby('project_category')['credits_quantity'].sum().nlargest(10).index.tolist()
        df_analysis = df[df['project_category'].isin(top_categories)]
        st.info(f"📊 Analisando as **{len(top_categories)}** categorias com maior volume total")
        return df_analysis, top_categories
    
    elif scope == "Seleção Personalizada":
        all_categories = sorted(df['project_category'].dropna().unique())
        selected_categories = st.multiselect(
            "Selecione categorias específicas:",
            all_categories,
            default=all_categories[:6] if len(all_categories) >= 6 else all_categories,
            help="Escolha categorias específicas para análise detalhada"
        )
        if selected_categories:
            df_analysis = df[df['project_category'].isin(selected_categories)]
            return df_analysis, selected_categories
    
    # Default: complete data
    return df, []


def render_statistical_page(df):
    """Render the statistical analysis page."""
    
    st.markdown("""
    <div class="page-header">
        <h2 class="page-title">Análise Estatística</h2>
        <p class="page-description">Testes estatísticos formais para validação científica de insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Tipo de Análise Estatística",
        ["Comparação de Categorias (Mann-Whitney U)", "Associação País-Categoria (Qui-Quadrado)", "Modelo Preditivo (Regressão)"],
        help="Escolha o tipo de teste estatístico a ser executado"
    )
    
    if "Mann-Whitney" in analysis_type:
        render_category_comparison_analysis(df)
    elif "Qui-Quadrado" in analysis_type:
        render_association_analysis(df)
    elif "Regressão" in analysis_type:
        render_regression_analysis(df)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_intelligence_page(df):
    """Render the market intelligence page."""
    
    st.markdown("""
    <div class="page-header">
        <h2 class="page-title">Inteligência de Mercado</h2>
        <p class="page-description">Insights estratégicos e análise executiva do mercado de carbono</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    render_market_intelligence_tab(df)
    st.markdown('</div>', unsafe_allow_html=True)


def render_calculators_page(df):
    """Render the predictive calculators page."""
    
    st.markdown("""
    <div class="page-header">
        <h2 class="page-title">Calculadoras Preditivas</h2>
        <p class="page-description">Ferramentas avançadas de análise preditiva para otimização de estratégias</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    render_predictive_calculators_tab(df)
    st.markdown('</div>', unsafe_allow_html=True)


def render_category_comparison_analysis(df):
    """Render category comparison using Mann-Whitney test."""
    
    st.markdown("#### Teste Mann-Whitney U: Comparação de Distribuições")
    st.markdown("Compare as distribuições de volume entre duas categorias de projeto.")
    
    # Category selection in columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    all_categories = sorted(df['project_category'].dropna().unique())
    
    with col1:
        cat1 = st.selectbox("Primeira Categoria:", all_categories, index=0)
    
    with col2:
        available_cats = [cat for cat in all_categories if cat != cat1]
        cat2 = st.selectbox("Segunda Categoria:", available_cats, index=0)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("🔬 Executar Teste", type="primary"):
            with st.spinner("Executando teste estatístico..."):
                stat, p_value = realizar_teste_mann_whitney(df, cat1, cat2)
                
                if p_value is not None:
                    # Results display
                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        st.metric("Estatística U", f"{stat:,.0f}")
                    with result_col2:
                        st.metric("Valor P", f"{p_value:.6f}")
                    
                    # Interpretation
                    if p_value < 0.05:
                        st.success("✅ **Diferença Estatisticamente Significativa** (p < 0,05)")
                        st.markdown("As distribuições das duas categorias são significativamente diferentes.")
                    else:
                        st.info("ℹ️ **Nenhuma Diferença Significativa** (p ≥ 0,05)")
                        st.markdown("Não há evidência estatística de diferença entre as distribuições.")
                else:
                    st.error("❌ Teste não pôde ser realizado com os dados disponíveis.")


def render_association_analysis(df):
    """Render association analysis using Chi-Square test."""
    
    st.markdown("#### Teste Qui-Quadrado: Associação País vs Categoria")
    st.markdown("Analise se existe associação estatística entre país de origem e categoria de projeto.")
    
    # Create and display contingency table
    tabela_contingencia = calcular_tabela_contingencia(df)
    
    if not tabela_contingencia.empty:
        with st.expander("📋 Visualizar Tabela de Contingência", expanded=False):
            st.dataframe(tabela_contingencia, use_container_width=True)
        
        if st.button("🔬 Executar Teste Qui-Quadrado", type="primary"):
            with st.spinner("Executando análise de associação..."):
                chi2_result, p_value = realizar_teste_qui_quadrado(tabela_contingencia)
                
                if p_value is not None:
                    # Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estatística χ²", f"{chi2_result:.2f}")
                    with col2:
                        st.metric("Valor P", f"{p_value:.6f}")
                    
                    # Interpretation
                    if p_value is not None and p_value < 0.05:
                        st.success("✅ **Associação Estatisticamente Significativa**")
                        st.markdown("Existe relação significativa entre país de origem e categoria de projeto.")
                    else:
                        st.info("ℹ️ **Nenhuma Associação Significativa**")
                        st.markdown("País e categoria de projeto são estatisticamente independentes.")
                else:
                    st.error("❌ Teste não pôde ser realizado.")
    else:
        st.warning("⚠️ Dados insuficientes para criar tabela de contingência.")


def render_regression_analysis(df):
    """Render regression analysis."""
    
    st.markdown("#### Modelo de Regressão Linear")
    st.markdown("Identifique fatores que influenciam o volume de transações de créditos de carbono.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎯 Treinar Modelo", type="primary"):
            with st.spinner("Treinando modelo de regressão..."):
                modelo = treinar_modelo_regressao(df)
                
                if modelo:
                    st.success("✅ Modelo treinado com sucesso!")
                else:
                    st.error("❌ Erro no treinamento do modelo.")
    
    with col1:
        st.markdown("O modelo analisa como diferentes variáveis impactam o volume de transações.")
    
    # Display results if model was trained
    if 'modelo' in locals() and modelo:
        st.markdown("#### 📊 Resultados do Modelo")
        st.code(str(modelo.summary()), language='text')


def render_market_intelligence_tab(df):
    """Render the Market Intelligence tab with market insights."""
    
    # Market Intelligence Sub-sections
    intel_section = st.selectbox(
        "Seção de Inteligência",
        ["Análise de Mercado", "Dashboard Executivo", "Insights Estratégicos"],
        help="Escolha a seção de inteligência de mercado"
    )
    
    if intel_section == "Análise de Mercado":
        _render_market_analysis(df)
    elif intel_section == "Dashboard Executivo":
        _render_executive_dashboard(df)
    else:
        _render_strategic_insights(df)


def _render_market_analysis(df):
    """Render comprehensive market analysis."""
    
    st.markdown("#### Análise Abrangente de Mercado")
    
    # Category Performance Analysis
    st.markdown("##### Performance por Categoria")
    
    category_analysis = calcular_analise_enriquecida_por_categoria(df)
    
    if not category_analysis.empty:
        # Show top 10 in a clean format
        top_10_categories = category_analysis.head(10)
        
        for i, (category, data) in enumerate(top_10_categories.iterrows(), 1):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                category_name = str(category)
                st.metric(
                    f"{i}. {category_name[:30]}{'...' if len(category_name) > 30 else ''}",
                    f"{data['total_credits_volume']:,.0f} tCO₂"
                )
            with col2:
                st.metric("Transações", f"{data['number_of_transactions']:,}")
            with col3:
                st.metric("Volume Médio", f"{data['mean_transaction_volume']:,.0f} tCO₂")
            with col4:
                st.metric("Volume Mediano", f"{data['median_transaction_volume']:,.0f} tCO₂")
    
    # Geographic Analysis
    st.markdown("##### Análise Geográfica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Países por Transações**")
        country_transactions = df['project_country'].value_counts().head(10)
        for i, (country, count) in enumerate(country_transactions.items(), 1):
            market_share = (count / len(df)) * 100
            st.write(f"{i}. **{country}**: {count:,} ({market_share:.1f}%)")
    
    with col2:
        st.markdown("**Top Países por Volume**")
        country_volumes = df.groupby('project_country')['credits_quantity'].sum().sort_values(ascending=False).head(10)
        total_volume = df['credits_quantity'].sum()
        for i, (country, volume) in enumerate(country_volumes.items(), 1):
            market_share = (volume / total_volume) * 100
            st.write(f"{i}. **{country}**: {volume:,.0f} tCO₂ ({market_share:.1f}%)")


def _render_executive_dashboard(df):
    """Render executive-level dashboard."""
    
    st.markdown("#### Dashboard Executivo")
    
    # Key Findings in two columns
    col1, col2 = st.columns(2)
    
    total_volume = df['credits_quantity'].sum()
    
    with col1:
        st.markdown("**Categorias de Maior Volume**")
        top_categories = df.groupby('project_category')['credits_quantity'].sum().nlargest(5)
        for i, (category, volume) in enumerate(top_categories.items(), 1):
            market_share = (volume / total_volume) * 100
            st.write(f"{i}. **{category}** - {market_share:.1f}%")
    
    with col2:
        st.markdown("**Principais Mercados Geográficos**")
        top_countries = df.groupby('project_country')['credits_quantity'].sum().nlargest(5)
        for i, (country, volume) in enumerate(top_countries.items(), 1):
            market_share = (volume / total_volume) * 100
            st.write(f"{i}. **{country}** - {market_share:.1f}%")


def _render_strategic_insights(df):
    """Render strategic insights and recommendations."""
    
    st.markdown("#### Insights Estratégicos")
    
    # Temporal Analysis
    monthly_volumes = df.groupby(df['transaction_date'].dt.month)['credits_quantity'].sum()
    peak_month = int(monthly_volumes.idxmax())
    lowest_month = int(monthly_volumes.idxmin())
    
    month_names = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                   7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pico de Atividade", f"{month_names[peak_month]} ({monthly_volumes[peak_month]:,.0f} tCO₂)")
    with col2:
        st.metric("Menor Atividade", f"{month_names[lowest_month]} ({monthly_volumes[lowest_month]:,.0f} tCO₂)")
    
    # Key insights
    st.markdown("**Principais Insights Estratégicos:**")
    
    insights = [
        "**Dominância REDD+**: Lidera em volume e transações, indicando mercado maduro",
        "**Padrões Sazonais**: Timing de transações apresenta variações significativas",
        "**Concentração Geográfica**: Participação diversificada mas com países dominantes",
        "**Distribuição de Volume**: Mercado atende desde pequenos até mega-projetos",
        "**Base Analítica**: Dados históricos fornecem base sólida para decisões estratégicas"
    ]
    
    for insight in insights:
        st.write(f"• {insight}")


if __name__ == "__main__":
    main() 