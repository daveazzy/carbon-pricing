"""
Reusable UI Components for Calculator Interfaces

This module provides consistent UI components for all calculator interfaces
to ensure uniform UX/UI across the Carbon Credits Analytics Platform.
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Callable


def render_calculator_header(title: str, description: str, icon: str = "🔢") -> None:
    """
    Render a consistent calculator header with improved styling.
    
    Args:
        title: Calculator title
        description: Calculator description
        icon: Icon emoji (default: 🔢)
    """
    st.markdown(f"""
    <div class="calculator-header">
        <h3>{icon} {title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_input_section(title: str, content: Callable) -> None:
    """
    Render a consistent input section with proper styling.
    
    Args:
        title: Section title
        content: Function that renders the input content
    """
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.markdown(f"**{title}**")
    content()
    st.markdown('</div>', unsafe_allow_html=True)


def render_primary_button(
    text: str, 
    key: str, 
    help_text: Optional[str] = None,
    center: bool = True
) -> bool:
    """
    Render a consistently styled primary action button.
    
    Args:
        text: Button text
        key: Unique button key
        help_text: Optional help text
        center: Whether to center the button
        
    Returns:
        bool: Whether button was clicked
    """
    if center:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            return st.button(
                text,
                type="primary",
                key=key,
                help=help_text,
                use_container_width=True
            )
    else:
        return st.button(
            text,
            type="primary", 
            key=key,
            help=help_text
        )


def render_results_container(content: Callable) -> None:
    """
    Render a consistent results container with proper styling.
    
    Args:
        content: Function that renders the results content
    """
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    content()
    st.markdown('</div>', unsafe_allow_html=True)


def render_section_title(title: str, level: int = 3) -> None:
    """
    Render a consistent section title.
    
    Args:
        title: Section title
        level: Header level (3 or 4)
    """
    header_tag = f"h{level}"
    if level == 3:
        st.markdown(f"### {title}")
    else:
        st.markdown(f"#### {title}")


def render_enhanced_metrics(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """
    Render enhanced metrics with consistent styling.
    
    Args:
        metrics: List of metric dictionaries with keys: label, value, help, delta (optional)
        columns: Number of columns for layout
    """
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta'),
                help=metric.get('help')
            )


def render_status_alert(
    status: str, 
    title: str, 
    message: str, 
    details: Optional[str] = None
) -> None:
    """
    Render consistent status alerts with enhanced formatting.
    
    Args:
        status: Alert type ('success', 'info', 'warning', 'error')
        title: Alert title
        message: Main message
        details: Optional additional details
    """
    full_message = f"**{title}**\n\n{message}"
    if details:
        full_message += f"\n\n{details}"
    
    if status == 'success':
        st.success(full_message)
    elif status == 'info':
        st.info(full_message)
    elif status == 'warning':
        st.warning(full_message)
    elif status == 'error':
        st.error(full_message)
    else:
        st.write(full_message)


def render_comparison_columns(
    left_title: str,
    left_content: List[str],
    right_title: str, 
    right_content: List[str]
) -> None:
    """
    Render comparison columns with consistent formatting.
    
    Args:
        left_title: Title for left column
        left_content: List of content items for left column
        right_title: Title for right column  
        right_content: List of content items for right column
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{left_title}:**")
        for item in left_content:
            st.markdown(f"• {item}")
    
    with col2:
        st.markdown(f"**{right_title}:**")
        for item in right_content:
            st.markdown(f"• {item}")


def render_info_box(title: str, content: str, icon: str = "💡") -> None:
    """
    Render an informational box with consistent styling.
    
    Args:
        title: Box title
        content: Box content
        icon: Icon emoji
    """
    st.info(f"**{icon} {title}**\n\n{content}")


def render_data_table(
    data, 
    title: str, 
    height: Optional[int] = None,
    format_dict: Optional[Dict[str, str]] = None
) -> None:
    """
    Render a data table with consistent styling.
    
    Args:
        data: DataFrame or data to display
        title: Table title
        height: Optional table height
        format_dict: Optional formatting dictionary
    """
    render_section_title(title, level=4)
    
    if format_dict:
        styled_data = data.style.format(format_dict)
        if 'activity_index' in data.columns:
            styled_data = styled_data.background_gradient(
                subset=['activity_index'], 
                cmap='RdYlGn'
            )
        st.dataframe(
            styled_data,
            use_container_width=True,
            hide_index=True,
            height=height
        )
    else:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
            height=height
        )


def render_legend(legend_items: List[Dict[str, str]], columns: int = 2) -> None:
    """
    Render a legend with consistent styling.
    
    Args:
        legend_items: List of legend items with 'icon', 'label', 'description'
        columns: Number of columns for layout
    """
    render_section_title("📝 Guia de Interpretação", level=4)
    
    cols = st.columns(columns)
    
    for i, group in enumerate([legend_items[i:i+len(legend_items)//columns] for i in range(0, len(legend_items), len(legend_items)//columns)]):
        with cols[i]:
            for item in group:
                st.markdown(f"{item['icon']} **{item['label']}**: {item['description']}")


def render_loading_with_message(message: str) -> Any:
    """
    Render a loading spinner with custom message.
    
    Args:
        message: Loading message
        
    Returns:
        Streamlit spinner context manager
    """
    return st.spinner(f"🔄 {message}")


def render_category_selector(
    available_categories: List[str],
    label: str = "Categoria para Análise",
    help_text: str = "Escolha uma categoria para análise detalhada",
    key: str = "category_selector",
    default_index: int = 0
) -> str:
    """
    Render a consistent category selector.
    
    Args:
        available_categories: List of available categories
        label: Selector label
        help_text: Help text
        key: Unique key
        default_index: Default selection index
        
    Returns:
        Selected category
    """
    return st.selectbox(
        label,
        available_categories,
        index=default_index,
        help=help_text,
        key=key
    )


def render_radio_selection(
    label: str,
    options: List[str],
    help_text: str = "",
    key: str = "radio_selection",
    horizontal: bool = True
) -> str:
    """
    Render a consistent radio button selection.
    
    Args:
        label: Selection label
        options: List of options
        help_text: Help text
        key: Unique key
        horizontal: Whether to display horizontally
        
    Returns:
        Selected option
    """
    return st.radio(
        f"**{label}**",
        options,
        help=help_text,
        key=key,
        horizontal=horizontal
    )


def render_enhanced_alert(
    alert_type: str,
    title: str,
    main_text: str,
    bullet_points: List[str],
    action_text: Optional[str] = None
) -> None:
    """
    Render an enhanced alert with structured content.
    
    Args:
        alert_type: Type of alert ('success', 'info', 'warning', 'error')
        title: Alert title with emoji
        main_text: Main descriptive text
        bullet_points: List of bullet points
        action_text: Optional action recommendation
    """
    content = f"**{title}**\n\n{main_text}\n\n"
    
    for point in bullet_points:
        content += f"• {point}\n"
    
    if action_text:
        content += f"\n**💡 {action_text}**"
    
    if alert_type == 'success':
        st.success(content)
    elif alert_type == 'info':
        st.info(content)
    elif alert_type == 'warning':
        st.warning(content)
    elif alert_type == 'error':
        st.error(content)


# Utility functions for common calculator patterns
def get_risk_color(risk_level: str) -> str:
    """Get color emoji for risk level."""
    risk_colors = {
        'LOW': '🟢',
        'MEDIUM': '🟡', 
        'HIGH': '🔴',
        'OPTIMAL': '🟢',
        'GOOD': '🟡',
        'FAIR': '🟠',
        'POOR': '🔴'
    }
    return risk_colors.get(risk_level.upper(), '⚪')


def format_large_number(number: float, unit: str = "") -> str:
    """Format large numbers with appropriate units."""
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M {unit}".strip()
    elif number >= 1_000:
        return f"{number/1_000:.1f}K {unit}".strip()
    else:
        return f"{number:,.0f} {unit}".strip()


def create_metric_dict(label: str, value: str, help_text: str, delta: Optional[str] = None) -> Dict[str, Any]:
    """Create a metric dictionary for enhanced metrics rendering."""
    return {
        'label': label,
        'value': value,
        'help': help_text,
        'delta': delta
    } 