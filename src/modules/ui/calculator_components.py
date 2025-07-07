"""
Reusable UI Components for Calculator Interfaces

This module provides consistent UI components for all calculator interfaces
to ensure uniform UX/UI across the Carbon Credits Analytics Platform.
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Callable


def render_calculator_header(title: str, description: str, icon: str = "🔢") -> None:
    
    st.markdown(f"""
    <div class="calculator-header">
        <h3>{icon} {title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_input_section(title: str, content: Callable) -> None:
    
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
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    content()
    st.markdown('</div>', unsafe_allow_html=True)


def render_section_title(title: str, level: int = 3) -> None:
    
    header_tag = f"h{level}"
    if level == 3:
        st.markdown(f"### {title}")
    else:
        st.markdown(f"#### {title}")


def render_enhanced_metrics(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    
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
    
    st.info(f"**{icon} {title}**\n\n{content}")


def render_data_table(
    data, 
    title: str, 
    columns_to_show: Optional[List[str]] = None,
    format_dict: Optional[Dict[str, str]] = None
) -> None:
    
    st.markdown(f"#### {title}")
    
    if format_dict:
        styled_data = data.style.format(format_dict)
        st.dataframe(styled_data, use_container_width=True, hide_index=True)
    else:
        st.dataframe(data, use_container_width=True, hide_index=True) 