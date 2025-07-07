"""
CSS Styles for Carbon Credits Analytics Platform

Professional Dark Theme UI/UX with DRAMATICALLY IMPROVED buttons, text organization,
and consistent calculator interfaces. Dark theme with excellent visibility and contrast.
"""

import streamlit as st

def load_custom_css():
    """Load custom CSS styles with dramatically improved dark theme UX/UI design."""
    
    st.markdown("""
    <style>
    
    /* ============================================================================
       DARK THEME BASE & TYPOGRAPHY SYSTEM
    ============================================================================ */
    
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Dark theme global settings - optimized for professional presentations */
    .stApp, .main, .element-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-weight: 500 !important;  /* Medium weight for better readability */
        font-size: 16px !important;   /* Larger base size for projector visibility */
        line-height: 1.6 !important;
        color: #e8e9ea !important;    /* Light text on dark backgrounds */
        background-color: #1a1a1a !important;  /* Rich dark background */
    }
    
    /* Force dark theme on Streamlit container */
    .main .block-container {
        background-color: #1a1a1a !important;
        padding-top: 1rem !important;
        max-width: 1200px !important;
    }
    
    /* All text elements - force readable light colors on dark backgrounds */
    p, span, div, label, .stMarkdown, .stText {
        color: #e8e9ea !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        background-color: transparent !important;
    }
    
    /* IMPROVED TYPOGRAPHY HIERARCHY - DARK THEME */
    h1, .main-title {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 32px !important;
        line-height: 1.2 !important;
        margin-bottom: 16px !important;
        background-color: transparent !important;
    }
    
    h2, .page-title {
        color: #f8f9fa !important;
        font-weight: 600 !important;
        font-size: 28px !important;
        line-height: 1.3 !important;
        margin-bottom: 16px !important;
        background-color: transparent !important;
    }
    
    h3, .section-title {
        color: #f8f9fa !important;
        font-weight: 600 !important;
        font-size: 22px !important;
        line-height: 1.4 !important;
        margin-bottom: 12px !important;
        background-color: transparent !important;
        border-bottom: 2px solid #444444 !important;
        padding-bottom: 8px !important;
    }
    
    h4, .subsection-title {
        color: #e8e9ea !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        line-height: 1.4 !important;
        margin-bottom: 10px !important;
        background-color: transparent !important;
    }
    
    /* Subtitle and descriptions - dark theme */
    .subtitle, .page-description {
        color: #c7c8ca !important;
        font-weight: 500 !important;
        font-size: 17px !important;
        line-height: 1.5 !important;
        margin-bottom: 24px !important;
        background-color: transparent !important;
    }
    
    /* ============================================================================
       PROFESSIONAL DARK HEADER DESIGN
    ============================================================================ */
    
    .professional-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #ecf0f1;
        padding: 24px 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        border-bottom: 3px solid #3498db;
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex: 1;
    }
    
    .logo-placeholder {
        font-size: 48px;
        line-height: 1;
        color: #3498db;
        opacity: 1;
    }
    
    .title-section h1 {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    
    .title-section p {
        color: #bdc3c7 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        margin: 4px 0 0 0 !important;
        line-height: 1.4 !important;
    }
    
    .metrics-section {
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .metric-pill {
        background: rgba(52, 73, 94, 0.8);
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        min-width: 100px;
        border: 2px solid #444444;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        display: block;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #95a5a6 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    
    .metric-value {
        display: block;
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #ecf0f1 !important;
        line-height: 1.2 !important;
    }
    
    /* ============================================================================
       DARK NAVIGATION SYSTEM
    ============================================================================ */
    
    .navigation-container {
        background: #2c3e50;
        border-bottom: 2px solid #34495e;
        margin: -1rem -1rem 2rem -1rem;
        padding: 0;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .nav-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        background: #2c3e50;
    }
    
    .nav-spacer {
        height: 0;
        margin: 0;
    }
    
    /* DRAMATICALLY IMPROVED NAVIGATION BUTTONS - DARK THEME */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
        color: #ecf0f1 !important;
        border: 2px solid #4a5f7a !important;
        border-radius: 8px !important;
        padding: 16px 8px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        margin: 4px 2px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4a5f7a 0%, #3d566e 100%) !important;
        color: #3498db !important;
        border-color: #3498db !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(52, 152, 219, 0.25) !important;
    }
    
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border: 2px solid #2980b9 !important;
        box-shadow: 0 4px 16px rgba(52, 152, 219, 0.3) !important;
    }
    
    .stButton > button[data-baseweb="button"][kind="primary"]:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* ============================================================================
       DARK CONTENT LAYOUT & SECTIONS
    ============================================================================ */
    
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
        background-color: #1a1a1a !important;
    }
    
    .page-header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        border-radius: 12px;
        border: 1px solid #444444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .content-section {
        background: #2c3e50 !important;
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        border: 1px solid #444444;
    }
    
    /* CALCULATOR SPECIFIC SECTIONS - DARK THEME */
    .calculator-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 2px solid #4a5f7a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    .calculator-header {
        background: #34495e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .input-group {
        background: #34495e;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #4a5f7a;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    
    .results-container {
        background: linear-gradient(135deg, #1e3a52 0%, #2c3e50 100%);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        border: 2px solid #3498db;
        box-shadow: 0 3px 15px rgba(52, 152, 219, 0.2);
    }
    
    .insights-sidebar {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #4a5f7a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .insights-sidebar h3 {
        color: #f8f9fa !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        background-color: transparent !important;
    }
    
    .insights-sidebar strong {
        color: #e8e9ea !important;
        font-weight: 600 !important;
    }
    
    /* ============================================================================
       DARK THEME FORM ELEMENTS & INPUTS - MINIMALIST & READABLE
    ============================================================================ */
    
    /* FORM LABELS - SIMPLE */
    .stSelectbox label, .stMultiselect label, .stSlider label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        margin-bottom: 8px !important;
        background-color: transparent !important;
        display: block !important;
        padding: 4px 0 !important;
    }
    
    /* SELECTBOX - MINIMALIST DESIGN */
    .stSelectbox > div > div {
        background: #404040 !important;
        border: 1px solid #666666 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* SELECTBOX BUTTON (SELECTED VALUE) */
    .stSelectbox > div > div > div[role="button"] {
        background: #404040 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        padding: 12px 16px !important;
        border: none !important;
        border-radius: 4px !important;
    }
    
    /* DROPDOWN ARROW */
    .stSelectbox > div > div svg {
        color: #cccccc !important;
        fill: #cccccc !important;
    }
    
    /* DROPDOWN OPTIONS LIST */
    .stSelectbox > div > div > div[role="listbox"] {
        background: #303030 !important;
        border: 1px solid #666666 !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
        margin-top: 2px !important;
    }
    
    /* INDIVIDUAL OPTIONS */
    .stSelectbox > div > div > div[role="option"] {
        background: #303030 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        padding: 12px 16px !important;
        border: none !important;
        border-bottom: 1px solid #555555 !important;
    }
    
    /* OPTION HOVER - SIMPLE GRAY */
    .stSelectbox > div > div > div[role="option"]:hover {
        background: #555555 !important;
        color: #ffffff !important;
    }
    
    /* MULTISELECT - SIMPLE */
    .stMultiselect > div > div {
        background: #404040 !important;
        border: 1px solid #666666 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
    }
    
    /* MULTISELECT SELECTED ITEMS */
    .stMultiselect > div > div > div {
        background: #666666 !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
        margin: 2px !important;
        border: none !important;
    }
    
    /* MULTISELECT DROPDOWN */
    .stMultiselect > div > div > div[role="listbox"] {
        background: #303030 !important;
        border: 1px solid #666666 !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
    }
    
    /* MULTISELECT OPTIONS */
    .stMultiselect > div > div > div[role="option"] {
        background: #303030 !important;
        color: #ffffff !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        border-bottom: 1px solid #555555 !important;
    }
    
    /* MULTISELECT OPTION HOVER */
    .stMultiselect > div > div > div[role="option"]:hover {
        background: #555555 !important;
        color: #ffffff !important;
    }
    
    /* RADIO BUTTONS - MINIMALIST */
    .stRadio > div > label {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        padding: 8px 0 !important;
    }
    
    /* RADIO BUTTON CIRCLE */
    .stRadio > div > label > div:first-child {
        border: 2px solid #666666 !important;
        background: #404040 !important;
    }
    
    /* SELECTED RADIO BUTTON */
    .stRadio > div > label > div:first-child[data-checked="true"] {
        border: 2px solid #cccccc !important;
        background: #cccccc !important;
    }
    
    /* NUMBER INPUT */
    .stNumberInput > div > div > input {
        background: #404040 !important;
        border: 1px solid #666666 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
    }
    
    /* FOCUS STATES - SIMPLE GRAY */
    .stSelectbox > div > div:focus-within, 
    .stMultiselect > div > div:focus-within, 
    .stNumberInput > div > div:focus-within {
        border-color: #888888 !important;
        box-shadow: 0 0 0 2px rgba(136, 136, 136, 0.3) !important;
    }
    
    /* STREAMLIT INTERNAL COMPONENTS - SIMPLE */
    [data-baseweb="select"] {
        background: #404040 !important;
        color: #ffffff !important;
        border: 1px solid #666666 !important;
    }
    
    [data-baseweb="select"] > div {
        background: #404040 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="popover"] {
        background: #303030 !important;
        border: 1px solid #666666 !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
    }
    
    [data-baseweb="menu"] {
        background: #303030 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="menu-item"] {
        background: #303030 !important;
        color: #ffffff !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }
    
    [data-baseweb="menu-item"]:hover {
        background: #555555 !important;
        color: #ffffff !important;
    }
    
    /* PLACEHOLDER TEXT */
    input::placeholder {
        color: #999999 !important;
        opacity: 1 !important;
    }
    
    /* FORCE TEXT VISIBILITY */
    .stSelectbox *, .stMultiselect *, .stNumberInput *, .stTextInput * {
        color: #ffffff !important;
    }
    
    /* FINAL OVERRIDE FOR TEXT */
    .stSelectbox div[role="button"] span,
    .stSelectbox div[role="option"] span,
    .stMultiselect div span {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }
    
    /* REMOVE ALL BLUE COLORS */
    .stSelectbox div[role="button"]:focus,
    .stSelectbox div[role="option"]:focus,
    .stMultiselect div:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* ============================================================================
       DRAMATICALLY IMPROVED BUTTONS - DARK THEME WITH PERFECT VISIBILITY
    ============================================================================ */
    
    /* PRIMARY ACTION BUTTONS - PREMIUM DARK DESIGN */
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 18px 36px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        min-height: 55px !important;
        width: 100% !important;
        margin: 8px 0 !important;
    }
    
    .stButton > button[data-baseweb="button"][kind="primary"]:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 28px rgba(52, 152, 219, 0.5) !important;
    }
    
    .stButton > button[data-baseweb="button"][kind="primary"]:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* SECONDARY BUTTONS - CLEAN DARK DESIGN */
    .stButton > button[data-baseweb="button"][kind="secondary"] {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
        color: #ecf0f1 !important;
        border: 2px solid #4a5f7a !important;
        border-radius: 12px !important;
        padding: 18px 36px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        min-height: 55px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        width: 100% !important;
        margin: 8px 0 !important;
    }
    
    .stButton > button[data-baseweb="button"][kind="secondary"]:hover {
        background: linear-gradient(135deg, #4a5f7a 0%, #3d566e 100%) !important;
        border-color: #3498db !important;
        color: #3498db !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.2) !important;
    }
    
    /* DANGER/WARNING BUTTONS - DARK THEME */
    .stButton > button.danger-button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 18px 36px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4) !important;
        min-height: 55px !important;
        width: 100% !important;
        margin: 8px 0 !important;
    }
    
    /* SUCCESS BUTTONS - DARK THEME */
    .stButton > button.success-button {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 18px 36px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4) !important;
        min-height: 55px !important;
        width: 100% !important;
        margin: 8px 0 !important;
    }
    
    /* BUTTON DISTRIBUTION AND SPACING */
    .stButton {
        margin: 12px 0 !important;
    }
    
    /* Ensure buttons are properly spaced in columns */
    [data-testid="column"] .stButton {
        margin: 8px 4px !important;
    }
    
    /* ============================================================================
       DRAMATICALLY IMPROVED METRIC CARDS - DARK THEME
    ============================================================================ */
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        border: 2px solid #4a5f7a !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    [data-testid="metric-container"]:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3498db, #2ecc71, #f39c12, #e74c3c);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
        border-color: #3498db !important;
        background: linear-gradient(135deg, #34495e 0%, #3d566e 100%) !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #e8e9ea !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        background-color: transparent !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 28px !important;
        background-color: transparent !important;
    }
    
    /* ============================================================================
       IMPROVED EXPANDABLE SECTIONS - DARK THEME
    ============================================================================ */
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        border: 2px solid #4a5f7a !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        color: #f8f9fa !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #34495e 0%, #3d566e 100%) !important;
        border-color: #3498db !important;
    }
    
    .streamlit-expanderContent {
        background: #34495e !important;
        border: 2px solid #4a5f7a !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 2rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* ============================================================================
       IMPROVED ALERT BOXES & NOTIFICATIONS - DARK THEME
    ============================================================================ */
    
    /* Success alerts - dark theme */
    .stAlert[data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #1e4d3d 0%, #27ae60 20%, #1e4d3d 100%) !important;
        border: 2px solid #27ae60 !important;
        border-left: 6px solid #27ae60 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: #a8e6cf !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 3px 15px rgba(39, 174, 96, 0.25) !important;
    }
    
    /* Info alerts - dark theme */
    .stAlert[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #1e3a52 0%, #3498db 20%, #1e3a52 100%) !important;
        border: 2px solid #3498db !important;
        border-left: 6px solid #3498db !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: #a8d4f0 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 3px 15px rgba(52, 152, 219, 0.25) !important;
    }
    
    /* Warning alerts - dark theme */
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #5d4037 0%, #f39c12 20%, #5d4037 100%) !important;
        border: 2px solid #f39c12 !important;
        border-left: 6px solid #f39c12 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: #ffcc7a !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 3px 15px rgba(243, 156, 18, 0.25) !important;
    }
    
    /* Error alerts - dark theme */
    .stAlert[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #5d2e2e 0%, #e74c3c 20%, #5d2e2e 100%) !important;
        border: 2px solid #e74c3c !important;
        border-left: 6px solid #e74c3c !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: #ffb3b3 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 3px 15px rgba(231, 76, 60, 0.25) !important;
    }
    
    /* ============================================================================
       IMPROVED TABLES & DATA FRAMES - DARK THEME
    ============================================================================ */
    
    .stDataFrame, .dataframe {
        border: 2px solid #4a5f7a !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        background: #2c3e50 !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
        color: #f8f9fa !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-bottom: 2px solid #4a5f7a !important;
        padding: 16px 12px !important;
        text-align: center !important;
    }
    
    .dataframe td {
        color: #e8e9ea !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        padding: 12px !important;
        border-bottom: 1px solid #4a5f7a !important;
        background: #34495e !important;
        text-align: center !important;
    }
    
    .dataframe tr:hover td {
        background: #3d566e !important;
        color: #ffffff !important;
    }
    
    /* ============================================================================
       TABS STYLING - DARK THEME
    ============================================================================ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: #2c3e50 !important;
        padding: 8px !important;
        border-radius: 12px !important;
        border: 2px solid #4a5f7a !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #34495e !important;
        border-radius: 8px !important;
        color: #e8e9ea !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        border: 1px solid #4a5f7a !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #3d566e !important;
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border: 1px solid #2980b9 !important;
        box-shadow: 0 2px 12px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* ============================================================================
       RESPONSIVE DESIGN - DARK THEME
    ============================================================================ */
    
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .metrics-section {
            justify-content: center;
        }
        
        .metric-pill {
            min-width: 80px;
            font-size: 12px;
        }
        
        .main-content {
            padding: 0 0.5rem;
        }
        
        .content-section, .calculator-section {
            padding: 1.5rem;
        }
        
        .stButton > button {
            padding: 16px 20px !important;
            font-size: 15px !important;
            min-height: 50px !important;
        }
    }
    
    /* ============================================================================
       UTILITY CLASSES - DARK THEME
    ============================================================================ */
    
    .text-center { text-align: center !important; }
    .text-left { text-align: left !important; }
    .text-right { text-align: right !important; }
    
    .font-weight-bold { font-weight: 700 !important; }
    .font-weight-medium { font-weight: 600 !important; }
    .font-weight-normal { font-weight: 500 !important; }
    
    .mb-1 { margin-bottom: 0.5rem !important; }
    .mb-2 { margin-bottom: 1rem !important; }
    .mb-3 { margin-bottom: 1.5rem !important; }
    .mb-4 { margin-bottom: 2rem !important; }
    
    .p-2 { padding: 1rem !important; }
    .p-3 { padding: 1.5rem !important; }
    .p-4 { padding: 2rem !important; }
    
    .bg-dark { background-color: #2c3e50 !important; }
    .bg-darker { background-color: #1a1a1a !important; }
    
    .border-radius-lg { border-radius: 12px !important; }
    .border-radius-xl { border-radius: 16px !important; }
    
    .shadow-sm { box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important; }
    .shadow-md { box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important; }
    .shadow-lg { box-shadow: 0 8px 24px rgba(0,0,0,0.25) !important; }
    
    /* Force dark theme on all Streamlit elements */
    .stApp {
        background-color: #1a1a1a !important;
    }
    
    /* Hide Streamlit default elements for cleaner dark look */
    #MainMenu { visibility: hidden; }
    .stDeployButton { display: none; }
    footer { visibility: hidden; }
    .stApp > header { display: none; }
    
    /* Ensure no light mode elements slip through */
    * {
        scrollbar-color: #4a5f7a #2c3e50 !important;
    }
    
    ::-webkit-scrollbar {
        background-color: #2c3e50 !important;
        width: 8px !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #4a5f7a !important;
        border-radius: 6px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #5a6f8a !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


def create_metric_card(title: str, content: str, card_type: str = "metric") -> str:
    """
    Create a styled metric card with consistent formatting and proper contrast.
    
    Args:
        title: Card title
        content: Card content (HTML allowed)
        card_type: Type of card ('metric', 'success', 'info', 'primary')
    
    Returns:
        HTML string for the styled card
    """
    
    card_class_map = {
        "metric": "metric-card",
        "success": "success-card", 
        "info": "info-card",
        "primary": "primary-card"
    }
    
    card_class = card_class_map.get(card_type, "metric-card")
    
    return f"""
    <div class="{card_class}">
        <h4 style="margin-bottom: 1rem;">{title}</h4>
        <div>{content}</div>
    </div>
    """


def apply_main_title_style(title: str) -> None:
    """
    Apply main title styling to a title with proper contrast.
    
    Args:
        title: Title text
    """
    st.markdown(f'<h1 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">{title}</h1>', unsafe_allow_html=True) 