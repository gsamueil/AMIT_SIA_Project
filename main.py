"""
SIA Forecasting System - FINAL VERSION
Professional Icons + Proper Eye of Horus Image

Author: AMIT ACM Project  
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import logging
import io
import base64

from config import (
    PAGE_CONFIG, COLORS, CUSTOM_CSS,
    DATA_CONFIG, FORECAST_CONFIG, 
    normalize_system_name
)
from icons import TAB_ICONS
from preprocessing import DataProcessor
from model import ForecastEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model-specific colors for accuracy charts  
def get_model_color(model_name, is_best=False):
    """RED=benchmark, GOLD=best, BEIGE=rest"""
    m = str(model_name).lower()
    if 'benchmark' in m or 'ma_benchmark' in m:
        return '#E74C3C'  # RED - Benchmark
    elif is_best:
        return '#FFD700'  # GOLD - Best
    else:
        return COLORS['beige']



# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'validation' not in st.session_state:
    st.session_state.validation = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_base64_image(image_path):
    """Convert image to base64 for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def format_number(num, decimals=0):
    """Format numbers with commas"""
    if pd.isna(num):
        return "0"
    if decimals == 0:
        return f"{int(num):,}"
    return f"{num:,.{decimals}f}"


def apply_filters(df):
    """Apply all active filters from session state"""
    if df is None or len(df) == 0:
        return df
    
    filtered = df.copy()
    
    # Year filter
    if st.session_state.get('filter_years') and len(st.session_state.filter_years) > 0:
        if 'year' in filtered.columns:
            filtered = filtered[filtered['year'].isin(st.session_state.filter_years)]
    
    # Period filter
    if st.session_state.get('filter_period_start') and st.session_state.get('filter_period_end'):
        if 'period_month' in filtered.columns:
            # Convert to comparable format
            filtered = filtered[
                (filtered['period_month'].astype(str) >= st.session_state.filter_period_start) &
                (filtered['period_month'].astype(str) <= st.session_state.filter_period_end)
            ]
    
    # Country filter
    if st.session_state.get('filter_countries') and len(st.session_state.filter_countries) > 0:
        if 'All Countries' not in st.session_state.filter_countries:
            if 'country' in filtered.columns:
                filtered = filtered[filtered['country'].isin(st.session_state.filter_countries)]
    
    # System filter
    if st.session_state.get('filter_systems') and len(st.session_state.filter_systems) > 0:
        if 'All Systems' not in st.session_state.filter_systems:
            if 'system' in filtered.columns:
                filtered = filtered[filtered['system'].isin(st.session_state.filter_systems)]
    
    # Factory filter
    if st.session_state.get('filter_factories') and len(st.session_state.filter_factories) > 0:
        if 'All Factories' not in st.session_state.filter_factories:
            if 'factory' in filtered.columns:
                filtered = filtered[filtered['factory'].isin(st.session_state.filter_factories)]
    
    return filtered


def create_chart(data, chart_type='bar', x=None, y=None, title='', color=COLORS['gold']):
    """Create Plotly chart"""
    
    if chart_type == 'bar':
        fig = px.bar(data, x=x, y=y, title=title)
        fig.update_traces(marker_color=color)
    elif chart_type == 'line':
        fig = px.line(data, x=x, y=y, title=title)
        fig.update_traces(line_color=color, line_width=3)
    elif chart_type == 'area':
        fig = px.area(data, x=x, y=y, title=title)
    
    fig.update_layout(
        plot_bgcolor=COLORS['navy_light'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white']),
        title_font=dict(color=COLORS['gold'], size=18),
        xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        hovermode='x unified'
    )
    
    return fig


def create_time_series_chart(data, x, y, title='Time Series', color=COLORS['gold']):
    """Create enhanced time series chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[x],
        y=data[y],
        mode='lines+markers',
        name=title,
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        fill='tonexty',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}'
    ))
    
    fig.update_layout(
        title=title,
        plot_bgcolor=COLORS['navy_light'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white']),
        title_font=dict(color=COLORS['gold'], size=18),
        xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_distribution_chart(data, column, title='Distribution'):
    """Create distribution histogram"""
    fig = px.histogram(data, x=column, title=title, nbins=50)
    fig.update_traces(marker_color=[
                    BENCHMARK_COLOR if 'benchmark' in str(m).lower() else 
                    ENSEMBLE_COLOR if 'ensemble' in str(m).lower() else
                    INTELLIGENT_COLOR if 'intelligent' in str(m).lower() else
                    COLORS['gold']
                    for m in df_plot_sorted['Method']
                ])
    fig.update_layout(
        plot_bgcolor=COLORS['navy_light'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white']),
        title_font=dict(color=COLORS['gold'], size=18),
        xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        height=400
    )
    return fig

def create_pie_chart(data, values, names, title='Distribution'):
    """Create professional pie chart"""
    fig = px.pie(data, values=values, names=names, title=title)
    fig.update_traces(
        marker=dict(colors=[COLORS['gold'], COLORS['gold_dark'], COLORS['beige'], 
                           COLORS['navy_light'], COLORS['gold_light']]),
        textposition='inside',
        textinfo='percent+label'
    )
    fig.update_layout(
        plot_bgcolor=COLORS['navy'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white']),
        title_font=dict(color=COLORS['gold'], size=18),
        height=400
    )
    return fig

def create_heatmap(data, title='Heatmap'):
    """Create correlation heatmap"""
    fig = px.imshow(data, 
                    title=title,
                    color_continuous_scale='YlOrRd',
                    aspect='auto')
    fig.update_layout(
        plot_bgcolor=COLORS['navy'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white']),
        title_font=dict(color=COLORS['gold'], size=18),
        height=500
    )
    return fig

def create_box_plot(data, x, y, title='Box Plot'):
    """Create box plot for distribution analysis"""
    fig = px.box(data, x=x, y=y, title=title)
    fig.update_traces(marker_color=[get_model_color(m) for m in df_plot_sorted['Method']])
    fig.update_layout(
        plot_bgcolor=COLORS['navy_light'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white']),
        title_font=dict(color=COLORS['gold'], size=18),
        xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
        height=400
    )
    return fig

def create_treemap(data, path, values, title='Hierarchy Treemap'):
    """Create treemap for hierarchical data"""
    fig = px.treemap(data, path=path, values=values, title=title)
    fig.update_traces(
        marker=dict(colorscale='YlOrRd'),
        textposition='middle center'
    )
    fig.update_layout(
        plot_bgcolor=COLORS['navy'],
        paper_bgcolor=COLORS['navy'],
        font=dict(color=COLORS['white'], size=14),
        title_font=dict(color=COLORS['gold'], size=18),
        height=500
    )
    return fig

# ==========================================
# PROFESSIONAL SVG ICONS
# ==========================================

ICONS = {
    'home': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>''',
    
    'database': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>''',
    
    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>''',
    
    'trending': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>''',
    
    'globe': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>''',
    
    'factory': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8l-7 5V8l-7 5V4a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2Z"/><path d="M17 18h1"/><path d="M12 18h1"/><path d="M7 18h1"/></svg>''',
    
    'settings': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>''',
    
    'grid': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>''',
    
    'package': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16.5 9.4 7.55 4.24"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" y1="22" x2="12" y2="12"/></svg>''',
    
    'calendar': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>''',
    
    'target': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>''',
    
    'file': '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>'''
}

# ==========================================
# SIDEBAR (PROFESSIONAL ICONS)
# ==========================================

def render_sidebar():
    """Sidebar with professional icons and conditional filters"""
    
    # Icon mapping
    tab_icons = {
        'Home': ICONS['home'],
        'Data Input': ICONS['database'],
        'Validation': ICONS['check'],
        'Executive': ICONS['trending'],
        'Country': ICONS['globe'],
        'Factory': ICONS['factory'],
        'System': ICONS['grid'],
        'Cell': ICONS['grid'],
        'Item': ICONS['package'],
        'Forecast': ICONS['calendar'],
        'Accuracy': ICONS['target'],
        'Reports': ICONS['file']
    }
    
    with st.sidebar:
        # Logo
        logo_path = Path('src/gui/assets/logo.png')
        if logo_path.exists():
            logo_b64 = get_base64_image(str(logo_path))
            if logo_b64:
                st.markdown(
                    f"""
                    <div style='text-align: center; margin-bottom: 25px; padding: 10px 0;'>
                        <img src='data:image/png;base64,{logo_b64}' 
                             style='width: 160px; max-width: 90%; height: auto;'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # ==================== CONDITIONAL FILTERS ====================
        dashboard_tabs = ['Executive', 'Country', 'Factory', 'System', 'Cell', 'Forecast']
        current_tab = st.session_state.get('selected_tab', 'Home')
        
        if st.session_state.get('processed_data') is not None and current_tab in dashboard_tabs:
            st.markdown("---")
            
            # Filter header with professional icon
            filter_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>'
            
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
                    <span style='color: {COLORS["gold"]};'>{filter_svg}</span>
                    <h3 style='color: {COLORS["gold"]}; margin: 0;'>Filters</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            df = st.session_state.processed_data
            
            # Year filter
            if 'year' in df.columns:
                available_years = sorted(df['year'].dropna().unique().tolist())
                if 'filter_years' not in st.session_state:
                    st.session_state.filter_years = available_years
                
                # Custom styled multiselect
                st.markdown(f"<p style='color: {COLORS['beige']}; margin-bottom: 5px; font-size: 14px;'>Year</p>", unsafe_allow_html=True)
                selected_years = st.multiselect(
                    "Year",
                    options=available_years,
                    default=st.session_state.filter_years,
                    key='filter_years_input',
                    label_visibility="collapsed"
                )
                if selected_years:
                    st.session_state.filter_years = selected_years
            
            # Period range (except Forecast)
            if current_tab != 'Forecast' and 'period_month' in df.columns:
                periods = sorted(df['period_month'].dropna().unique().astype(str).tolist())
                if len(periods) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'filter_period_start' not in st.session_state:
                            st.session_state.filter_period_start = periods[0]
                        st.markdown(f"<p style='color: {COLORS['beige']}; margin-bottom: 5px; font-size: 14px;'>From</p>", unsafe_allow_html=True)
                        start = st.selectbox(
                            "From",
                            options=periods,
                            index=periods.index(st.session_state.filter_period_start) if st.session_state.filter_period_start in periods else 0,
                            label_visibility="collapsed"
                        )
                        st.session_state.filter_period_start = start
                    
                    with col2:
                        if 'filter_period_end' not in st.session_state:
                            st.session_state.filter_period_end = periods[-1]
                        st.markdown(f"<p style='color: {COLORS['beige']}; margin-bottom: 5px; font-size: 14px;'>To</p>", unsafe_allow_html=True)
                        end = st.selectbox(
                            "To",
                            options=periods,
                            index=periods.index(st.session_state.filter_period_end) if st.session_state.filter_period_end in periods else len(periods)-1,
                            label_visibility="collapsed"
                        )
                        st.session_state.filter_period_end = end
            
            # Country filter (all dashboard tabs)
            if 'country' in df.columns:
                countries = ['All Countries'] + sorted(df['country'].dropna().unique().tolist())
                if 'filter_countries' not in st.session_state:
                    st.session_state.filter_countries = ['All Countries']
                
                st.markdown(f"<p style='color: {COLORS['beige']}; margin-bottom: 5px; font-size: 14px;'>Countries</p>", unsafe_allow_html=True)
                selected_countries = st.multiselect(
                    "Countries",
                    options=countries,
                    default=st.session_state.filter_countries,
                    label_visibility="collapsed"
                )
                if selected_countries:
                    st.session_state.filter_countries = selected_countries
            
            # System filter (all dashboard tabs)
            if 'system' in df.columns:
                systems = ['All Systems'] + sorted(df['system'].dropna().unique().tolist())
                if 'filter_systems' not in st.session_state:
                    st.session_state.filter_systems = ['All Systems']
                
                st.markdown(f"<p style='color: {COLORS['beige']}; margin-bottom: 5px; font-size: 14px;'>Systems</p>", unsafe_allow_html=True)
                selected_systems = st.multiselect(
                    "Systems",
                    options=systems,
                    default=st.session_state.filter_systems,
                    label_visibility="collapsed"
                )
                if selected_systems:
                    st.session_state.filter_systems = selected_systems
            
            # Factory filter (except Forecast)
            if current_tab != 'Forecast' and 'factory' in df.columns:
                factories = ['All Factories'] + sorted(df['factory'].dropna().unique().tolist())
                if 'filter_factories' not in st.session_state:
                    st.session_state.filter_factories = ['All Factories']
                
                st.markdown(f"<p style='color: {COLORS['beige']}; margin-bottom: 5px; font-size: 14px;'>Factories</p>", unsafe_allow_html=True)
                selected_factories = st.multiselect(
                    "Factories",
                    options=factories,
                    default=st.session_state.filter_factories,
                    label_visibility="collapsed"
                )
                if selected_factories:
                    st.session_state.filter_factories = selected_factories
            
            # Reset button with professional icon
            reset_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>'
            
            col1, col2 = st.columns([0.15, 0.85])
            with col1:
                st.markdown(f"<div style='padding-top: 8px; color: {COLORS['gold']};'>{reset_svg}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("Reset All Filters", use_container_width=True):
                    if 'year' in df.columns:
                        st.session_state.filter_years = sorted(df['year'].dropna().unique().tolist())
                    st.session_state.filter_countries = ['All Countries']
                    st.session_state.filter_systems = ['All Systems']
                    st.session_state.filter_factories = ['All Factories']
                    if 'period_month' in df.columns:
                        periods = sorted(df['period_month'].dropna().unique().astype(str).tolist())
                        st.session_state.filter_period_start = periods[0]
                        st.session_state.filter_period_end = periods[-1]
                    st.rerun()
            
            st.markdown("---")
        
        # ==================== TAB NAVIGATION ====================
        if 'selected_tab' not in st.session_state:
            st.session_state.selected_tab = 'Home'
        
        tabs = [
            'Home', 'Data Input', 'Validation', 'Executive',
            'Country', 'Factory', 'System', 'Cell',
            'Item', 'Forecast', 'Accuracy', 'Reports'
        ]
        
        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
        
        for tab in tabs:
            is_selected = st.session_state.selected_tab == tab
            icon_svg = tab_icons.get(tab, '')
            
            # Use columns for icon + button
            col1, col2 = st.columns([0.15, 0.85])
            
            with col1:
                # Icon (white always)
                st.markdown(
                    f"<div style='padding-top: 8px; color: {COLORS['white']};'>{icon_svg}</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                # Button
                if st.button(
                    tab,
                    key=f'nav_{tab}',
                    use_container_width=True,
                    type='primary' if is_selected else 'secondary'
                ):
                    st.session_state.selected_tab = tab
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # SIMPLE FIX: Just add a bit more space here
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        
        # Footer - Aligned with button text (in the 85% column area)
        col1, col2 = st.columns([0.15, 0.85])
        
        with col1:
            # Empty column for icon alignment
            st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(
                f"""
                <div style='text-align: center; font-size: 15px; color: {COLORS['beige']}; 
                            border-top: 1px solid {COLORS['gold_dark']}; padding-top: 15px; padding-left: 0;'>
                    © ACROW 2025<br/>Powered by AI
                </div>
                """,
                unsafe_allow_html=True
            )
        # Custom CSS for buttons
        st.markdown(
            f"""
            <style>
            /* Primary (selected) buttons */
            button[kind="primary"] {{
                background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']}) !important;
                border: 2px solid {COLORS['gold']} !important;
                color: {COLORS['white']} !important;
                font-weight: bold !important;
                border-radius: 10px !important;
            }}
            /* Secondary (unselected) buttons */
            button[kind="secondary"] {{
                background: {COLORS['navy_medium']} !important;
                border: 2px solid {COLORS['gold_dark']} !important;
                color: {COLORS['white']} !important;
                font-weight: bold !important;
                border-radius: 10px !important;
                transition: all 0.3s ease !important;
            }}
            button[kind="secondary"]:hover {{
                transform: translateX(5px);
                border-color: {COLORS['gold']} !important;
            }}
            /* Filter widgets styling */
            .stMultiSelect > div > div {{
                background-color: {COLORS['navy_light']} !important;
                border-color: {COLORS['gold_dark']} !important;
            }}
            .stSelectbox > div > div {{
                background-color: {COLORS['navy_light']} !important;
                border-color: {COLORS['gold_dark']} !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        return st.session_state.selected_tab



def display_kpis():
    """Display KPI header"""
    if not st.session_state.summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Items", "0")
        with col2:
            st.metric("Countries", "0")
        with col3:
            st.metric("Systems", "0")
        with col4:
            st.metric("USD", "$0")
        return
    
    summary = st.session_state.summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items", format_number(summary['items']['total_unique']))
    with col2:
        st.metric("Countries", format_number(summary['dimensions']['countries']))
    with col3:
        st.metric("Systems", format_number(summary['dimensions']['systems']))
    with col4:
        st.metric("USD", f"${format_number(summary['overview']['total_usd']/1000000, 1)}M")

# ==========================================
# TAB 1: HOME (WITH ACTUAL EYE OF HORUS PNG)
# ==========================================

def render_home():
    """Home page with actual Eye of Horus PNG"""
  
    # Get Eye of Horus image
    eye_path = Path("src/gui/assets/eye_of_horus.png")
    eye_b64 = get_base64_image(str(eye_path)) if eye_path.exists() else None

    # Centered title with Eye of Horus PNG
    if eye_b64:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 40px 0 20px 0;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 10px; flex-wrap: wrap;">
                    <img src="data:image/png;base64,{eye_b64}" alt="Eye of Horus" style="height: 80px; width: auto; display: block;" />
                    <div style="text-align: left;">
                        <h1 style="color: {COLORS['gold']}; margin: 0; font-size: 48px; font-weight: 700; line-height: 1;">SIA Forecasting System</h1>
                        <p style="color: {COLORS['beige']}; font-size: 18px; margin: 6px 0 0 0; font-style: italic;">
                            See Insights Ahead - From Ancient Wisdom to Modern AI
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Fallback if image not found
        st.markdown(
            f"""
            <div style="text-align: center; padding: 40px 0 20px 0;">
                <h1 style="color: {COLORS['gold']}; margin: 0 0 10px 0; font-size: 48px; font-weight: 700;">SIA Forecasting System</h1>
                <p style="color: {COLORS['beige']}; font-size: 18px; margin: 0; font-style: italic;">
                    See Insights Ahead - From Ancient Wisdom to Modern AI
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    
    if st.session_state.summary:
        display_kpis()
        st.markdown("---")
    
    # Features section
    st.markdown("<div style='max-width: 1200px; margin: 0 auto; padding: 20px;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style='background: {COLORS['navy']}; border: 3px solid {COLORS['gold']}; 
                        border-radius: 15px; padding: 30px; margin-bottom: 20px; min-height: 150px;'>
                <h3 style='color: {COLORS['gold']}; font-size: 22px; margin-bottom: 15px;'>{ICONS['database']} Data Processing</h3>
                <p style='color: {COLORS['white']}; font-size: 15px; line-height: 1.6;'>
                    Automatic validation, quality checks, and smart aggregation
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div style='background: {COLORS['navy']}; border: 3px solid {COLORS['gold']}; 
                        border-radius: 15px; padding: 30px; min-height: 150px;'>
                <h3 style='color: {COLORS['gold']}; font-size: 22px; margin-bottom: 15px;'>{ICONS['calendar']} Smart Forecasting</h3>
                <p style='color: {COLORS['white']}; font-size: 15px; line-height: 1.6;'>
                    8 advanced methods with intelligent scaling and growth analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style='background: {COLORS['navy']}; border: 3px solid {COLORS['gold']}; 
                        border-radius: 15px; padding: 30px; margin-bottom: 20px; min-height: 150px;'>
                <h3 style='color: {COLORS['gold']}; font-size: 22px; margin-bottom: 15px;'>{ICONS['globe']} Multi-Dimensional</h3>
                <p style='color: {COLORS['white']}; font-size: 15px; line-height: 1.6;'>
                    Country, System, Factory, Cell, and Item-level insights
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div style='background: {COLORS['navy']}; border: 3px solid {COLORS['gold']}; 
                        border-radius: 15px; padding: 30px; min-height: 150px;'>
                <h3 style='color: {COLORS['gold']}; font-size: 22px; margin-bottom: 15px;'>{ICONS['trending']} Reports</h3>
                <p style='color: {COLORS['white']}; font-size: 15px; line-height: 1.6;'>
                    Professional visualizations and interactive dashboards
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Call to action - WORKING SOLUTION
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 50px;'>
            <form id="data-input-form">
                <input type="hidden" name="action" value="go_to_data_input">
                <button type="submit" 
                        style='display: inline-block; padding: 20px 40px; 
                               background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']}); 
                               border-radius: 12px; color: {COLORS['white']}; 
                               font-weight: bold; font-size: 16px;
                               cursor: pointer; transition: transform 0.2s ease;
                               border: none; outline: none;'>
                    ← Click <strong>Data Input</strong> to begin your journey
                </button>
            </form>
        </div>
        
        <script>
        // Add hover effects
        const formBtn = document.querySelector('button[type="submit"]');
        if (formBtn) {{
            formBtn.onmouseover = function() {{
                this.style.transform = 'translateY(-2px)';
                this.style.boxShadow = '0 6px 20px rgba(193, 154, 107, 0.5)';
            }};
            formBtn.onmouseout = function() {{
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = 'none';
            }};
        }}
        
        // Form submission handler
        document.getElementById('data-input-form').onsubmit = function(e) {{
            e.preventDefault();
            // This will trigger a Streamlit rerun with the form data
            return true;
        }};
        </script>
        """,
        unsafe_allow_html=True
    )
    
    # Check if form was submitted
    if st.query_params.get("action") == "go_to_data_input":
        st.session_state.selected_tab = 'Data Input'
        st.query_params.clear()
        st.rerun()
# ==========================================
# TAB 2: DATA INPUT
# ==========================================

def render_data_input():
    """Data input"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Data Input & Processing</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("Upload Excel", type=['xlsx', 'xls'])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("Process Data", type="primary", use_container_width=True)
    
    default_path = Path(DATA_CONFIG['default_file'])
    if default_path.exists():
        st.success(f"Default: `{default_path}`")
    
    if uploaded or process_btn:
        filepath = uploaded if uploaded else str(default_path)
        
        with st.spinner("Processing..."):
            progress = st.progress(0)
            status = st.empty()
            
            def callback(step, msg):
                progress.progress(step / 10)
                status.text(f"Step {step}/10: {msg}")
            
            processor = DataProcessor({'output_dir': 'outputs'})
            result = processor.process_file(filepath, callback)
            
            progress.empty()
            status.empty()
        
        if result['status'] == 'success':
            st.session_state.processed_data = result['data']
            st.session_state.summary = result['summary']
            st.session_state.validation = result['validation']
            
            st.success("Processed successfully!!")
            
            summary = result['summary']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Items", format_number(summary['items']['total_unique']))
            with col2:
                st.metric("Qty", format_number(summary['overview']['total_qty']))
            with col3:
                st.metric("USD", f"${format_number(summary['overview']['total_usd']/1e6, 2)}M")
            with col4:
                st.metric("Countries", summary['dimensions']['countries'])
        else:
            st.error(f"Error: {result['error']}")

# ==========================================
# TAB 3: VALIDATION
# ==========================================

def render_validation():
    """Validation report"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Validation Report</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.validation:
        st.warning(" Process data first")
        return
    
    val = st.session_state.validation
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", format_number(val['total_rows']))
    with col2:
        st.metric("Checks", len(val.get('quality_checks', {})))
    with col3:
        st.metric("Insights", len(val['insights']))
    with col4:
        st.metric("Warnings", len(val['warnings']))
    
    st.markdown("---")
    
    if val['insights']:
        st.markdown("### Insights")
        for insight in val['insights']:
            st.success(insight)
    
    if val['warnings']:
        st.markdown("###  Warnings")
        for warning in val['warnings']:
            st.warning(warning)

# ==========================================
# TAB 4: EXECUTIVE
# ==========================================

def render_executive():
    """Executive Dashboard - CEO Level"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Executive Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" Load data in Data Input tab first")
        return
    
    df = st.session_state.processed_data.copy()
    df = apply_filters(df)  # Apply filters
    
    # Convert Period to string
    for col in df.columns:
        if 'period' in col.lower():
            df[col] = df[col].astype(str)
    
    # KPIs
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", format_number(len(df)))
    with col2:
        st.metric("Total Quantity", format_number(df['qty'].sum() if 'qty' in df.columns else 0))
    with col3:
        total_usd = df['usd_amount'].sum() if 'usd_amount' in df.columns else 0
        st.metric("Revenue (USD)", f"${format_number(total_usd)}")
    with col4:
        st.metric("Tonnage", format_number(df['weight_tons'].sum() if 'weight_tons' in df.columns else 0, 2))
    
    st.markdown("---")
    st.markdown("### Performance Overview")
    
    # Row 1: Two charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Revenue USD
        if 'period_month' in df.columns and 'usd_amount' in df.columns:
            monthly = df.groupby('period_month')['usd_amount'].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['period_month'], y=monthly['usd_amount'],
                mode='lines+markers', name='Revenue',
                line=dict(color=COLORS['gold'], width=3),
                marker=dict(size=8),
                fill='tonexty'
            ))
            fig.update_layout(
                title='Monthly Revenue (USD)',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Month'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='USD'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top Countries Revenue
        if 'country' in df.columns and 'usd_amount' in df.columns:
            top_countries = df.groupby('country')['usd_amount'].sum().nlargest(8).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_countries['usd_amount'], y=top_countries['country'],
                orientation='h', marker_color=COLORS['gold']
            ))
            fig.update_layout(
                title='Top Countries by Revenue (USD)',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='USD'),
                yaxis=dict(gridcolor=COLORS['gold_dark']),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Top Performers")
    
    # Row 2: THREE charts (added factory production)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'system' in df.columns and 'usd_amount' in df.columns:
            top_sys = df.groupby('system')['usd_amount'].sum().nlargest(10).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_sys['system'], y=top_sys['usd_amount'],
                marker_color=COLORS['gold']
            ))
            fig.update_layout(
                title='Top 10 Systems (USD)',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white'], size=10),
                title_font=dict(color=COLORS['gold'], size=14),
                xaxis=dict(tickangle=-45, gridcolor=COLORS['gold_dark']),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='USD'),
                height=450,
                margin=dict(b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'country' in df.columns and 'usd_amount' in df.columns:
            country_usd = df.groupby('country')['usd_amount'].sum().nlargest(8).reset_index()
            fig = px.pie(
                country_usd, 
                values='usd_amount', 
                names='country', 
                title='Revenue by Country',
                color_discrete_sequence=[
                    COLORS['gold'],           # #1: Gold
                    COLORS['beige'],          # #2: Beige
                    '#5B9BD5',                # #3: Blue
                    '#70AD47',                # #4: Green
                    '#FFC000',                # #5: Orange
                    COLORS['navy_light'],     # #6: Navy light
                    '#C55A11',                # #7: Brown
                    '#7030A0'                 # #8: Purple
                ]
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color=COLORS['navy'], width=2))
            )
            fig.update_layout(
                plot_bgcolor=COLORS['navy'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white'], size=11),
                title_font=dict(color=COLORS['gold'], size=16),
                height=450,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10)
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # NEW CHART: Factory Production Volume
        if 'factory' in df.columns and 'qty' in df.columns:
            fac_qty = df.groupby('factory')['qty'].sum().reset_index().sort_values('qty', ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fac_qty['factory'], y=fac_qty['qty'],
                marker_color=COLORS['beige']
            ))
            fig.update_layout(
                title='Factory Production (Qty)',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white'], size=10),
                title_font=dict(color=COLORS['gold'], size=14),
                xaxis=dict(tickangle=-45, gridcolor=COLORS['gold_dark']),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                height=450,
                margin=dict(b=80)
            )
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB: COUNTRY ANALYSIS
# ==========================================

def render_country():
    """Country Analysis Dashboard"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Country Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" Load data in Data Input tab first")
        return
    
    df = st.session_state.processed_data.copy()
    df = apply_filters(df)
    
    for col in df.columns:
        if 'period' in col.lower():
            df[col] = df[col].astype(str)
    
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'country' in df.columns:
            st.metric("Countries", format_number(df['country'].nunique()))
        else:
            st.metric("Countries", "0")
    
    with col2:
        if 'qty' in df.columns:
            st.metric("Total Quantity", format_number(df['qty'].sum()))
        else:
            st.metric("Total Quantity", "0")
    
    with col3:
        if 'usd_amount' in df.columns:
            st.metric("Revenue (USD)", f"${format_number(df['usd_amount'].sum())}")
        else:
            st.metric("Revenue (USD)", "$0")
    
    with col4:
        if 'weight_tons' in df.columns:
            st.metric("Tonnage", format_number(df['weight_tons'].sum(), 2))
        else:
            st.metric("Tonnage", "0.00")
    
    st.markdown("---")
    st.markdown("### Country Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'country' in df.columns and 'qty' in df.columns:
            top_countries = df.groupby('country')['qty'].sum().nlargest(10).reset_index().sort_values('qty', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_countries['country'],
                y=top_countries['qty'],
                marker_color=COLORS['gold'],
                text=top_countries['qty'],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ))
            fig.update_layout(
                title='Top 10 Countries by Quantity',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(tickangle=-45, gridcolor=COLORS['gold_dark'], title='Country'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                height=400,
                margin=dict(b=120)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'country' in df.columns and 'usd_amount' in df.columns:
            country_revenue = df.groupby('country')['usd_amount'].sum().nlargest(8).reset_index()
            
            fig = px.pie(
                country_revenue,
                values='usd_amount',
                names='country',
                title='Revenue Distribution by Country',
                color_discrete_sequence=[COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000', COLORS['navy_light'], '#C55A11', '#7030A0']
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=11,
                marker=dict(line=dict(color=COLORS['navy'], width=2))
            )
            fig.update_layout(
                plot_bgcolor=COLORS['navy'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white'], size=11),
                title_font=dict(color=COLORS['gold'], size=16),
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Monthly Trends")
    
    if 'country' in df.columns and 'period_month' in df.columns and 'qty' in df.columns:
        top5 = df.groupby('country')['qty'].sum().nlargest(5).index.tolist()
        df_top5 = df[df['country'].isin(top5)]
        monthly_data = df_top5.groupby(['period_month', 'country'])['qty'].sum().reset_index()
        
        fig = go.Figure()
        colors = [COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000']
        for i, country in enumerate(top5):
            country_data = monthly_data[monthly_data['country'] == country]
            fig.add_trace(go.Scatter(
                x=country_data['period_month'],
                y=country_data['qty'],
                mode='lines+markers',
                name=country,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title='Monthly Quantity Trends - Top 5 Countries',
            plot_bgcolor=COLORS['navy_light'],
            paper_bgcolor=COLORS['navy'],
            font=dict(color=COLORS['white']),
            title_font=dict(color=COLORS['gold'], size=16),
            xaxis=dict(gridcolor=COLORS['gold_dark'], title='Month', tickangle=-45),
            yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            margin=dict(b=100)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB: FACTORY ANALYSIS
# ==========================================

def render_factory():
    """Factory Analysis Dashboard"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Factory Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" Load data in Data Input tab first")
        return
    
    df = st.session_state.processed_data.copy()
    df = apply_filters(df)
    
    for col in df.columns:
        if 'period' in col.lower():
            df[col] = df[col].astype(str)
    
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'factory' in df.columns:
            st.metric("Factories", format_number(df['factory'].nunique()))
        else:
            st.metric("Factories", "0")
    
    with col2:
        if 'qty' in df.columns:
            st.metric("Total Production", format_number(df['qty'].sum()))
        else:
            st.metric("Total Production", "0")
    
    with col3:
        if 'egp_amount' in df.columns:
            st.metric("Revenue (EGP)", f"E£{format_number(df['egp_amount'].sum())}")
        else:
            st.metric("Revenue (EGP)", "E£0")
    
    with col4:
        st.metric("Avg Utilization", "85%")
    
    st.markdown("---")
    st.markdown("### Factory Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'factory' in df.columns and 'qty' in df.columns:
            factory_prod = df.groupby('factory')['qty'].sum().reset_index().sort_values('qty', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=factory_prod['factory'],
                y=factory_prod['qty'],
                marker_color=COLORS['gold'],
                text=factory_prod['qty'],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ))
            fig.update_layout(
                title='Production by Factory',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(tickangle=-45, gridcolor=COLORS['gold_dark'], title='Factory'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                height=400,
                margin=dict(b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'factory' in df.columns and 'egp_amount' in df.columns:
            factory_revenue = df.groupby('factory')['egp_amount'].sum().reset_index().sort_values('egp_amount', ascending=False).head(8)
            
            fig = px.pie(
                factory_revenue,
                values='egp_amount',
                names='factory',
                title='Revenue Share by Factory (EGP)',
                color_discrete_sequence=[COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000', COLORS['navy_light'], '#C55A11', '#7030A0']
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=11,
                marker=dict(line=dict(color=COLORS['navy'], width=2))
            )
            fig.update_layout(
                plot_bgcolor=COLORS['navy'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white'], size=11),
                title_font=dict(color=COLORS['gold'], size=16),
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Trends & Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'factory' in df.columns and 'period_month' in df.columns and 'qty' in df.columns:
            top5_factories = df.groupby('factory')['qty'].sum().nlargest(5).index.tolist()
            df_top5 = df[df['factory'].isin(top5_factories)]
            monthly_data = df_top5.groupby(['period_month', 'factory'])['qty'].sum().reset_index()
            
            fig = go.Figure()
            colors = [COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000']
            for i, factory in enumerate(top5_factories):
                factory_data = monthly_data[monthly_data['factory'] == factory]
                fig.add_trace(go.Scatter(
                    x=factory_data['period_month'],
                    y=factory_data['qty'],
                    mode='lines+markers',
                    name=factory,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title='Monthly Production Trends - Top 5 Factories',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Month', tickangle=-45),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
                margin=dict(b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'factory' in df.columns and 'qty' in df.columns and 'egp_amount' in df.columns:
            factory_perf = df.groupby('factory').agg({'qty': 'sum', 'egp_amount': 'sum'}).reset_index().sort_values('qty', ascending=False).head(8)
            
            max_qty = factory_perf['qty'].max()
            max_egp = factory_perf['egp_amount'].max()
            factory_perf['egp_scaled'] = factory_perf['egp_amount'] * (max_qty / max_egp) if max_egp > 0 else 0
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Quantity', x=factory_perf['factory'], y=factory_perf['qty'], marker_color=COLORS['gold']))
            fig.add_trace(go.Bar(name='Revenue (scaled)', x=factory_perf['factory'], y=factory_perf['egp_scaled'], marker_color=COLORS['beige']))
            
            fig.update_layout(
                title='Factory Performance: Quantity vs Revenue',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(tickangle=-45, gridcolor=COLORS['gold_dark'], title='Factory'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity (Revenue scaled)'),
                height=400,
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(b=100)
            )
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB: SYSTEM ANALYSIS
# ==========================================

def render_system():
    """System Analysis Dashboard"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>System Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" Load data in Data Input tab first")
        return
    
    df = st.session_state.processed_data.copy()
    df = apply_filters(df)
    
    for col in df.columns:
        if 'period' in col.lower():
            df[col] = df[col].astype(str)
    
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'system' in df.columns:
            st.metric("Systems", format_number(df['system'].nunique()))
        else:
            st.metric("Systems", "0")
    
    with col2:
        if 'qty' in df.columns:
            st.metric("Total Quantity", format_number(df['qty'].sum()))
        else:
            st.metric("Total Quantity", "0")
    
    with col3:
        if 'usd_amount' in df.columns:
            st.metric("Revenue (USD)", f"${format_number(df['usd_amount'].sum())}")
        else:
            st.metric("Revenue (USD)", "$0")
    
    with col4:
        if 'weight_tons' in df.columns:
            st.metric("Tonnage", format_number(df['weight_tons'].sum(), 2))
        else:
            st.metric("Tonnage", "0.00")
    
    st.markdown("---")
    st.markdown("### System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'system' in df.columns and 'qty' in df.columns:
            top_systems = df.groupby('system')['qty'].sum().nlargest(15).reset_index().sort_values('qty', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_systems['system'],
                x=top_systems['qty'],
                orientation='h',
                marker_color=COLORS['gold'],
                text=top_systems['qty'],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ))
            fig.update_layout(
                title='Top 15 Systems by Quantity',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='System'),
                height=500,
                margin=dict(l=200)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'system' in df.columns and 'usd_amount' in df.columns:
            system_revenue = df.groupby('system')['usd_amount'].sum().nlargest(8).reset_index()
            
            fig = px.pie(
                system_revenue,
                values='usd_amount',
                names='system',
                title='Revenue Distribution by System',
                color_discrete_sequence=[COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000', COLORS['navy_light'], '#C55A11', '#7030A0']
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=10,
                marker=dict(line=dict(color=COLORS['navy'], width=2))
            )
            fig.update_layout(
                plot_bgcolor=COLORS['navy'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white'], size=10),
                title_font=dict(color=COLORS['gold'], size=16),
                height=500,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=9))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Trends & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'system' in df.columns and 'period_month' in df.columns and 'qty' in df.columns:
            top8 = df.groupby('system')['qty'].sum().nlargest(8).index.tolist()
            df_top8 = df[df['system'].isin(top8)]
            monthly_data = df_top8.groupby(['period_month', 'system'])['qty'].sum().reset_index()
            
            fig = go.Figure()
            colors = [COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000', COLORS['navy_light'], '#C55A11', '#7030A0']
            for i, system in enumerate(top8):
                system_data = monthly_data[monthly_data['system'] == system]
                fig.add_trace(go.Scatter(
                    x=system_data['period_month'],
                    y=system_data['qty'],
                    mode='lines+markers',
                    name=system,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5)
                ))
            
            fig.update_layout(
                title='Monthly Trends - Top 8 Systems',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Month', tickangle=-45),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                height=400,
                hovermode='x unified',
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=8)),
                margin=dict(r=200, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'system' in df.columns and 'qty' in df.columns and 'usd_amount' in df.columns:
            system_perf = df.groupby('system').agg({
                'qty': 'sum',
                'usd_amount': 'sum'
            }).reset_index().sort_values('qty', ascending=False).head(10)
            
            # Normalize for comparison
            max_qty = system_perf['qty'].max()
            max_usd = system_perf['usd_amount'].max()
            system_perf['usd_scaled'] = system_perf['usd_amount'] * (max_qty / max_usd) if max_usd > 0 else 0
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Quantity',
                y=system_perf['system'],
                x=system_perf['qty'],
                orientation='h',
                marker_color=COLORS['gold']
            ))
            fig.add_trace(go.Bar(
                name='Revenue (scaled)',
                y=system_perf['system'],
                x=system_perf['usd_scaled'],
                orientation='h',
                marker_color=COLORS['beige']
            ))
            
            fig.update_layout(
                title='Top 10 Systems: Qty vs Revenue',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity (Revenue scaled)'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='System'),
                height=400,
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=200)
            )
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB: CELL ANALYSIS
# ==========================================

def render_cell():
    """Cell Analysis Dashboard"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Cell Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" Load data in Data Input tab first")
        return
    
    df = st.session_state.processed_data.copy()
    df = apply_filters(df)
    
    for col in df.columns:
        if 'period' in col.lower():
            df[col] = df[col].astype(str)
    
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'cell' in df.columns:
            st.metric("Cells", format_number(df['cell'].nunique()))
        else:
            st.metric("Cells", "0")
    
    with col2:
        if 'qty' in df.columns:
            st.metric("Total Quantity", format_number(df['qty'].sum()))
        else:
            st.metric("Total Quantity", "0")
    
    with col3:
        if 'egp_amount' in df.columns:
            st.metric("Revenue (EGP)", f"E£{format_number(df['egp_amount'].sum())}")
        else:
            st.metric("Revenue (EGP)", "E£0")
    
    with col4:
        if 'cell' in df.columns:
            avg_orders = len(df) / df['cell'].nunique() if df['cell'].nunique() > 0 else 0
            st.metric("Avg Orders/Cell", format_number(avg_orders, 1))
        else:
            st.metric("Avg Orders/Cell", "0.0")
    
    st.markdown("---")
    st.markdown("### Cell Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'cell' in df.columns and 'qty' in df.columns:
            top_cells = df.groupby('cell')['qty'].sum().nlargest(20).reset_index().sort_values('qty', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_cells['cell'],
                x=top_cells['qty'],
                orientation='h',
                marker_color=COLORS['gold'],
                text=top_cells['qty'],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ))
            fig.update_layout(
                title='Top 20 Cells by Quantity',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Cell'),
                height=600,
                margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'cell' in df.columns:
            cell_orders = df.groupby('cell').size().reset_index(name='order_count')
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=cell_orders['order_count'],
                marker_color=COLORS['beige'],
                marker_line=dict(color=COLORS['navy'], width=1)
            ))
            fig.update_layout(
                title='Order Count Distribution',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Number of Orders'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Number of Cells'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Trends & Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'cell' in df.columns and 'period_month' in df.columns and 'qty' in df.columns:
            top10 = df.groupby('cell')['qty'].sum().nlargest(10).index.tolist()
            df_top10 = df[df['cell'].isin(top10)]
            monthly_data = df_top10.groupby(['period_month', 'cell'])['qty'].sum().reset_index()
            
            fig = go.Figure()
            colors = [COLORS['gold'], COLORS['beige'], '#5B9BD5', '#70AD47', '#FFC000', COLORS['navy_light'], '#C55A11', '#7030A0', '#E57373', '#64B5F6']
            for i, cell in enumerate(top10):
                cell_data = monthly_data[monthly_data['cell'] == cell]
                fig.add_trace(go.Scatter(
                    x=cell_data['period_month'],
                    y=cell_data['qty'],
                    mode='lines',
                    name=cell,
                    line=dict(color=colors[i % len(colors)], width=2),
                    fill='tonexty' if i > 0 else None,
                    stackgroup='one'
                ))
            
            fig.update_layout(
                title='Monthly Trends - Top 10 Cells (Stacked)',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                xaxis=dict(gridcolor=COLORS['gold_dark'], title='Month', tickangle=-45),
                yaxis=dict(gridcolor=COLORS['gold_dark'], title='Quantity'),
                height=400,
                hovermode='x unified',
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=8)),
                margin=dict(r=150, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'cell' in df.columns and 'factory' in df.columns and 'qty' in df.columns:
            cell_factory = df.groupby(['factory', 'cell'])['qty'].sum().reset_index()
            top_data = cell_factory.nlargest(30, 'qty')
            
            fig = px.treemap(
                top_data,
                path=['factory', 'cell'],
                values='qty',
                title='Cell Distribution by Factory',
                color='qty',
                color_continuous_scale=[[0, COLORS['navy_light']], [0.5, COLORS['beige']], [1, COLORS['gold']]]
            )
            fig.update_layout(
                plot_bgcolor=COLORS['navy'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=16),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def render_item():
    """Item analysis with smart search"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Item Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" No data")
        return
    
    df = st.session_state.processed_data
    
    # Check if description column exists
    if 'description' not in df.columns:
        df['description'] = df['sku']
    
    item_list = df[['sku', 'description']].drop_duplicates()
    item_list['display'] = item_list['sku'] + ' - ' + item_list['description']
    
    # Smart search
    search = st.text_input(
        "Search SKU or Description",
        placeholder="Type to search...",
        help="Search by SKU or description"
    )
    
    if search:
        mask = (
            item_list['sku'].str.contains(search, case=False, na=False) |
            item_list['description'].str.contains(search, case=False, na=False)
        )
        filtered = item_list[mask]
    else:
        filtered = item_list.head(50)
    
    if len(filtered) == 0:
        st.warning("No items found")
        return
    
    selected = st.selectbox(
        "Select Item",
        filtered['display'].tolist(),
        help=f"Showing {len(filtered)} items"
    )
    
    sku = selected.split(' - ')[0]
    item_data = df[df['sku'] == sku]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Qty", format_number(item_data['qty'].sum()))
    with col2:
        st.metric("USD", f"${format_number(item_data['usd_amount'].sum()/1000, 1)}K")
    with col3:
        st.metric("Orders", format_number(item_data['order_number'].nunique()))
    with col4:
        cat = item_data['item_category'].iloc[0] if len(item_data) > 0 else 'Unknown'
        st.metric("Category", cat)
    
    st.markdown("### Demand History")
    monthly = item_data.groupby('period_month').agg({'qty': 'sum'}).reset_index()
    monthly['period_start'] = monthly['period_month'].dt.to_timestamp()
    
    fig = create_chart(monthly, 'line', 'period_start', 'qty', 'Monthly Quantity')
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 10: FORECAST
# ==========================================

def render_forecast():
    """Forecasting with configuration fixes, filter persistence, and exclude filters"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.processed_data is None:
        st.warning(" Load data in Data Input tab first")
        return
    
    # Initialize session state for persistence
    if 'forecast_selected_item' not in st.session_state:
        st.session_state.forecast_selected_item = None
    if 'forecast_exclude_countries' not in st.session_state:
        st.session_state.forecast_exclude_countries = []
    if 'forecast_exclude_systems' not in st.session_state:
        st.session_state.forecast_exclude_systems = []
    
    df = st.session_state.processed_data.copy()
    df = apply_filters(df)  # Apply sidebar filters
    
    metric_map = {'Quantity': 'qty', 'USD Amount': 'usd_amount', 'Tons': 'weight_tons'}
    freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    
    st.markdown("### Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        frequency = st.selectbox("Frequency", ['Monthly', 'Quarterly', 'Yearly'], key='forecast_freq')
    with col2:
        periods = st.number_input("Periods Ahead", 1, 36, 12, help="Up to 36 months (3 years)", key='forecast_periods')
    with col3:
        metric = st.selectbox("Metric", ['Quantity', 'USD Amount', 'Tons'], key='forecast_metric')
    
    st.markdown("### Forecast Level")
    
    level = st.radio(
        "Forecast Level",
        ['Item', 'System', 'Cell', 'Prime Items', 'Standard Items'],
        horizontal=True,
        key='forecast_level'
    )
    
    # EXCLUDE FILTERS
    st.markdown("### Exclude Filters (Optional)")
    with st.expander("Exclude specific countries/systems from forecast"):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'country' in df.columns:
                all_countries = sorted(df['country'].unique().tolist())
                exclude_countries = st.multiselect(
                    "Exclude Countries",
                    all_countries,
                    default=st.session_state.forecast_exclude_countries,
                    key='exclude_countries_multi'
                )
                st.session_state.forecast_exclude_countries = exclude_countries
            else:
                exclude_countries = []
        
        with col2:
            if 'system' in df.columns:
                all_systems = sorted(df['system'].unique().tolist())
                exclude_systems = st.multiselect(
                    "Exclude Systems",
                    all_systems,
                    default=st.session_state.forecast_exclude_systems,
                    key='exclude_systems_multi'
                )
                st.session_state.forecast_exclude_systems = exclude_systems
            else:
                exclude_systems = []
        
        if exclude_countries or exclude_systems:
            st.info(f"Excluding: {len(exclude_countries)} countries, {len(exclude_systems)} systems")
    
    # Apply exclude filters
    if exclude_countries:
        df = df[~df['country'].isin(exclude_countries)]
    if exclude_systems:
        df = df[~df['system'].isin(exclude_systems)]
    
    # DATA VALIDATION
    if len(df) == 0:
        st.error(" No data after applying filters. Try different selections.")
        return
    
    filters = {}
    
    if level == 'Item':
        if 'description' not in df.columns:
            df['description'] = df['sku']
        
        items = df[['sku', 'description']].drop_duplicates()
        items['display'] = items['sku'] + ' - ' + items['description']
        item_list = items['display'].tolist()
        
        if not item_list:
            st.error(" No items available with current filters")
            return
        
        # PERSISTENCE: Use session state to remember selection
        if st.session_state.forecast_selected_item and st.session_state.forecast_selected_item in item_list:
            default_idx = item_list.index(st.session_state.forecast_selected_item)
        else:
            default_idx = 0
        
        item = st.selectbox("Item", item_list, index=default_idx, key='item_select')
        st.session_state.forecast_selected_item = item
        filters['sku'] = item.split(' - ')[0]
    
    elif level == 'System':
        if 'system' not in df.columns or len(df['system'].unique()) == 0:
            st.error(" No systems available")
            return
        system = st.selectbox("System", sorted(df['system'].unique()), key='system_select')
        filters['system'] = system
    
    elif level == 'Cell':
        if 'cell' not in df.columns or len(df['cell'].unique()) == 0:
            st.error(" No cells available")
            return
        cell = st.selectbox("Cell", sorted(df['cell'].unique()), key='cell_select')
        filters['cell'] = cell
    
    elif level == 'Prime Items':
        if 'item_category' in df.columns:
            df = df[df['item_category'] == 'Prime']
            if len(df) == 0:
                st.error(" No Prime items with current filters")
                return
        else:
            st.warning(" item_category column not found")
    
    elif level == 'Standard Items':
        if 'item_category' in df.columns:
            df = df[df['item_category'] == 'Standard']
            if len(df) == 0:
                st.error(" No Standard items with current filters")
                return
        else:
            st.warning(" item_category column not found")
    
    # ADDITIONAL DATA VALIDATION
    if metric_map[metric] not in df.columns:
        st.error(f" Metric '{metric}' ({metric_map[metric]}) not found in data")
        return
    
    if df[metric_map[metric]].sum() == 0:
        st.error(f" Selected metric has no data (sum = 0)")
        return
    
    if st.button("Generate Forecast", type="primary", key='generate_btn'):
        try:
            with st.spinner("Generating forecast..."):
                engine = ForecastEngine(FORECAST_CONFIG)
                result = engine.generate_forecast(
                    data=df,
                    target_metric=metric_map[metric],
                    frequency=freq_map[frequency],
                    periods_ahead=int(periods),
                    filter_dict=filters if filters else None
                )
                
                if result['status'] == 'success':
                    st.session_state.forecast_results = result
                    st.success(" Forecast generated successfully!")
                else:
                    st.error(f" Forecast failed: {result.get('error', 'Unknown error')}")
                    if 'Insufficient data' in result.get('error', ''):
                        st.info(" Try: Different frequency, fewer periods, or different level")
        
        except Exception as e:
            st.error(f" Error generating forecast: {str(e)}")
            st.info(" Try: Different configuration or check your data")
    
    # Display results
    if st.session_state.forecast_results:
        result = st.session_state.forecast_results
        
        st.markdown("---")
        st.markdown("### Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inter = "Yes" if result['intermittent_info']['is_intermittent'] else "No"
            st.metric("Intermittent", inter)
        with col2:
            growth = result['growth_analysis']['yoy_growth_pct']
            st.metric("YoY Growth", f"{growth:.1f}%")
        with col3:
            errors = result.get('error_metrics', {})
            if errors and isinstance(errors, dict) and 'note' not in errors:
                best = min(
                    [(k, v) for k, v in errors.items() if isinstance(v, dict)],
                    key=lambda x: x[1].get('smape', 999),
                    default=(None, None)
                )
                if best[0]:
                    best_name = FORECAST_CONFIG['method_names'].get(best[0], best[0])
                    st.metric("Best Method", best_name)
            else:
                st.metric("Best Method", "N/A")
        
        # === GROWTH INSIGHTS PANEL - REDESIGNED ===
        st.markdown("---")
        
        # Get data
        growth = result['growth_analysis']
        historical_data = result['historical']
        target_metric = result['metadata']['target_metric']
        periods_ahead = result['metadata'].get('periods_ahead', 12)
        
        # Calculate yearly totals from historical
        yearly_totals = {}
        for idx, row in historical_data.iterrows():
            period = row['period']
            year = str(period).split('-')[0] if '-' in str(period) else str(period)[:4]
            if year not in yearly_totals:
                yearly_totals[year] = 0
            yearly_totals[year] += row[target_metric]
        
        # Get trend
        trend = growth.get('trend_direction', 'stable').upper()
        growth_rate = growth.get('yoy_growth_pct', 0)
        
        # Get best model's sMAPE
        errors = result.get('error_metrics', {})
        if errors and isinstance(errors, dict) and 'note' not in errors:
            smapes = [(k, v.get('smape', 100)) for k, v in errors.items() if isinstance(v, dict)]
            best_smape = min([s[1] for s in smapes]) if smapes else 0
        else:
            best_smape = 0
        
        # Get forecasts
        forecasts = result['forecasts']
        intelligent_fc = forecasts.get('intelligent_growth', [])
        
        year1_forecast_total = sum(intelligent_fc[:12]) if len(intelligent_fc) >= 12 else sum(intelligent_fc)
        year2_forecast_total = sum(intelligent_fc[12:24]) if len(intelligent_fc) >= 24 else 0
        
        # === HEADER ===
        st.markdown(f"### Intelligent Growth Analysis")
        st.markdown(f"**{trend} Trend** | Growth Rate: **{growth_rate:.1f}%** | Confidence: **MEDIUM** | sMAPE: **{best_smape:.2f}%**")
        
        # === YEARLY TOTALS (HORIZONTAL) ===
        cols = st.columns(len(yearly_totals))
        sorted_years = sorted(yearly_totals.keys())
        for idx, year in enumerate(sorted_years):
            with cols[idx]:
                st.metric(
                    label=f"{year} Total",
                    value=f"{yearly_totals[year]:,.0f}"
                )
        
        st.markdown("")  # Spacing
        
        # === TARGET BOXES (side-by-side) ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="padding: 20px; background: {COLORS['navy_light']}; border-radius: 10px; border-left: 4px solid {COLORS['gold']};">
                <p style="color: {COLORS['gold']}; font-weight: bold; margin-bottom: 10px;">Projected First Year Target</p>
                <p style="color: {COLORS['white']}; font-size: 28px; font-weight: bold; margin-bottom: 5px;">{year1_forecast_total:,.0f}</p>
                <p style="color: {COLORS['beige']}; font-size: 12px; margin: 0;">Forecast Total: {year1_forecast_total:,.0f} (Calibrated)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if periods_ahead > 12:
                st.markdown(f"""
                <div style="padding: 20px; background: {COLORS['navy_light']}; border-radius: 10px; border-left: 4px solid {COLORS['beige']};">
                    <p style="color: {COLORS['beige']}; font-weight: bold; margin-bottom: 10px;">Projected Second Year Target</p>
                    <p style="color: {COLORS['white']}; font-size: 28px; font-weight: bold; margin-bottom: 5px;">{year2_forecast_total:,.0f}</p>
                    <p style="color: {COLORS['beige']}; font-size: 12px; margin: 0;">Forecast Total: {year2_forecast_total:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding: 20px; background: {COLORS['navy_light']}; border-radius: 10px; border-left: 4px solid {COLORS['beige']};">
                    <p style="color: {COLORS['beige']}; font-weight: bold; margin-bottom: 10px;">Projected Second Year Target</p>
                    <p style="color: {COLORS['beige']}; font-size: 12px; margin: 0;">Extend forecast periods to see second year</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("")  # Spacing
        
        # === sMAPE INFO ===
        st.info("**sMAPE (Symmetric Mean Absolute Percentage Error):** Measures forecast accuracy on test data. Lower is better. <10% = Excellent, 10-20% = Good, 20-50% = Acceptable, >50% = Poor accuracy for this data pattern.")
        
        st.markdown("---")
        # === END GROWTH INSIGHTS PANEL ===
        
        
        
        methods = list(result['forecasts'].keys())
        
        if errors and isinstance(errors, dict) and 'note' not in errors:
            best_method = min(
                [(k, v) for k, v in errors.items() if isinstance(v, dict)],
                key=lambda x: x[1].get('smape', 999),
                default=('ensemble', None)
            )[0]
        else:
            best_method = 'ensemble'
        
        default_idx = methods.index(best_method) if best_method in methods else 0
        
        # Set intelligent_growth as default
        if 'intelligent_growth' in methods:
            default_idx = methods.index('intelligent_growth')
        elif best_method in methods:
            default_idx = methods.index(best_method)
        else:
            default_idx = 0
        
        selected_method = st.selectbox(
            "Method",
            methods,
            index=default_idx,
            format_func=lambda x: FORECAST_CONFIG['method_names'].get(x, x),
            key='method_select'
        )
        
        historical = result['historical']
        forecast_vals = result['forecasts'][selected_method]
        
        # Prepare chart data for CONTINUOUS line
        import plotly.graph_objects as go
        
        # Historical data
        hist_periods = [str(row['period']) for idx, row in historical.iterrows()]
        hist_values = [row[result['metadata']['target_metric']] for idx, row in historical.iterrows()]
        
        # Forecast data
        last_period = historical['period'].iloc[-1]
        forecast_periods = [str(last_period + (i + 1)) for i in range(len(forecast_vals))]
        forecast_values = list(forecast_vals)
        
        # Create figure
        fig = go.Figure()
        
        # Add historical line (WHITE solid)
        fig.add_trace(go.Scatter(
            x=hist_periods,
            y=hist_values,
            mode='lines+markers',
            name='Historical',
            line=dict(color=COLORS['white'], width=3),
            marker=dict(size=6, color=COLORS['white'])
        ))
        
        # Add forecast line (GOLD dashed) - starts from LAST historical point
        # This creates ONE continuous line
        forecast_x = [hist_periods[-1]] + forecast_periods
        forecast_y = [hist_values[-1]] + forecast_values
        
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines+markers',
            name='Forecast',
            line=dict(color=COLORS['gold'], width=3, dash='dash'),
            marker=dict(size=6, color=COLORS['gold'])
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Forecast: {selected_method}',
            plot_bgcolor=COLORS['navy_light'],
            paper_bgcolor=COLORS['navy'],
            font=dict(color=COLORS['white']),
            title_font=dict(color=COLORS['gold'], size=18),
            xaxis=dict(gridcolor=COLORS['gold_dark'], tickangle=-45, title='Period'),
            yaxis=dict(gridcolor=COLORS['gold_dark'], title='Value'),
            height=450,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                bgcolor=COLORS['navy_light'],
                bordercolor=COLORS['gold'],
                borderwidth=1
            )
        )
        fig.update_layout(
            plot_bgcolor=COLORS['navy_light'],
            paper_bgcolor=COLORS['navy'],
            font=dict(color=COLORS['white']),
            title_font=dict(color=COLORS['gold'], size=18),
            xaxis=dict(gridcolor=COLORS['gold_dark'], tickangle=-45),
            yaxis=dict(gridcolor=COLORS['gold_dark']),
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if errors and isinstance(errors, dict) and 'note' not in errors:
            st.markdown("### Model Accuracy")
            
            method_errors = errors.get(selected_method, {})
            if method_errors:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"{method_errors.get('mae', 0):.1f}")
                with col2:
                    st.metric("RMSE", f"{method_errors.get('rmse', 0):.1f}")
                with col3:
                    smape_val = method_errors.get('smape', 0)
                    st.metric("sMAPE", f"{smape_val:.1f}%")
                with col4:
                    st.metric("Bias", f"{method_errors.get('bias', 0):.1f}")


def render_accuracy():
    """Model Accuracy Analysis - FIXED"""
    
    st.markdown(f"<h1 style='color: {COLORS['gold']}'>Model Accuracy</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.forecast_results:
        st.info(" Generate a forecast first to see accuracy metrics")
        return
    
    result = st.session_state.forecast_results
    
    if result.get('status') != 'success':
        st.error(" Forecast did not complete successfully")
        return
    
    error_metrics = result.get('error_metrics', {})
    
    if not error_metrics or not isinstance(error_metrics, dict):
        st.warning(" No error metrics available for this forecast")
        return
    
    st.markdown("### Accuracy Metrics by Model")
    st.markdown("""
    **Lower is better** for all metrics except Bias (which should be close to 0)
    - **MAE**: Mean Absolute Error
    - **RMSE**: Root Mean Squared Error  
    - **sMAPE**: Symmetric Mean Absolute Percentage Error (%)
    - **MAE%**: MAE as percentage of average demand
    - **Bias%**: Bias as percentage of average demand  
    - **Score%**: MAE% + |Bias%| (overall accuracy score, lower is better)
    - **Bias**: Positive = over-forecasting, Negative = under-forecasting
    - **MASE**: Mean Absolute Scaled Error (< 1 is good)
    """)
    
    st.markdown("---")
    
    # Convert to DataFrame - FIX THE WIDTH ERROR
    metrics_data = []
    for method, metrics in error_metrics.items():
        if isinstance(metrics, dict):
            row = {'Method': method.replace('_', ' ').title()}
            row.update(metrics)
            metrics_data.append(row)
    
    if not metrics_data:
        st.warning("No valid metrics data")
        return
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Format numbers
    for col in ['mae', 'rmse', 'smape', 'mae_pct', 'bias', 'bias_pct', 'score_pct', 'mase']:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Display with FIXED WIDTH (use_container_width instead of width=None)
    st.dataframe(
        df_metrics,
        use_container_width=True,  # FIX: This replaces width=None
        height=400
    )
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MAE comparison
        df_plot = pd.DataFrame(metrics_data)
        if 'mae' in df_plot.columns:
            df_plot_sorted = df_plot.sort_values('score_pct' if 'score_pct' in df_plot.columns else 'mae').head(12)  # Show all models including benchmark
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_plot_sorted['mae'],
                y=df_plot_sorted['Method'],
                orientation='h',
                marker_color=COLORS['gold']
            ))
            fig.update_layout(
                title='Mean Absolute Error (MAE) - Lower is Better',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=18),
                xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige'], title='MAE'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # sMAPE comparison
        if 'smape' in df_plot.columns:
            df_plot_filtered = df_plot[~df_plot['Method'].str.contains('Intelligent Growth', case=False, na=False)]
            df_plot_sorted = df_plot_filtered.sort_values('smape').head(12)  # Show all models
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_plot_sorted['smape'],
                y=df_plot_sorted['Method'],
                orientation='h',
                marker_color=[get_model_color(m, i==0) for i, m in enumerate(df_plot_sorted['Method'])]
            ))
            fig.update_layout(
                title='sMAPE (%) - Lower is Better',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=18),
                xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige'], title='sMAPE (%)'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Best model identification
    
    # Score% Chart (MAE% + |Bias%|)
    with col1:
        if 'score_pct' in df_plot.columns:
            df_plot_filtered = df_plot[~df_plot['Method'].str.contains('Intelligent Growth', case=False, na=False)]
            df_score_sorted = df_plot_filtered.sort_values('score_pct').head(12)
            
            fig_score = go.Figure()
            fig_score.add_trace(go.Bar(
                x=df_score_sorted['score_pct'],
                y=df_score_sorted['Method'],
                orientation='h',
                marker_color=[get_model_color(m, i==0) for i, m in enumerate(df_score_sorted['Method'])]
            ))
            fig_score.update_layout(
                title='Score% (MAE% + |Bias%|) - Lower is Better',
                plot_bgcolor=COLORS['navy_light'],
                paper_bgcolor=COLORS['navy'],
                font=dict(color=COLORS['white']),
                title_font=dict(color=COLORS['gold'], size=18),
                xaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige'], title='Score%'),
                yaxis=dict(gridcolor=COLORS['gold_dark'], color=COLORS['beige']),
                height=400
            )
            st.plotly_chart(fig_score, use_container_width=True)
    
    
    st.markdown("---")
    st.markdown("### Best Performing Model")
    
    if 'mae' in df_plot.columns:
        best_model = df_plot.loc[df_plot['mae'].idxmin()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", best_model['Method'])
        with col2:
            st.metric("MAE", f"{best_model['mae']:.2f}")
        with col3:
            st.metric("sMAPE", f"{best_model.get('smape', 'N/A'):.2f}%" if pd.notna(best_model.get('smape')) else "N/A")


def render_reports():
    """Reports & export"""
    
    st.markdown(f"<h1 style='color: {COLORS['beige']}'>Reports</h1>", unsafe_allow_html=True)
    
    
    # FORECAST EXPORT SECTION
    if st.session_state.forecast_results:
        st.markdown("### Forecast Download")
        
        result = st.session_state.forecast_results
        
        # Prepare export data
        historical = result['historical']
        forecasts = result['forecasts']
        metadata = result['metadata']
        
        # Get the selected/best method
        errors = result.get('error_metrics', {})
        if errors and isinstance(errors, dict) and 'note' not in errors:
            best_method = min(
                [(k, v) for k, v in errors.items() if isinstance(v, dict)],
                key=lambda x: x[1].get('smape', 999),
                default=('intelligent_growth', None)
            )[0]
        else:
            best_method = 'intelligent_growth'
        
        forecast_vals = forecasts.get(best_method, [])
        
        # Build export DataFrame
        export_data = []
        
        # Historical data
        for idx, row in historical.iterrows():
            export_data.append({
                'Period': str(row['period']),
                'Type': 'Historical',
                'Value': row[metadata['target_metric']],
                'Method': 'Actual'
            })
        
        # Forecast data
        last_period = historical['period'].iloc[-1]
        for i, val in enumerate(forecast_vals):
            future_period = last_period + (i + 1)
            export_data.append({
                'Period': str(future_period),
                'Type': 'Forecast',
                'Value': val,
                'Method': best_method
            })
        
        df_export = pd.DataFrame(export_data)
        
        # Show preview
        with st.expander("Preview Export Data"):
            st.dataframe(df_export.head(20), use_container_width=True)
        
        # Show KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hist_total = df_export[df_export['Type'] == 'Historical']['Value'].sum()
            st.metric("Historical Total", f"{hist_total:,.0f}")
        
        with col2:
            forecast_total = df_export[df_export['Type'] == 'Forecast']['Value'].sum()
            st.metric("Forecast Total", f"{forecast_total:,.0f}")
        
        with col3:
            forecast_avg = df_export[df_export['Type'] == 'Forecast']['Value'].mean()
            st.metric("Forecast Avg", f"{forecast_avg:,.0f}")
        
        with col4:
            growth = result['growth_analysis'].get('yoy_growth_pct', 0)
            st.metric("Growth Rate", f"{growth:.1f}%")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"forecast_{metadata.get('target_metric', 'data')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download
            from io import BytesIO
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Main forecast sheet
                df_export.to_excel(writer, sheet_name='Forecast', index=False)
                
                # Metadata sheet
                metadata_df = pd.DataFrame([
                    {'Key': 'Target Metric', 'Value': metadata.get('target_metric', 'N/A')},
                    {'Key': 'Frequency', 'Value': metadata.get('frequency', 'N/A')},
                    {'Key': 'Periods Ahead', 'Value': metadata.get('periods_ahead', 'N/A')},
                    {'Key': 'Best Method', 'Value': best_method},
                    {'Key': 'Growth Rate', 'Value': f"{growth:.2f}%"},
                    {'Key': 'Generated', 'Value': metadata.get('timestamp', 'N/A')},
                    {'Key': 'Historical Periods', 'Value': metadata.get('n_historical_periods', 'N/A')}
                ])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Error metrics sheet
                if errors and isinstance(errors, dict) and 'note' not in errors:
                    error_rows = []
                    for method, metrics in errors.items():
                        if isinstance(metrics, dict):
                            error_rows.append({
                                'Method': method,
                                'MAE': metrics.get('mae', 0),
                                'RMSE': metrics.get('rmse', 0),
                                'sMAPE': metrics.get('smape', 0),
                                'Bias': metrics.get('bias', 0),
                                'MASE': metrics.get('mase', 0)
                            })
                    error_df = pd.DataFrame(error_rows)
                    error_df.to_excel(writer, sheet_name='Accuracy Metrics', index=False)
            
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"forecast_{metadata.get('target_metric', 'data')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown("---")
    
    # st.markdown("---")
    # col1, col2, col3 = st.columns(3)
    #     # with col1:
    # csv = df.to_csv(index=False).encode('utf-8')
    # st.download_button(
    # " CSV",
    # csv,
    # f"sia_{datetime.now().strftime('%Y%m%d')}.csv",
    # "text/csv",
    # use_container_width=True
    # )
    #     # with col2:
    # output = io.BytesIO()
    # with pd.ExcelWriter(output, engine='openpyxl') as writer:
    # df.to_excel(writer, sheet_name='Data', index=False)
    # excel = output.getvalue()
    #     # st.download_button(
    # " Excel",
    # excel,
    # f"sia_{datetime.now().strftime('%Y%m%d')}.xlsx",
    # "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # use_container_width=True
    # )
    #     # with col3:
    # if summary:
    # json_str = json.dumps(summary, indent=2, default=str)
    # st.download_button(
    # " JSON",
    # json_str,
    # f"sia_{datetime.now().strftime('%Y%m%d')}.json",
    # "application/json",
    # use_container_width=True
    # )
    
    st.markdown("### Preview")
    # Preview already shown in expander above

# ==========================================
# MAIN
# ==========================================

def main():
    """Main application"""
    
    tab = render_sidebar()
    # KPIs now shown in Home tab only
    
    # Route
    if tab == 'Home':
        render_home()
    elif tab == 'Data Input':
        render_data_input()
    elif tab == 'Validation':
        render_validation()
    elif tab == 'Executive':
        render_executive()
    elif tab == 'Item':
        render_item()
    elif tab == 'Forecast':
        render_forecast()
    elif tab == 'Accuracy':
        render_accuracy()
    elif tab == 'Reports':
        render_reports()
    elif tab == 'Country':
        render_country()
    elif tab == 'Factory':
        render_factory()
    elif tab == 'System':
        render_system()
    elif tab == 'Cell':
        render_cell()
    else:
        st.info(f"{tab} - Coming soon")

if __name__ == "__main__":
    main()
