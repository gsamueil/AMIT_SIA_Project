"""
SIA Forecasting System - Configuration
Egyptian-Themed Professional Design System
"""

# ==========================================
# EGYPTIAN LUXURY COLOR PALETTE
# ==========================================
COLORS = {
    # EXACT PROTOTYPE COLORS
    'navy': '#173744',  # Lighter, less dark
    'navy_light': '#2A3F54',  # Much lighter
    'navy_medium': '#213040',  # Better medium tone
    
    # Gold Accents
    'gold': '#C19A6B',
    'gold_dark': '#8B6F47',
    'gold_light': '#D4AF6A',
    
    # Beige
    'beige': '#D4AF6A',
    'beige_light': '#F5E6D3',
    'beige_dark': '#B8935A',
    
    # Alert Colors
    'red': '#C62828',
    'red_light': '#E57373',
    
    # Neutrals
    'white': '#FFFFFF',
    'black': '#000000',
    'gray': '#BDBDBD',
    'gray_dark': '#424242',
    'gray_light': '#E0E0E0',
    
    # Success/Info
    'success': '#2E7D32',
    'blue': '#1565C0',
    'warning': '#F57C00',
}

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
PAGE_CONFIG = {
    'page_title': 'SIA Forecasting System',
    'page_icon': '𓂀',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'menu_items': {
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
}

# ==========================================
# SYSTEM NAME MAPPING (EDITABLE)
# ==========================================
SYSTEM_NAME_MAPPING = {
    'SYS0001: Shore Brace System': ['shorbrace Frames System'],
    'SYS0002: Acrow Frame System': ['Acrow Frames System'],
    'SYS0003: Cup Lock (St37) System': ['Cup Lock System (St37)'],
    'SYS0004: Cup Lock (St52) System': ['Cup lock System'],
    'SYS0005: Ring Lock (2.0 Inch) System': ['Ring lock System'],
    'SYS0006: Ring Lock (1.5 Inch) System': ['Ring Lock System 1.5"'],
    'SYS0007: Staircase System': ['Cup lock Stair Case', 'Ring Lock Stair case', 'Ring Lock Stair Case 1.5"'],
    'SYS0008: DYNAMIC H-20': ['SYS0008: H20 and Soldier System', 'H20 and Soldier'],
    'SYS0009: Acrow Heavy Duty Shoring System': ['Acrow Heavy Duty Shoring System'],
    'SYS0010: Acrow Heavy Duty Trusses System': ['Acrow Heavy Duty Trusses'],
    'SYS0011: DYNAMIC A-12': ['SYS0011: Acrow Beam System'],
    'SYS0012: Scaffolding Tube': ['Scaffolding'],
    'SYS0013: Accessories': ['Accessories'],
    'SYS0014: Timber H20': ['Timber H20'],
    'SYS0015: LVL Board': ['LvL Board Group'],
    'SYS0016: ECO Form System': ['ECO Form Panels System'],
    'SYS0017: TECH Form System': ['Tech Form Panels System'],
    'SYS0018: VMC Panel System': ['V.M.C Panels System'],
    'SYS0019: Vari Form System': ['Vari-Form'],
    'SYS0020: Alu Deck System': ['Alu deck system'],
    'SYS0021: Acrow Board': ['PVC Board'],
    'SYS0022: Acrow EURO Prop System': ['Acrow Euro Prop (AEP) System'],
    'SYS0023: Acrow LIGHT Prop System': ['Acrow Light Prop'],
    'SYS0024: Acrow ECO Prop System': ['ECO Prop'],
    'SYS0025: Crane Climbing System': ['Climbing System'],
    'SYS0026: Self-Climbing System': ['Self Climbing'],
    'SYS0027: Safety Net System': ['Safety Net'],
    'SYS0029: Big Steel Panel System': ['Big Steel Panel', 'Circular Columns'],
    'SYS0030: Cantilever Carriage System': ['Cantilever carriage'],
    'SYS0031: Post Tension': ['Post Tension Anchors'],
    'SYS0032: Loading Platform': ['Loading platform', 'Platform'],
    'SYS0033: Hi-Skaf System': ['Acrow Skaf System', 'Hi-Skaf System'],
    'SYS0034: Tunnel Form System': ['Tunnel Form'],
    'SYS0036: Exhibitions and Samples': ['Exhibitions and samples'],
    'SYS0037: Material': ['Material'],
    'SYS0038: Inner Platforms Beams': ['Inner Platforms Beams'],
    'SYS0039: Scrap': ['Scrap'],
    'SYS0040: Service': ['Service'],
    'SYS0042: Jump Form System': [],
    'SYS0043: ACROW X Beam': ['S-Beam'],
    'Pipes': ['Pipes'],
    'Plywood': ['Plywood'],
    'Protection Screen': ['Protection Screen'],
    'Table Form with Shorebrace & H20': ['Table Form with Shorebrace & H20'],
    'Z-FRAME System': ['Z-FRAME System']
}

# ==========================================
# DATA PROCESSING SETTINGS
# ==========================================
DATA_CONFIG = {
    'default_file': 'Data/Historical Demand Sales.xlsx',
    'output_dir': 'outputs',
    'required_sheets': ['SOs', 'Standard', 'Prime'],
    'date_column': 'Booked Date',
    'sku_column': 'Ordered Item',
    'qty_column': 'SO Remaining Qty',
}

# ==========================================
# FORECASTING SETTINGS
# ==========================================
FORECAST_CONFIG = {
    'min_periods': 12,
    'default_frequency': 'M',
    'default_periods_ahead': 12,
    'default_metric': 'qty',
    
    'intermittent_threshold_pct': 50,
    'adi_threshold': 1.32,
    'scaling_factor': 0.85,
    'growth_threshold': -10,
    
    'methods': [
        'croston',
        'sba',
        'tsb',
        'adida',
        'arima',
        'prophet',
        'iets',
        'holt_winters',
        'weighted_ma',
        'exponential_trend',
        'advanced_ensemble',
        'ensemble'
    ],
    
    'method_names': {
        'croston': "Croston's Method (Enhanced)",
        'sba': 'SBA (Syntetos-Boylan) Enhanced',
        'tsb': 'TSB (Teunter-Syntetos-Babai) Enhanced',
        'adida': 'ADIDA (Aggregate-Disaggregate) Enhanced',
        'arima': 'Auto-ARIMA with Seasonality',
        'prophet': 'Prophet (Facebook) Enhanced',
        'iets': 'Intermittent ETS Enhanced',
        'holt_winters': 'Holt-Winters Triple Smoothing',
        'weighted_ma': 'Weighted Moving Average',
        'exponential_trend': 'Exponential Trend',
        'advanced_ensemble': 'Advanced Ensemble (AI-Weighted)',
        'ensemble': 'Standard Ensemble'
    }
}

# ==========================================
# CUSTOM CSS - PROTOTYPE STYLE
# ==========================================
CUSTOM_CSS = f"""
<style>
    /* ==================== GLOBAL - MAIN LIGHT BACKGROUND ==================== */
    .main {{
        background-color: {COLORS['navy_light']} !important;
    }}
    
    .stApp {{
        background-color: {COLORS['navy_light']} !important;
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['navy_light']} !important;
    }}
    
    [data-testid="stHeader"] {{
        background-color: {COLORS['navy_light']} !important;
    }}
    
    /* ==================== SIDEBAR - NAVY DARK BACKGROUND ==================== */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['navy']} !important;
        border-right: 3px solid {COLORS['gold']} !important;
    }}
    
                    /* ==================== TAB BUTTON STYLING ==================== */
    /* Active tab (primary button) */
    button[kind="primary"] {{
        background: linear-gradient(135deg, #8B6F47, #C19A6B) !important;
        border: 2px solid #C19A6B !important;
        color: #0A1929 !important;
        font-weight: bold !important;
        box-shadow: 0 4px 12px rgba(193, 154, 107, 0.4) !important;
    }}

    /* Inactive tabs (secondary button) */
    button[kind="secondary"] {{
        background: #152535 !important;
        border: 2px solid #8B6F47 !important;
        color: #D4AF6A !important;
        font-weight: bold !important;
    }}
    
    /* Hover on inactive tabs only */
    button[kind="secondary"]:hover {{
        background: #1A2F42 !important;
        border-color: #C19A6B !important;
        transform: translateX(3px) !important;
    }}

    /* Hide radio circles */
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {{
        display: none !important;
    }}
    
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{
        display: none !important;
    }}
    
    /* Text inherits color */
    [data-testid="stSidebar"] div[role="radiogroup"] label span {{
        color: inherit !important;
    }}
    
    /* SVG icons inherit color */
    [data-testid="stSidebar"] div[role="radiogroup"] label svg {{
        width: 18px !important;
        height: 18px !important;
        flex-shrink: 0 !important;
        stroke: currentColor !important;
        fill: none !important;
    }}
    
        /* ==================== TAB BUTTON STYLING ==================== */
    /* Active tab (primary button) */
    button[kind="primary"] {{
        background: linear-gradient(135deg, #8B6F47, #C19A6B) !important;
        border: 2px solid #C19A6B !important;
        color: #0A1929 !important;
        font-weight: bold !important;
        box-shadow: 0 4px 12px rgba(193, 154, 107, 0.4) !important;
    }}
    
    /* Inactive tabs (secondary button) */
    button[kind="secondary"] {{
        background: #152535 !important;
        border: 2px solid #8B6F47 !important;
        color: #D4AF6A !important;
        font-weight: bold !important;
    }}
    
    /* Hover on inactive tabs only */
    button[kind="secondary"]:hover {{
        background: #1A2F42 !important;
        border-color: #C19A6B !important;
        transform: translateX(3px) !important;
    }}

    
    /* ==================== TAB BUTTONS - PROPER STYLING ==================== */
    /* All tab buttons base style */
    button[data-testid^="baseButton"] {{
        border-radius: 8px !important;
        padding: 12px 18px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }}
    
    /* Active tab - Gold gradient */
    button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']}) !important;
        color: {COLORS['navy']} !important;
        box-shadow: 0 4px 12px rgba(193, 154, 107, 0.4) !important;
    }}
    
    /* Inactive tabs - Browse button style */
    button[kind="secondary"] {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']}) !important;
        color: {COLORS['white']} !important;
        opacity: 0.65 !important;
    }}
    
    /* Hover on inactive tabs */
    button[kind="secondary"]:hover {{
        opacity: 1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(193, 154, 107, 0.3) !important;
    }}

    /* ==================== BUTTONS ==================== */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']}) !important;
        color: {COLORS['white']} !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['gold']}, {COLORS['beige']}) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(193, 154, 107, 0.4) !important;
    }}
    
    /* ==================== RADIO BUTTONS ==================== */
    .stRadio > label {{
        color: {COLORS['white']} !important;
        font-weight: bold !important;
    }}
    
    .stRadio > div > label {{
        color: {COLORS['white']} !important;
        background-color: {COLORS['navy_light']} !important;
        padding: 10px 16px !important;
        border-radius: 6px !important;
        border: 1px solid {COLORS['gold_dark']} !important;
        margin-right: 8px !important;
    }}
    
    .stRadio > div > label:hover {{
        background-color: {COLORS['gold_dark']} !important;
        border-color: {COLORS['gold']} !important;
    }}
    
    /* ==================== METRICS ==================== */
    [data-testid="stMetricValue"] {{
        color: {COLORS['gold']} !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS['beige']} !important;
        font-size: 14px !important;
    }}
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {COLORS['navy_light']} !important;
        border: 2px dashed {COLORS['gold']} !important;
        border-radius: 10px !important;
    }}
    
    /* ==================== SELECT BOXES ==================== */
    .stSelectbox > label {{
        color: {COLORS['white']} !important;
        font-weight: bold !important;
    }}
    
    .stSelectbox > div > div {{
        background-color: {COLORS['navy']} !important;
        color: {COLORS['white']} !important;
        border: 1px solid {COLORS['gold_dark']} !important;
    }}
    
    /* ==================== TEXT INPUTS ==================== */
    .stTextInput > label {{
        color: {COLORS['white']} !important;
        font-weight: bold !important;
    }}
    
    .stTextInput input {{
        background-color: {COLORS['navy']} !important;
        color: {COLORS['white']} !important;
        border: 1px solid {COLORS['gold_dark']} !important;
    }}
    
    /* ==================== NUMBER INPUTS ==================== */
    .stNumberInput > label {{
        color: {COLORS['white']} !important;
        font-weight: bold !important;
    }}
    
    .stNumberInput input {{
        background-color: {COLORS['navy']} !important;
        color: {COLORS['white']} !important;
        border: 1px solid {COLORS['gold_dark']} !important;
    }}
    
    /* ==================== ALERTS ==================== */
    .stSuccess {{
        background-color: {COLORS['success']} !important;
        color: {COLORS['white']} !important;
    }}
    
    .stWarning {{
        background-color: {COLORS['warning']} !important;
        color: {COLORS['white']} !important;
    }}
    
    .stError {{
        background-color: {COLORS['red']} !important;
        color: {COLORS['white']} !important;
    }}
</style>
"""

# ==========================================
# VALIDATION RULES
# ==========================================
VALIDATION_RULES = {
    'required_columns': [
        'Ordered Item',
        'SO Remaining Qty',
        'Booked Date',
        'SO Line Amount EGP',
        'Shipped Amount [USD]',
        'Not Shipped Amount [USD]'
    ],
    'max_missing_pct': {
        'sku': 5,
        'qty': 5,
        'date': 10
    }
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_system_name(system_name: str) -> str:
    """Normalize system name to standard format"""
    if not system_name or str(system_name).strip().upper() == 'NAN':
        return 'Unknown'
    
    system_name = str(system_name).strip()
    
    # Check if it's already a standard name
    if system_name in SYSTEM_NAME_MAPPING:
        return system_name
    
    # Check aliases
    for standard_name, aliases in SYSTEM_NAME_MAPPING.items():
        if system_name in aliases:
            return standard_name
        # Case-insensitive matching
        if system_name.lower() in [a.lower() for a in aliases]:
            return standard_name
    
    # Return as-is if not found
    return system_name

def get_system_aliases(standard_name: str) -> list:
    """Get all aliases for a standard system name"""
    return SYSTEM_NAME_MAPPING.get(standard_name, [])
