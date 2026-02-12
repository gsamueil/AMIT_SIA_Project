"""
SIA Forecasting System - Enterprise GUI (Professional Icons + Eye of Horus logo)
This modified file replaces emoji icons with premium inline SVG icons rendered to QIcons.
Drop this file into your project (replace your existing sia_gui.py) and run as before.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtSvg import QSvgRenderer

# ✅ CORRECT IMPORTS
from src.preprocessing.preprocessing import DataProcessor
from src.model.model import ForecastEngine

# Egyptian-Modern Dark Theme Colors
COLORS = {
    'navy': '#0A1929',
    'navy_light': '#1A2F42',
    'gold': '#C19A6B',
    'gold_dark': '#8B6F47',
    'beige': '#D4AF6A',
    'beige_light': '#F5E6D3',
    'red': '#C62828',
    'white': '#FFFFFF',
    'black': '#000000',
    'gray': '#BDBDBD',
    'gray_dark': '#424242',
    'success': '#2E7D32',
    'blue': '#1565C0'
}

# -----------------------
# SVG ICONS (PRO QUALITY)
# -----------------------
# We render these SVGs to QPixmap via QSvgRenderer so icons look crisp and consistent.

EYE_OF_HORUS_SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 240" width="160" height="96">
  <defs>
    <!-- realistic multi-stop gold gradient -->
    <linearGradient id="goldA" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#FFD98A"/>
      <stop offset="0.35" stop-color="#F4C76E"/>
      <stop offset="0.6" stop-color="#D3A85A"/>
      <stop offset="1" stop-color="#8B6F47"/>
    </linearGradient>
    <linearGradient id="goldB" x1="0" x2="1">
      <stop offset="0" stop-color="#FFF7DF" stop-opacity="0.95"/>
      <stop offset="1" stop-color="#C99F69" stop-opacity="0.9"/>
    </linearGradient>
    <filter id="innerShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feOffset dx="0" dy="2"/>
      <feGaussianBlur stdDeviation="3" result="offset-blur"/>
      <feComposite operator="out" in="SourceGraphic" in2="offset-blur" result="inverse"/>
      <feFlood flood-color="#000" flood-opacity="0.18" result="color"/>
      <feComposite operator="in" in="color" in2="inverse" result="shadow"/>
      <feComposite operator="over" in="shadow" in2="SourceGraphic"/>
    </filter>
    <filter id="dropShadow" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="3" dy="6" stdDeviation="6" flood-color="#000" flood-opacity="0.25"/>
    </filter>
  </defs>

  <!-- outer eyebrow / arch -->
  <g filter="url(#dropShadow)">
    <path d="M20 80 C80 10, 320 10, 380 80 C320 110, 80 110, 20 80 Z" fill="url(#goldA)"/>
  </g>

  <!-- main eyelid body with subtle bevel highlight -->
  <g filter="url(#innerShadow)">
    <path d="M30 110 C90 84, 160 78, 220 96 C280 114, 340 118, 370 110 C342 136, 280 152, 220 154 C160 156, 100 150, 30 134 Z" fill="url(#goldA)"/>
    <!-- highlight overlay -->
    <path d="M38 100 C90 76, 158 72, 216 88 C272 104, 330 108, 360 100" fill="none" stroke="url(#goldB)" stroke-width="6" stroke-linecap="round" opacity="0.9"/>
  </g>

  <!-- iris and pupil -->
  <g>
    <circle cx="222" cy="108" r="26" fill="#0A1929"/>
    <circle cx="222" cy="108" r="11" fill="#FFD98A"/>
    <!-- tiny specular highlight -->
    <ellipse cx="234" cy="96" rx="6" ry="3" fill="#FFF9DF" opacity="0.7"/>
  </g>

  <!-- lower decorative stroke (bevel) -->
  <path d="M40 144 C110 188, 250 182, 340 144" stroke="#E0B76A" stroke-width="12" fill="none" stroke-linecap="round" opacity="0.95"/>

  <!-- spiral tail refined to match photo -->
  <g transform="translate(28,162) scale(0.9)">
    <path d="M0 0 C18 18, 46 22, 60 10 C74 -2, 98 -4, 114 12" stroke="#C99F69" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M6 -2 C24 10, 42 12, 54 6" stroke="#FFD98A" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.95"/>
    <!-- spiral center -->
    <path d="M6 2 a12 12 0 1 1 24 0 a8 8 0 1 0 -16 0" fill="#D3A85A"/>
  </g>

  <!-- thin outline for crisp contrast -->
  <path d="M20 80 C80 10, 320 10, 380 80 C320 110, 80 110, 20 80 Z" fill="none" stroke="#5F3F24" stroke-width="1" opacity="0.6"/>
</svg>'''

# Minimal, consistent SVG icons for navigation (all use the same stroke/fill style) (all use the same stroke/fill style) (all use the same stroke/fill style)
ICON_STROKE = COLORS['beige']
ICON_FILL = 'none'

SVG_HOME = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7"/><path d="M9 22V12h6v10"/></svg>'
SVG_DATABASE = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><ellipse cx="12" cy="6" rx="9" ry="3"/><path d="M3 6v6c0 1.1 4 2 9 2s9-0.9 9-2V6"/><path d="M3 12v6c0 1.1 4 2 9 2s9-0.9 9-2v-6"/></svg>'
SVG_CHECK = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>'
SVG_CHART = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><path d="M4 19h16"/><path d="M10 19v-6"/><path d="M14 19v-10"/><path d="M18 19v-4"/></svg>'
SVG_GLOBE = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 0 20"/></svg>'
SVG_FACTORY = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><path d="M3 21h18v-7l-5 3-4-4-6 5v3z"/><path d="M9 10v4"/></svg>'
SVG_SETTINGS = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18"/><path d="M6 21V10h9v-2H9V3h2v5h4v4l3 2v7"/><path d="M14 7h4"/></svg>'
SVG_GRID = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
SVG_BOX = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><path d="M21 16V8l-9-5-9 5v8l9 5 9-5z"/><path d="M3.27 6.96L12 12.01l8.73-5.05"/></svg>'
SVG_CALENDAR = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/></svg>'
SVG_TREND = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="2"><polyline points="3 17 9 11 13 15 21 7"/></svg>'
SVG_REPORT = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="{ICON_FILL}" stroke="{ICON_STROKE}" stroke-width="1.6"><path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/><path d="M13 2v6h6"/></svg>'

# Helper: render SVG string into QPixmap
from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QPixmap, QPainter


def svg_to_pixmap(svg_str: str, size: int = 20) -> QPixmap:
    """Render an SVG string to a QPixmap of given square size."""
    svg_bytes = QByteArray(svg_str.encode('utf-8'))
    renderer = QSvgRenderer(svg_bytes)
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix


class WorkerThread(QThread):
    """Background worker for data processing"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, task_type: str, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == 'preprocess':
                processor = DataProcessor(self.kwargs.get('config', {}))
                result = processor.process_file(
                    self.kwargs['filepath'],
                    progress_callback=lambda s, m: self.progress.emit(s, m)
                )
                self.finished.emit(result)

            elif self.task_type == 'forecast':
                engine = ForecastEngine(self.kwargs.get('config', {}))
                result = engine.generate_forecast(
                    data=self.kwargs['data'],
                    target_metric=self.kwargs.get('metric', 'qty'),
                    frequency=self.kwargs.get('frequency', 'M'),
                    periods_ahead=self.kwargs.get('periods', 12),
                    filter_dict=self.kwargs.get('filters', None)
                )
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class SIAMainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Data storage
        self.current_data = None
        self.processed_data = None
        self.summary = None
        self.validation = None
        self.dimensional_views = {}
        self.forecast_result = None

        # ✅ FIXED: Use original filename without "Sample"
        self.config_file = Path('Data/Historical Demand Sales.xlsx')

        self.setup_window()
        self.create_ui()
        self.apply_theme()

    def setup_window(self):
        """Setup main window"""
        self.setWindowTitle("SIA Forecasting System - Enterprise Edition")
        self.setGeometry(50, 50, 1800, 1000)

        # Set logo
        logo_path = Path('src/gui/assets/logo.png')
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        else:
            # fallback - use Eye of Horus rendered as QIcon
            pix = svg_to_pixmap(EYE_OF_HORUS_SVG, size=48)
            self.setWindowIcon(QIcon(pix))

    def create_ui(self):
        """Create main UI"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)

        # Main content area
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = self.create_header()
        content_layout.addWidget(header)

        # Stacked widget for different views
        self.stacked_widget = QStackedWidget()

        # Create all dashboard views (ALL 12 VIEWS RESTORED)
        self.home_view = self.create_home_view()
        self.data_input_view = self.create_data_input_view()
        self.validation_view = self.create_validation_view()
        self.executive_view = self.create_executive_view()
        self.country_view = self.create_country_view()
        self.factory_view = self.create_factory_view()
        self.system_view = self.create_system_view()
        self.cell_view = self.create_cell_view()
        self.item_view = self.create_item_view()
        self.forecast_view = self.create_forecast_view()
        self.accuracy_view = self.create_accuracy_view()
        self.reports_view = self.create_reports_view()

        self.stacked_widget.addWidget(self.home_view)          # 0
        self.stacked_widget.addWidget(self.data_input_view)    # 1
        self.stacked_widget.addWidget(self.validation_view)    # 2
        self.stacked_widget.addWidget(self.executive_view)     # 3
        self.stacked_widget.addWidget(self.country_view)       # 4
        self.stacked_widget.addWidget(self.factory_view)       # 5
        self.stacked_widget.addWidget(self.system_view)        # 6
        self.stacked_widget.addWidget(self.cell_view)          # 7
        self.stacked_widget.addWidget(self.item_view)          # 8
        self.stacked_widget.addWidget(self.forecast_view)      # 9
        self.stacked_widget.addWidget(self.accuracy_view)      # 10
        self.stacked_widget.addWidget(self.reports_view)       # 11

        content_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(content_area, stretch=1)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("SIA System Ready - See Insights Ahead")

    def create_sidebar(self):
        """Create navigation sidebar"""
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setObjectName("sidebar")

        layout = QVBoxLayout(sidebar)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 20, 15, 20)

        # Logo
        logo_container = QWidget()
        logo_layout = QVBoxLayout(logo_container)
        logo_label = QLabel()

        logo_path = Path('src/gui/assets/logo.png')
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled = pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled)
        else:
            pix = svg_to_pixmap(EYE_OF_HORUS_SVG, size=120)
            logo_label.setPixmap(pix)

        logo_label.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(logo_label)

        tagline = QLabel("See Insights Ahead")
        tagline.setStyleSheet(f"font-size: 13px; color: {COLORS['beige']}; font-style: italic;")
        tagline.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(tagline)

        layout.addWidget(logo_container)

        # Separator
        layout.addWidget(self.create_separator())

        # Navigation buttons (ALL 12 RESTORED) - using premium SVG icons
        nav_items = [
            (SVG_HOME, "Home", 0),
            (SVG_DATABASE, "Data Input", 1),
            (SVG_CHECK, "Validation", 2),
            (SVG_CHART, "Executive", 3),
            (SVG_GLOBE, "Country", 4),
            (SVG_FACTORY, "Factory", 5),
            (SVG_SETTINGS, "System", 6),
            (SVG_GRID, "Cell", 7),
            (SVG_BOX, "Item", 8),
            (SVG_CALENDAR, "Forecast", 9),
            (SVG_TREND, "Accuracy", 10),
            (SVG_REPORT, "Reports", 11)
        ]

        for svg, label, index in nav_items:
            btn = self.create_nav_button_with_icon(svg, label)
            btn.clicked.connect(lambda checked, idx=index: self.switch_view(idx))
            layout.addWidget(btn)

        layout.addStretch()

        # Footer
        footer = QLabel("© ACROW 2025\nPowered by AI")
        footer.setStyleSheet(f"color: {COLORS['gray']}; font-size: 10px;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        return sidebar

    def create_nav_button_with_icon(self, svg: str, text: str) -> QPushButton:
        """Create a navigation button with an SVG icon rendered as QIcon"""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setObjectName("navButton")
        btn.setMinimumHeight(45)
        # Render icon
        pix = svg_to_pixmap(svg, size=20)
        btn.setIcon(QIcon(pix))
        btn.setIconSize(QSize(20, 20))
        # Align text to left and add spacing
        btn.setStyleSheet("text-align: left; padding-left: 16px;")
        return btn

    def create_nav_button(self, text: str) -> QPushButton:
        """Create styled navigation button (fallback)"""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setObjectName("navButton")
        btn.setMinimumHeight(45)
        return btn

    def create_separator(self) -> QFrame:
        """Create horizontal separator"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {COLORS['gold']}; max-height: 2px;")
        return line

    def create_header(self):
        """Create header with KPI cards"""
        header = QWidget()
        header.setFixedHeight(120)
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # KPI Cards with proper structure
        self.kpi_cards = []
        kpis = [
            ("Total SKUs", "0", COLORS['gold']),
            ("Countries", "0", COLORS['blue']),
            ("Systems", "0", COLORS['success']),
            ("Total USD", "$0", COLORS['red'])
        ]

        for title, value, color in kpis:
            card = self.create_kpi_card(title, value, color)
            self.kpi_cards.append(card)
            layout.addWidget(card)

        return header

    def create_kpi_card(self, title: str, value: str, color: str):
        """Create KPI card with proper styling"""
        card = QFrame()
        card.setObjectName("kpiCard")
        card.setMinimumHeight(80)

        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)
        card_layout.setContentsMargins(15, 10, 15, 10)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS['gray']};
            font-weight: bold;
            letter-spacing: 1px;
        """)
        card_layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setStyleSheet(f"""
            font-size: 26px;
            color: {color};
            font-weight: bold;
        """)
        value_label.setObjectName("kpiValue")
        card_layout.addWidget(value_label)

        return card

    # ============================================
    # VIEW CREATORS - ALL 12 VIEWS
    # ============================================

    def create_home_view(self):
        """Home/Welcome screen"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setAlignment(Qt.AlignCenter)

        webview = QWebEngineView()
        webview.setHtml(self.get_welcome_html())
        layout.addWidget(webview)

        return view

    def create_data_input_view(self):
        """Data input screen"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        title = QLabel("Data Input & Processing")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['gold']};")
        layout.addWidget(title)

        # File path display
        path_group = QGroupBox("Current Data File")
        path_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-weight: bold; 
                color: {COLORS['beige']}; 
                font-size: 14px;
                padding-top: 15px;
            }}
        """)
        path_layout = QVBoxLayout()

        self.lblFilePath = QLabel(str(self.config_file))
        self.lblFilePath.setStyleSheet(f"""
            color: {COLORS['white']}; 
            padding: 15px; 
            background: {COLORS['navy_light']}; 
            border-radius: 8px;
            font-size: 13px;
        """)
        self.lblFilePath.setWordWrap(True)
        path_layout.addWidget(self.lblFilePath)

        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btnUpload = QPushButton("Upload New File")
        self.btnUpload.clicked.connect(self.upload_file)
        self.btnUpload.setObjectName("primaryButton")
        self.btnUpload.setMinimumHeight(50)
        btn_layout.addWidget(self.btnUpload)

        self.btnRefresh = QPushButton("Process Data")
        self.btnRefresh.clicked.connect(self.refresh_data)
        self.btnRefresh.setObjectName("primaryButton")
        self.btnRefresh.setMinimumHeight(50)
        btn_layout.addWidget(self.btnRefresh)

        layout.addLayout(btn_layout)

        # Progress
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.progressBar.setMinimumHeight(30)
        self.progressBar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {COLORS['gold']};
                border-radius: 8px;
                background: {COLORS['navy_light']};
                text-align: center;
                color: {COLORS['white']};
                font-weight: bold;
                font-size: 13px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {COLORS['gold']}, stop:1 {COLORS['beige']});
                border-radius: 6px;
            }}
        """)
        layout.addWidget(self.progressBar)

        self.lblProgress = QLabel("")
        self.lblProgress.setStyleSheet(f"color: {COLORS['beige']}; font-size: 12px; padding: 5px;")
        self.lblProgress.setVisible(False)
        layout.addWidget(self.lblProgress)

        layout.addStretch()

        return view

    def create_validation_view(self):
        """Data validation screen"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Data Validation & Quality Report")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['success']};")
        layout.addWidget(title)

        self.txtValidation = QTextEdit()
        self.txtValidation.setReadOnly(True)
        self.txtValidation.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS['navy_light']};
                color: {COLORS['white']};
                border: 2px solid {COLORS['gold']};
                border-radius: 10px;
                padding: 20px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.6;
            }}
        """)
        layout.addWidget(self.txtValidation)

        return view

    def create_executive_view(self):
        """Executive dashboard"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Executive Dashboard")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['gold']};")
        layout.addWidget(title)

        self.webExecutive = QWebEngineView()
        layout.addWidget(self.webExecutive)

        return view

    def create_country_view(self):
        """Country analysis"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Country Analysis")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['blue']};")
        layout.addWidget(title)

        self.webCountry = QWebEngineView()
        layout.addWidget(self.webCountry)

        return view

    def create_factory_view(self):
        """Factory analysis"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Factory Analysis")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['success']};")
        layout.addWidget(title)

        self.webFactory = QWebEngineView()
        layout.addWidget(self.webFactory)

        return view

    def create_system_view(self):
        """System analysis"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("System Analysis")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['red']};")
        layout.addWidget(title)

        self.webSystem = QWebEngineView()
        layout.addWidget(self.webSystem)

        return view

    def create_cell_view(self):
        """Cell analysis"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Cell (Manufacturing Family) Analysis")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['beige']};")
        layout.addWidget(title)

        self.webCell = QWebEngineView()
        layout.addWidget(self.webCell)

        return view

    def create_item_view(self):
        """Item/SKU analysis"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Item (SKU) Analysis")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['gold']};")
        layout.addWidget(title)

        # Item selector
        selector_layout = QHBoxLayout()
        lbl = QLabel("Select Item:")
        lbl.setStyleSheet(f"color: {COLORS['white']}; font-weight: bold; font-size: 14px;")
        selector_layout.addWidget(lbl)

        self.cmbItem = QComboBox()
        self.cmbItem.currentTextChanged.connect(self.load_item_analysis)
        selector_layout.addWidget(self.cmbItem, stretch=1)

        layout.addLayout(selector_layout)

        self.webItem = QWebEngineView()
        layout.addWidget(self.webItem)

        return view

    def create_forecast_view(self):
        """Forecasting screen"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Demand Forecasting")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['gold']};")
        layout.addWidget(title)

        # Forecast configuration
        config_group = QGroupBox("Forecast Configuration")
        config_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-weight: bold; 
                color: {COLORS['gold']}; 
                font-size: 14px;
                padding-top: 15px;
            }}
        """)
        config_layout = QGridLayout()
        config_layout.setSpacing(15)

        # Frequency
        freq_label = QLabel("Frequency:")
        freq_label.setStyleSheet(f"color: {COLORS['white']}; font-size: 13px;")
        config_layout.addWidget(freq_label, 0, 0)

        self.cmbFrequency = QComboBox()
        self.cmbFrequency.addItems(['Monthly', 'Quarterly', 'Yearly', 'Weekly'])
        self.cmbFrequency.setMinimumHeight(35)
        config_layout.addWidget(self.cmbFrequency, 0, 1)

        # Periods ahead
        periods_label = QLabel("Periods Ahead:")
        periods_label.setStyleSheet(f"color: {COLORS['white']}; font-size: 13px;")
        config_layout.addWidget(periods_label, 0, 2)

        self.spinPeriods = QSpinBox()
        self.spinPeriods.setRange(1, 36)
        self.spinPeriods.setValue(12)
        self.spinPeriods.setMinimumHeight(35)
        config_layout.addWidget(self.spinPeriods, 0, 3)

        # Metric
        metric_label = QLabel("Metric:")
        metric_label.setStyleSheet(f"color: {COLORS['white']}; font-size: 13px;")
        config_layout.addWidget(metric_label, 1, 0)

        self.cmbMetric = QComboBox()
        self.cmbMetric.addItems(['Quantity', 'USD Amount', 'Tons'])
        self.cmbMetric.setMinimumHeight(35)
        config_layout.addWidget(self.cmbMetric, 1, 1)

        # Generate button
        self.btnGenerate = QPushButton("Generate Forecast")
        self.btnGenerate.clicked.connect(self.generate_forecast)
        self.btnGenerate.setObjectName("primaryButton")
        self.btnGenerate.setMinimumHeight(45)
        config_layout.addWidget(self.btnGenerate, 1, 2, 1, 2)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Results
        self.webForecast = QWebEngineView()
        layout.addWidget(self.webForecast)

        return view

    def create_accuracy_view(self):
        """Forecast accuracy screen"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Forecast Accuracy & Model Comparison")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['red']};")
        layout.addWidget(title)

        self.webAccuracy = QWebEngineView()
        layout.addWidget(self.webAccuracy)

        return view

    def create_reports_view(self):
        """Reports screen"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Reports & Data Export")
        title.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {COLORS['beige']};")
        layout.addWidget(title)

        self.tblReports = QTableWidget()
        self.tblReports.setStyleSheet(f"""
            QTableWidget {{
                background: {COLORS['navy_light']};
                color: {COLORS['white']};
                border: 2px solid {COLORS['gold']};
                border-radius: 10px;
                gridline-color: {COLORS['gold_dark']};
            }}
            QHeaderView::section {{
                background: {COLORS['gold']};
                color: {COLORS['navy']};
                padding: 8px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.tblReports)

        return view

    def get_welcome_html(self):
        """Generate welcome HTML"""
        # Embed the Eye of Horus SVG inline at the top so the web view shows the premium logo
        eye_svg_inline = EYE_OF_HORUS_SVG.replace('\n', '')
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    background: {COLORS['navy']};
                    color: {COLORS['white']};
                    font-family: 'Segoe UI', Arial, sans-serif;
                    padding: 60px;
                    text-align: center;
                }}
                h1 {{
                    color: {COLORS['gold']};
                    font-size: 48px;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 16px;
                }}
                h2 {{
                    color: {COLORS['beige']};
                    font-size: 24px;
                    font-style: italic;
                    margin-top: 0;
                    margin-bottom: 40px;
                }}
                .features {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 30px;
                    margin-top: 40px;
                    max-width: 1000px;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .feature {{
                    background: {COLORS['navy_light']};
                    border: 3px solid {COLORS['gold']};
                    border-radius: 12px;
                    padding: 20px;
                    transition: transform 0.25s;
                }}
                .feature:hover {{
                    transform: translateY(-4px);
                    border-color: {COLORS['beige']};
                }}
                .feature h3 {{
                    color: {COLORS['gold']};
                    margin-top: 0;
                    font-size: 20px;
                }}
                .feature p {{
                    color: {COLORS['beige_light']};
                    font-size: 14px;
                    line-height: 1.6;
                }}
                .cta {{
                    margin-top: 40px;
                    padding: 14px 20px;
                    background: linear-gradient(135deg, {COLORS['gold_dark']}, {COLORS['gold']});
                    border-radius: 8px;
                    display: inline-block;
                    color: {COLORS['navy']};
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <h1>{eye_svg_inline}<span>SIA Forecasting System</span></h1>
            <h2>See Insights Ahead - From Ancient Wisdom to Modern AI</h2>

            <div class="features">
                <div class="feature">
                    <h3>Data Processing</h3>
                    <p>Automatic validation, quality checks, and smart aggregation</p>
                </div>
                <div class="feature">
                    <h3>Multi-Dimensional</h3>
                    <p>Country, System, Factory, Cell, and Item-level insights</p>
                </div>
                <div class="feature">
                    <h3>Smart Forecasting</h3>
                    <p>8 advanced methods with intelligent scaling and growth analysis</p>
                </div>
                <div class="feature">
                    <h3>Beautiful Reports</h3>
                    <p>Professional visualizations and interactive dashboards</p>
                </div>
            </div>

            <div class="cta">Click <strong>Data Input</strong> to begin your journey →</div>
        </body>
        </html>
        """

    # ============================================
    # EVENT HANDLERS - COMPLETE IMPLEMENTATIONS
    # ============================================

    def switch_view(self, index: int):
        """Switch to different view"""
        self.stacked_widget.setCurrentIndex(index)

        view_names = ["Home", "Data Input", "Validation", "Executive", "Country",
                     "Factory", "System", "Cell", "Item", "Forecast", "Accuracy", "Reports"]
        if index < len(view_names):
            self.statusBar().showMessage(f"Viewing: {view_names[index]}")

    def upload_file(self):
        """Upload new Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            str(Path.home()),
            "Excel Files (*.xlsx *.xls);;All Files (*.*)"
        )

        if file_path:
            self.config_file = Path(file_path)
            self.lblFilePath.setText(str(self.config_file))
            self.statusBar().showMessage(f"File selected: {self.config_file.name}")
            QMessageBox.information(
                self,
                "File Selected",
                f"Selected: {self.config_file.name}\n\nClick 'Process Data' to begin processing."
            )

    def refresh_data(self):
        """Refresh/process data from file"""
        if not self.config_file.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"File not found:\n{self.config_file}\n\nPlease upload a new file."
            )
            return

        # Show progress
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.lblProgress.setVisible(True)
        self.lblProgress.setText("Initializing data processing...")
        self.btnRefresh.setEnabled(False)
        self.btnUpload.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            'preprocess',
            filepath=str(self.config_file),
            config={'output_dir': 'outputs'}
        )

        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_preprocessing_complete)
        self.worker.error.connect(self.on_error)

        self.worker.start()

    def on_progress(self, step: int, message: str):
        """Update progress bar"""
        progress = int((step / 10) * 100)
        self.progressBar.setValue(progress)
        self.lblProgress.setText(f"{message}")
        QApplication.processEvents()  # Keep UI responsive

    def on_preprocessing_complete(self, result: dict):
        """Handle preprocessing completion"""
        self.progressBar.setVisible(False)
        self.lblProgress.setVisible(False)
        self.btnRefresh.setEnabled(True)
        self.btnUpload.setEnabled(True)

        if result['status'] == 'success':
            self.processed_data = result['data']
            self.summary = result['summary']
            self.validation = result['validation']

            # Update KPI cards
            self.update_kpi_cards()

            # Update validation view
            self.update_validation_view()

            # Update executive dashboard
            self.update_executive_view()

            # Show success message
            QMessageBox.information(
                self,
                "Success!",
                f"Data processed successfully!\n\nItems: {self.summary['items']['total_unique']:,}\nTotal Qty: {self.summary['overview']['total_qty']:,.0f}\nTotal USD: ${self.summary['overview']['total_usd']:,.2f}"
            )

            self.statusBar().showMessage("Data processing complete! Ready to analyze and forecast.")

        else:
            QMessageBox.critical(
                self,
                "Processing Error",
                f"Processing failed:\n\n{result.get('error', 'Unknown error')}"
            )
            self.statusBar().showMessage("Processing failed")

    def on_error(self, error_msg: str):
        """Handle errors"""
        self.progressBar.setVisible(False)
        self.lblProgress.setVisible(False)
        self.btnRefresh.setEnabled(True)
        self.btnUpload.setEnabled(True)

        QMessageBox.critical(self, "Error", f"Error occurred:\n\n{error_msg}")
        self.statusBar().showMessage("Error occurred")

    def update_kpi_cards(self):
        """Update KPI cards with data"""
        if self.summary:
            kpi_values = [
                f"{self.summary['items']['total_unique']:,}",
                f"{self.summary['dimensions']['countries']}",
                f"{self.summary['dimensions']['systems']}",
                f"${self.summary['overview']['total_usd']:,.0f}"
            ]

            for card, value in zip(self.kpi_cards, kpi_values):
                value_label = card.findChild(QLabel, "kpiValue")
                if value_label:
                    value_label.setText(value)

    def update_validation_view(self):
        """Update validation view with report"""
        if self.validation:
            report = "="*90 + "\n"
            report += "  DATA VALIDATION & QUALITY REPORT\n"
            report += "="*90 + "\n\n"

            report += f"Timestamp: {self.validation['timestamp']}\n"
            report += f"Total Rows Processed: {self.validation['total_rows']:,}\n\n"

            report += "="*90 + "\n"
            report += "KEY INSIGHTS\n"
            report += "="*90 + "\n"
            for i, insight in enumerate(self.validation.get('insights', []), 1):
                report += f"  {i}. {insight}\n"

            if self.validation.get('warnings'):
                report += "\n" + "="*90 + "\n"
                report += "WARNINGS & RECOMMENDATIONS\n"
                report += "="*90 + "\n"
                for i, warning in enumerate(self.validation['warnings'], 1):
                    report += f"  {i}. {warning}\n"

            report += "\n" + "="*90 + "\n"
            report += "DETAILED QUALITY CHECKS\n"
            report += "="*90 + "\n\n"

            qc = self.validation.get('quality_checks', {})

            report += f"Missing Data:\n"
            report += f"  • Missing SKUs: {qc.get('missing_sku', {}).get('count', 0):,} ({qc.get('missing_sku', {}).get('percentage', 0):.2f}%)\n"
            report += f"  • Missing Quantities: {qc.get('missing_qty', {}).get('count', 0):,} ({qc.get('missing_qty', {}).get('percentage', 0):.2f}%)\n"
            report += f"  • Missing Dates: {qc.get('missing_date', {}).get('count', 0):,} ({qc.get('missing_date', {}).get('percentage', 0):.2f}%)\n\n"

            report += f"Data Integrity:\n"
            report += f"  • Duplicate Rows: {qc.get('duplicates', 0):,}\n\n"

            if 'item_classification' in qc:
                ic = qc['item_classification']
                report += f"Item Classification:\n"
                report += f"  • Standard Items: {ic.get('standard_items', 0):,} ({ic.get('standard_pct', 0):.1f}%)\n"
                report += f"  • Prime Items: {ic.get('prime_items', 0):,} ({ic.get('prime_pct', 0):.1f}%)\n"
                report += f"  • Non-Standard Items: {ic.get('non_standard_items', 0):,}\n\n"

            if 'split_quantities' in qc:
                sq = qc['split_quantities']
                report += f"Split Quantities:\n"
                report += f"  • Order-item combinations with splits: {sq.get('split_groups', 0):,}\n"
                report += f"  • Total order-item combinations: {sq.get('total_groups', 0):,}\n"
                report += f"  • Split percentage: {sq.get('split_percentage', 0):.1f}%\n"

            report += "\n" + "="*90 + "\n"
            report += "VALIDATION COMPLETE - DATA READY FOR ANALYSIS\n"
            report += "="*90

            self.txtValidation.setText(report)

    def update_executive_view(self):
        """Update executive dashboard"""
        if not self.summary:
            return

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    background: {COLORS['navy']};
                    color: {COLORS['white']};
                    font-family: 'Segoe UI', Arial, sans-serif;
                    padding: 30px;
                }}
                h2 {{
                    color: {COLORS['gold']};
                    font-size: 32px;
                    border-bottom: 3px solid {COLORS['gold']};
                    padding-bottom: 10px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-top: 30px;
                }}
                .summary-card {{
                    background: {COLORS['navy_light']};
                    border: 2px solid {COLORS['gold_dark']};
                    border-radius: 12px;
                    padding: 25px;
                }}
                .summary-card h3 {{
                    color: {COLORS['gold']};
                    margin-top: 0;
                    font-size: 20px;
                }}
                .summary-card .value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: {COLORS['beige']};
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <h2>Executive Summary</h2>

            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Items</h3>
                    <div class="value">{self.summary['items']['total_unique']:,}</div>
                    <div class="label">Unique SKUs</div>
                </div>

                <div class="summary-card">
                    <h3>Total Quantity</h3>
                    <div class="value">{self.summary['overview']['total_qty']:,.0f}</div>
                    <div class="label">Units Ordered</div>
                </div>

                <div class="summary-card">
                    <h3>Total Revenue</h3>
                    <div class="value">${self.summary['overview']['total_usd']:,.0f}</div>
                    <div class="label">USD Amount</div>
                </div>

                <div class="summary-card">
                    <h3>Total Weight</h3>
                    <div class="value">{self.summary['overview']['total_tons']:,.2f}</div>
                    <div class="label">Metric Tons</div>
                </div>
            </div>

            <h2 style="margin-top: 40px;">Top 10 Items by Quantity</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>SKU</th>
                    <th>Quantity</th>
                </tr>
        """

        for i, (sku, qty) in enumerate(list(self.summary.get('top_items_by_qty', {}).items())[:10], 1):
            html += f"""
                <tr>
                    <td><strong>#{i}</strong></td>
                    <td>{sku}</td>
                    <td>{qty:,.0f}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        self.webExecutive.setHtml(html)

    def load_item_analysis(self, item_sku: str):
        """Load analysis for specific item"""
        if not item_sku or not self.processed_data:
            return

        # Filter data for this item
        item_data = self.processed_data[self.processed_data['sku'] == item_sku]

        if len(item_data) == 0:
            return

        # Generate simple analysis HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    background: {COLORS['navy']};
                    color: {COLORS['white']};
                    font-family: 'Segoe UI', Arial, sans-serif;
                    padding: 20px;
                }}
                h2 {{ color: {COLORS['gold']}; }}
                .stat {{ 
                    background: {COLORS['navy_light']};
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid {COLORS['gold']};
                }}
            </style>
        </head>
        <body>
            <h2>{item_sku}</h2>
            <div class="stat">
                <strong>Total Quantity:</strong> {item_data['qty'].sum():,.0f}
            </div>
            <div class="stat">
                <strong>Total USD:</strong> ${item_data['usd_amount'].sum():,.2f}
            </div>
            <div class="stat">
                <strong>Total Orders:</strong> {item_data['order_number'].nunique():,}
            </div>
        </body>
        </html>
        """

        self.webItem.setHtml(html)

    def generate_forecast(self):
        """Generate forecast"""
        if self.processed_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please process data first:\n1. Go to 'Data Input'\n2. Click 'Process Data'"
            )
            return

        # Get parameters
        freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y', 'Weekly': 'W'}
        metric_map = {'Quantity': 'qty', 'USD Amount': 'usd_amount', 'Tons': 'weight_tons'}

        frequency = freq_map[self.cmbFrequency.currentText()]
        metric = metric_map[self.cmbMetric.currentText()]
        periods = self.spinPeriods.value()

        self.statusBar().showMessage("Generating forecast...")
        self.btnGenerate.setEnabled(False)

        # Create worker thread for forecasting
        self.forecast_worker = WorkerThread(
            'forecast',
            data=self.processed_data,
            metric=metric,
            frequency=frequency,
            periods=periods,
            config={}
        )

        self.forecast_worker.finished.connect(self.on_forecast_complete)
        self.forecast_worker.error.connect(self.on_error)

        self.forecast_worker.start()

    def on_forecast_complete(self, result: dict):
        """Handle forecast completion"""
        self.btnGenerate.setEnabled(True)

        if result['status'] == 'success':
            self.forecast_result = result

            # Get ensemble forecast
            ensemble_fc = result['forecast'].get('ensemble', [])

            # Display result
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        background: {COLORS['navy']};
                        color: {COLORS['white']};
                        font-family: 'Segoe UI', Arial, sans-serif;
                        padding: 30px;
                    }}
                    h2 {{
                        color: {COLORS['gold']};
                        font-size: 32px;
                        border-bottom: 3px solid {COLORS['gold']};
                        padding-bottom: 10px;
                    }}
                </style>
            </head>
            <body>
                <h2>Forecast Generated Successfully!</h2>

                <div class="info-grid">
            """

            # basic stats
            html += f"<p style=\"color:{COLORS['beige']}\">Intermittent: {'Yes' if result['intermittent_info']['is_intermittent'] else 'No'} | YoY Growth: {result['growth_analysis']['yoy_growth_pct']:.1f}%</p>"

            html += "<h2>Ensemble Forecast Results</h2><table style=\"width:100%;background:#123;\"><tr><th>Period</th><th>Forecast Value</th></tr>"
            for i, value in enumerate(ensemble_fc[:12], 1):
                html += f"<tr><td>Period {i}</td><td>{value:,.2f}</td></tr>"
            html += "</table></body></html>"

            self.webForecast.setHtml(html)
            self.statusBar().showMessage("Forecast generation complete!")

            QMessageBox.information(
                self,
                "Forecast Complete",
                f"Forecast generated successfully!\n\nIntermittent: {'Yes' if result['intermittent_info']['is_intermittent'] else 'No'}\nGrowth: {result['growth_analysis']['yoy_growth_pct']:.1f}%\nMethods: {len(result['forecast'])}"
            )

        else:
            QMessageBox.critical(
                self,
                "Forecast Error",
                f"Forecast generation failed:\n\n{result.get('error', 'Unknown error')}"
            )
            self.statusBar().showMessage("Forecast generation failed")

    def apply_theme(self):
        """Apply Egyptian theme stylesheet"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {COLORS['navy']};
            }}

            QWidget#sidebar {{
                background: {COLORS['navy_light']};
                border-right: 3px solid {COLORS['gold']};
            }}

            QWidget#header {{
                background: {COLORS['navy_light']};
                border-bottom: 3px solid {COLORS['gold']};
            }}

            QPushButton#navButton {{
                background: {COLORS['navy']};
                color: {COLORS['beige']};
                border: 2px solid {COLORS['gold_dark']};
                border-radius: 10px;
                padding: 14px;
                font-size: 14px;
                font-weight: bold;
                text-align: left;
                padding-left: 20px;
            }}

            QPushButton#navButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['gold_dark']}, stop:1 {COLORS['gold']});
                color: {COLORS['white']};
                border-color: {COLORS['gold']};
            }}

            QPushButton#navButton:pressed {{
                background: {COLORS['gold']};
                color: {COLORS['navy']};
            }}

            QPushButton#primaryButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['gold']}, stop:1 {COLORS['beige']});
                color: {COLORS['navy']};
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 15px;
                font-weight: bold;
            }}

            QPushButton#primaryButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['beige']}, stop:1 {COLORS['gold']});
            }}

            QPushButton#primaryButton:pressed {{
                background: {COLORS['gold_dark']};
            }}

            QPushButton#primaryButton:disabled {{
                background: {COLORS['gray_dark']};
                color: {COLORS['gray']};
            }}

            QFrame#kpiCard {{
                background: {COLORS['navy_light']};
                border: 3px solid {COLORS['gold_dark']};
                border-radius: 12px;
                padding: 15px;
            }}

            QFrame#kpiCard:hover {{
                border-color: {COLORS['gold']};
            }}

            QGroupBox {{
                color: {COLORS['gold']};
                font-weight: bold;
                font-size: 15px;
                border: 3px solid {COLORS['gold_dark']};
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: {COLORS['gold']};
            }}

            QLabel {{
                color: {COLORS['white']};
            }}

            QComboBox, QSpinBox {{
                background: {COLORS['navy_light']};
                color: {COLORS['white']};
                border: 2px solid {COLORS['gold_dark']};
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }}

            QComboBox:hover, QSpinBox:hover {{
                border-color: {COLORS['gold']};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 25px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid {COLORS['gold']};
                margin-right: 8px;
            }}

            QSpinBox::up-button, QSpinBox::down-button {{
                background: {COLORS['gold_dark']};
                border: none;
                width: 20px;
            }}

            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background: {COLORS['gold']};
            }}

            QStatusBar {{
                background: {COLORS['navy_light']};
                color: {COLORS['beige']};
                border-top: 3px solid {COLORS['gold']};
                font-size: 12px;
                padding: 5px;
            }}

            QScrollBar:vertical {{
                background: {COLORS['navy_light']};
                width: 12px;
                border-radius: 6px;
            }}

            QScrollBar::handle:vertical {{
                background: {COLORS['gold_dark']};
                border-radius: 6px;
                min-height: 20px;
            }}

            QScrollBar::handle:vertical:hover {{
                background: {COLORS['gold']};
            }}
        """)


def main():
    """Launch application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SIAMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
