import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QHBoxLayout,
    QFrame,
    QFileDialog,
    QDialog,
    QTextEdit,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtGui import QFont, QIcon, QPainter, QColor, QPen, QGuiApplication
from PyQt5.QtCore import Qt, QRectF, QObject, QEvent, QPropertyAnimation, QRect
from src.model.model import main, get_app_dir


def resource_path(relative_path):
    """Always returns the path to the external data directory."""
    # If you ever bundle with --onefile and still want external data,
    # ignore sys._MEIPASS entirely:
    base = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base, relative_path)


class InfoDialog(QDialog):
    def __init__(self, parent=None, title="Information"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedHeight(150)

        # self.resize(400, 300)
        layout = QVBoxLayout(self)

        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.text_area)
        
        self._set_line_spacing(120)
        self.text_area.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;        /* match your window style */
                color: #ffffff;                   /* message text color */
                font-family: 'Capriola';          /* font family */
                font-size: 9pt;                  /* font size in points */
                padding: 6px;                     /* inner padding */
            }
            /* The scroll bar groove (background track) */
            QScrollBar:vertical {
                background: #2c2c2c;             /* track background */          
                width: 8px;
                margin: 0px 0px 0px 0px;
                border-radius: 4px;
            }
            /* The draggable handle */
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #706c61;             /* normal handle color */
                min-height: 4px;
                max-height: 4px;
                border-radius: 4px;
            }
            /* Handle hover state */
            QScrollBar::handle:vertical:hover {
                background: #887f74;             /* handle on hover */              
            }
            /* Sub-page (above the handle) and add-page (below the handle) */
            QScrollBar::sub-page:vertical, QScrollBar::add-page:vertical {
                background: #3a3a3a;             /* area outside handle */  
                border-radius: 4px
            }
            /* The up/down arrow buttons (optional) */
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                background: none;
                height: 4px;
            }
            /* Intersection corner when both scrollbars present */
            QScrollBar::up-arrow, QScrollBar::down-arrow {
                background: none;
            }
            QScrollBar::add-line:vertical {
                subcontrol-position: bottom;
            }
            QScrollBar::sub-line:vertical {
                subcontrol-position: top;
            }
            QScrollBar::corner {
                background: #2c2c2c;             /* the little square */             /* :contentReference[oaicite:4]{index=4} */
            }
            
            
            QScrollBar:horizontal {
                background: #2c2c2c;              /* track background */          
                height: 8px;                      /* thickness of the bar */      
                margin: 0px 0px 0px 0px;
                border-radius: 4px;               /* matches your vertical radius */
            }
            /* 2) Draggable handle (thumb) */
            QScrollBar::handle:horizontal {
                background: #706c61;              /* normal handle color */      
                min-width: 20px;                  /* minimum handle length */      
                border-radius: 4px;               /* keep it rounded */
            }
            /* 3) Hover state for handle */
            QScrollBar::handle:horizontal:hover {
                background: #887f74;              /* handle on hover */            
            }
            /* 4) Page areas (left/right of the handle) */
            QScrollBar::sub-page:horizontal, QScrollBar::add-page:horizontal {
                background: #3a3a3a;              /* area outside handle */      
                border-radius: 4px;
            }
            /* 5) Hide arrow buttons */
            QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {
                background: none;
                width: 0px;                       /* zero‐width to remove */
            }
            /* 6) Corner between scrollbars */
            QScrollBar::corner {
                background: #2c2c2c;              /* the little empty square */   
            }
        """)

    def append_message(self, msg: str):             
        self.text_area.append(msg)
        self.text_area.moveCursor(self.text_area.textCursor().End)

    def _set_line_spacing(self, percent: int):
        """Set line spacing as a percentage of the default line height."""
        from PyQt5.QtGui import QTextBlockFormat, QTextCursor
        block_fmt = QTextBlockFormat()
        block_fmt.setLineHeight(percent, QTextBlockFormat.ProportionalHeight)
        cursor = self.text_area.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.mergeBlockFormat(block_fmt)
        self.text_area.setTextCursor(cursor)

class ForecastItemApp(QWidget):
    def __init__(self):
        super().__init__()
        self.info_dialog = InfoDialog(self, "Status Messages")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setGeometry(0, 0, 960, 700)

        self.initUI()
        self.move_to_center()
        # Enable dragging the window only by the title bar using eventFilter method
        self.titleBar.installEventFilter(self)
        
        self.import_btn.clicked.connect(self.open_file_dialog)
        self.forecast_btn.clicked.connect(self.apply_models)

    def initUI(self):
        self.create_title_bar()
        self.add_title()
        
        self.add_box()
        # self.add_inner_title()
        self.add_search_bar()
        self.add_regions()
        self.add_time_period()
        self.add_buttons()
        self.box_layout.addWidget(self.info_dialog)

        # 2. Main layout for the window
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.main_layout.addWidget(self.titleBar)
        self.main_layout.addWidget(self.title)
        # self.main_layout.addSpacing(40)
        self.main_layout.addWidget(self.box_container)
        self.main_layout.addStretch()

    def create_title_bar(self):
        # Create the title bar
        self.titleBar = QWidget(self)

        self.titleBar.setObjectName("TitleBar")
        self.titleBar.setFixedHeight(100)
        h = QHBoxLayout(self.titleBar)
        h.setContentsMargins(30, 30, 30, 30)

        # Icon
        iconLabel = QLabel(self.titleBar)
        iconPixmap = QIcon(resource_path("icon.png")).pixmap(100, 100)
        iconLabel.setPixmap(iconPixmap)
        h.addWidget(iconLabel)
        h.addStretch()

        # Control buttons
        for sym in ("—", "✕"):
            btn = QPushButton(sym, self.titleBar)
            btn.setFixedSize(40, 40)
            btn.setStyleSheet(
                """                              
                border: 2px solid #F3E09F;                              
                border-radius: 20px;                   
                color: #ffffff;                               
                font-size: 20px;                              
            """
            )

            btn.clicked.connect(
                lambda _, s=sym: getattr(
                    self, {"—": "showMinimized", "✕": "close"}[s]
                )()
            )
            h.addSpacing(20)
            h.addWidget(btn)

    def add_title(self):
        self.title = QLabel("FORECAST ITEM")
        self.title.setFont(QFont("Bungee Inline", 24, QFont.Bold))
        self.title.setStyleSheet("color: #ffffff")
        self.title.setAlignment(Qt.AlignCenter)

    def add_box(self):
        box = QFrame(self)
        box.setFrameShape(QFrame.StyledPanel)
        box.setFrameShadow(QFrame.Plain)
        box.setStyleSheet(
            """
            QFrame {
                border: 2px solid #8D825C;
                border-radius: 8px;
                background-color: #1e1e1e;
            }
        """
        )

        self.box_layout = QVBoxLayout(box)
        self.box_layout.setContentsMargins(20, 20, 20, 20)
        self.box_layout.setSpacing(20)

        self.box_container = QWidget(self)
        container_layout = QVBoxLayout(self.box_container)
        container_layout.setContentsMargins(
            140, 20, 140, 40
        )  # outer padding just for this frame
        container_layout.addWidget(box)

    def add_inner_title(self):
        title = QLabel("ITEM")
        font = QFont("Bungee", 18)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 6)
        title.setFont(font)
        title.setStyleSheet("color: white; border: None;")
        title.setContentsMargins(0, 0, 0, 20)
        title.setAlignment(Qt.AlignCenter)
        self.box_layout.addWidget(title)

    def add_search_bar(self):
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search by ID or name")
        search_bar.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #8D825C;
                border-radius: 8px;
                background-color: #1e1e1e;
                color: #ffffff;
                width: 500;
                height: 30;
            }
        """
        )
        search_bar.setAlignment(Qt.AlignCenter)
        self.box_layout.addWidget(search_bar, alignment=Qt.AlignHCenter)
        search_bar.setEnabled(False)

    def add_regions(self):
        container = QWidget(self)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(54, 15, 56, 10)

        region_label = QLabel("Select the region")
        font = QFont("Capriola", 10)
        region_label.setFont(font)
        region_label.setStyleSheet("color: white; border: None;")

        region_dropdown = QComboBox()
        region_dropdown.addItems(
            ["Egypt", "Region 2", "Region 3"]
        )  # Example regions
        region_dropdown.setStyleSheet(
            """
                                      QComboBox {
                                          border: 1px solid #8D825C;
                                          border-radius: 8px;
                                          background-color: #2c2c2c;
                                          color: #ffffff;
                                          font-size: 14px;
                                          height: 25px
                                          
                                      }
                                      QComboBox::drop-down {
                                          border-left: 0px solid #8D825C;
                                      }
                                      """
        )
        region_dropdown.view().setStyleSheet(
            """
                                            /* Popup background and border */
                                            background-color: #3a3a3a;
                                            border: 1px solid #8D825C;
                                            border-radius: 8px;
                                            padding: 2px;

                                            /* Item text and hover/selection */
                                            color: #ffffff;
                                            """
        )

        vbox.addWidget(region_label)
        vbox.addWidget(region_dropdown)

        self.box_layout.addWidget(container)
        region_dropdown.setEnabled(False)

    def add_time_period(self):
        container = QWidget(self)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(54, 15, 56, 10)

        time_label = QLabel("Time period")
        font = QFont("Capriola", 10)
        time_label.setFont(font)
        time_label.setStyleSheet("color: white; border: None;")

        time_dropdown = QComboBox()
        time_dropdown.addItems(["Next Month", "Next Quarter"])
        time_dropdown.setStyleSheet(
            """
                                      QComboBox {
                                          border: 1px solid #8D825C;
                                          border-radius: 8px;
                                          background-color: #2c2c2c;
                                          color: #ffffff;
                                          font-size: 14px;
                                          height: 25px;
                                          
                                      }
                                      QComboBox::drop-down {
                                          border-left: 0px solid #8D825C;
                                      }
                                      """
        )
        time_dropdown.view().setStyleSheet(
            """
                                            /* Popup background and border */
                                            background-color: #3a3a3a;
                                            border: 1px solid #8D825C;
                                            border-radius: 8px;
                                            padding: 2px;

                                            /* Item text and hover/selection */
                                            color: #ffffff;
                                            """
        )

        vbox.addWidget(time_label)
        vbox.addWidget(time_dropdown)

        self.box_layout.addWidget(container)
        time_dropdown.setEnabled(False)

    def add_buttons(self):
        container = QWidget(self)
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(44, 15, 66, 10)

        font = QFont("Capriola", 18, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 2)

        self.import_btn = QPushButton("Import")
        self.import_btn.setFont(font)
        self.import_btn.setFixedHeight(40)
        self.import_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #706c61;
                color: #fff;
                border-radius: 20px;
                font-size: 22px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #887f74;
            }
        """
        )

        self.forecast_btn = QPushButton("Forecast")
        self.forecast_btn.setFont(font)
        self.forecast_btn.setFixedHeight(40)
        self.forecast_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #706c61;
                color: #fff;
                border-radius: 20px;
                font-size: 22px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #887f74;
            }
        """
        )

        hbox.addWidget(self.import_btn)
        hbox.addWidget(self.forecast_btn)
        hbox.setSpacing(50)
        self.box_layout.addWidget(container)
        self.forecast_btn.setEnabled(False)

    def open_file_dialog(self):
        # 1. Compute the script’s folder

        base_dir = get_app_dir()

        # 2. Launch the native dialog, filtered to CSV/XLSX
        self.file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            base_dir,
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx)",
        )
        if self.file_path:
            self.info_dialog.append_message(f"Uploaded File: {self.file_path}")
            self.info_dialog.append_message("-"*114)
            self.forecast_btn.setEnabled(True)
            self.import_btn.setEnabled(False)

    def apply_models(self):
        self.forecast_btn.setEnabled(False)
        main(self.file_path)
        self.import_btn.setEnabled(True)
        
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Only handle events for the titleBar
        if watched == self.titleBar:
            # Mouse pressed on titleBar
            if (
                event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton
            ):
                # Store offset of click relative to the window top-left
                self._dragPos = event.globalPos() - self.frameGeometry().topLeft()
                return True  # eat the event (no further processing)

            # Mouse moved while button held down on titleBar
            elif (
                event.type() == QEvent.MouseMove
                and event.buttons() & Qt.LeftButton
                and self._dragPos
            ):
                # Move window by the delta
                self.move(event.globalPos() - self._dragPos)
                return True  # eat the event

            # Mouse released: reset drag state
            elif event.type() == QEvent.MouseButtonRelease and self._dragPos:
                self._dragPos = None
                return True

        # For all other cases, use default processing
        return super().eventFilter(watched, event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw rounded background
        rect = QRectF(0, 0, self.width(), self.height())
        bg = QColor(27, 27, 27, 255)  # fully opaque dark fill
        painter.setBrush(bg)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 15, 15)

        # Draw the stroke
        stroke = QPen(QColor("#F3E09F"))
        stroke.setWidth(2)
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        inset = stroke.width() / 2
        r2 = QRectF(
            inset, inset, self.width() - stroke.width(), self.height() - stroke.width()
        )
        painter.drawRoundedRect(r2, 15, 15)

    def move_to_center(self):
        screen = QGuiApplication.primaryScreen()
        screen_geom = screen.availableGeometry()
        # Calculate centered top-left point
        x = screen_geom.x() + (screen_geom.width() - self.width()) // 2
        y = screen_geom.y() + (screen_geom.height() - self.height()) // 2
        self.move(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ForecastItemApp()
    window.setWindowTitle("Forecast")
    window.setWindowIcon(QIcon(resource_path("icon.png")))
    # window.resize(600, 400)
    window.show()
    sys.exit(app.exec_())