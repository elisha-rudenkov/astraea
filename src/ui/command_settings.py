from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QGuiApplication

class InfoOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.init_ui()


    def init_ui(self):
        master_layout = QVBoxLayout()

        # Title bar at the top
        self.header_label = QLabel("Command Maker", self)
        self.header_label.setStyleSheet("""
            color: white;
            background-color: rgba(22,23,51,240);
            border-radius: 12px;
            border: 3px solid #43459a;
            font-weight: bold;
        """)
        self.header_label.setFont(QFont("Arial", 16))
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        self.header_label.setMaximumHeight(60)


        # Two-column layout below the title
        content_layout = QHBoxLayout()

        # Left info area
        self.left_label = QLabel("Loading data...", self)
        self.left_label.setStyleSheet("""
            color: white;
            background-color: rgba(22,23,51,240);
            padding: 30px 40px;
            border-radius: 12px;
            border: 3px solid #43459a;
        """)
        self.left_label.setFont(QFont("Arial", 14))
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Right info area
        self.right_label = QLabel("Additional info", self)
        self.right_label.setStyleSheet("""
            color: white;
            background-color: rgba(22,23,51,240);
            padding: 30px 40px;
            border-radius: 12px;
            border: 3px solid #43459a;
        """)
        self.right_label.setFont(QFont("Arial", 14))
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add to the horizontal layout
        content_layout.addWidget(self.left_label)
        content_layout.addWidget(self.right_label)

        # Assemble final layout
        master_layout.addWidget(self.header_label)
        master_layout.addLayout(content_layout)

        self.setLayout(master_layout)


    def show_overlay(self):
        screen = QGuiApplication.primaryScreen()
        geometry = screen.geometry()

        # Shrink the overlay by 10% on all sides
        margin_w = geometry.width() * 0.25
        margin_h = geometry.height() * 0.25

        reduced_geometry = geometry.adjusted(
            int(margin_w),     # left
            int(margin_h),     # top
            -int(margin_w),    # right
            -int(margin_h)     # bottom
        )

        self.setGeometry(reduced_geometry)
        self.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Resize header to be 10% of the widget's height
        new_height = int(self.height() * 0.15)
        self.header_label.setFixedHeight(new_height)

        # Scale font size based on height (you can tweak the ratio)
        font_size = max(10, int(new_height * 0.35))
        font = QFont("Segoe UI", font_size)
        self.header_label.setFont(font)
        


    def hide_overlay(self):
        self.hide()
