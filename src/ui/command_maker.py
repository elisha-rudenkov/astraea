from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QGuiApplication, QPainter, QColor, QPen

class CommandMaker(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | # Always on top of other window
            Qt.WindowType.FramelessWindowHint |  # Removes window frame
            Qt.WindowType.Tool |                 # Window not shown on taskbar
            Qt.WindowType.BypassWindowManagerHint # Don't let window manager handle this window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.text_lines = []

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
        self.header_label.setFont(QFont("Segoe UI", 16))
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
        self.right_label = QLabel("...", self)

        for label in [self.left_label, self.right_label]:
            label.setFont(QFont("Segoe UI", 14))
            label.setStyleSheet("""
                color: white;
                background-color: rgba(22,23,51,240);
                border-radius: 12px;
                border: 3px solid #43459a;
                padding: 10px 5px 0px 5px;
            """)

            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setWordWrap(True)

            content_layout.addWidget(label, 1)

        # Assemble final layout
        master_layout.addWidget(self.header_label)
        master_layout.addLayout(content_layout)

        self.setLayout(master_layout)

    

    def show_overlay(self, isActivated : bool):

        if not isActivated:
            self.hide_overlay()
            return

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

    
    def update_walkthrough(self, info):
        self.left_label.setText(str(info))
    
    def update_answers(self, info):
        self.text_lines.append(str(info))

        new_text = '\n'.join(self.text_lines)

        self.right_label.setText(new_text)
