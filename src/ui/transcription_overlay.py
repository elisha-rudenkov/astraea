from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QApplication, QWidget
from collections import deque

class TextDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 400, 300)  # Set the window size
        self.text_lines = deque(maxlen=4)
        
        # Set window flags
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | # Always on top of other window
            Qt.WindowType.FramelessWindowHint |  # Removes window frame
            Qt.WindowType.Tool |                 # Window not shown on taskbar
            Qt.WindowType.BypassWindowManagerHint # Don't let window manager handle this window
        )

        # Set background as transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Make the cursor able to click through the overlay
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)

        self.setGeometry(0, 50, 400, 80)
        desktop = QApplication.primaryScreen().geometry()
        self.move(desktop.width() - self.width() - 20, 20)

    def update_text(self, new_text):
        """This function updates the text to display a different message."""
        self.text_lines.append(new_text)
        self.update()  # Call the update method to trigger a repaint

    def paintEvent(self, event):
        """Override paintEvent to draw the text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent background
        painter.setBrush(QColor(40, 40, 40, 180))  # Dark gray with alpha
        painter.setPen(QPen(QColor(60, 60, 220), 2))  # Blue border
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 10, 10)

        # Draw the updated text lines, starting from the bottom of the widget
        painter.setPen(QColor(255, 255, 255))  # White text
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        # Draw the text at the center of the widget
        y_offset = self.height() - 20  # Start at the bottom of the widget
        for text in reversed(self.text_lines):  # Draw from the most recent to the oldest
            painter.drawText(10, y_offset, text)
            y_offset -= 20  # Adjust vertical spacing between lines