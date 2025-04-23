import sys
import random
import threading
import queue
import time
from collections import deque
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QApplication, QWidget

class RandomStringWorker:
    def __init__(self, callback):
        self.callback = callback
        self.thread = threading.Thread(target=self.generate_random_strings)
        self.running = True
        self.data_queue = queue.Queue()  # Thread-safe queue for data transfer

    def start(self):
        """Start the background thread."""
        self.thread.start()

    def stop(self):
        """Stop the background thread."""
        self.running = False
        self.thread.join()

    def generate_random_strings(self):
        """Generate random strings in a loop and put them in the queue."""
        while self.running:
            random_text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=10))
            self.data_queue.put(random_text)  # Put new string into the queue
            time.sleep(0.5)  # Sleep for 500 ms before generating the next string

class AutoPaintEventWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automatic paintEvent Example")
        self.setGeometry(100, 100, 400, 300)

        self.text_lines = deque(maxlen=2)

        # Initialize the worker and start the background thread
        self.worker = RandomStringWorker(self.update_text)
        self.worker.start()

        # Set up a timer to periodically check the queue and update the UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_queue_and_update)  # Periodically check for new data
        self.timer.start(100)  # Check every 100 ms

    def check_queue_and_update(self):
        """Check if there is new data in the queue and update the UI."""
        try:
            # Check if there is new data in the queue
            new_text = self.worker.data_queue.get_nowait()
            self.update_text(new_text)
        except queue.Empty:
            pass

    def update_text(self, new_text):
        """Update the text with a new random value from the worker thread."""
        self.text_lines.append(new_text)
        self.update()  # Request a repaint

    def paintEvent(self, event):
        """Override paintEvent to draw the updated text."""
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

        # Draw the lines starting from the bottom, ensuring they wrap inside the box
        y_offset = self.height() - 30  # Start at the bottom of the widget
        for text in reversed(self.text_lines):  # Draw from the most recent to the oldest
            painter.drawText(10, y_offset, text)
            y_offset -= 20  # Adjust vertical spacing between lines

    def closeEvent(self, event):
        """Ensure the background thread is stopped when closing the window."""
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoPaintEventWidget()
    window.show()
    sys.exit(app.exec())
