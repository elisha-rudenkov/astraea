import sys, os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QToolBar, QTextEdit, QWidget, QSlider,
                              QLabel, QFormLayout, QVBoxLayout, QPushButton, QHBoxLayout, 
                              QTabWidget, QStackedWidget, QGraphicsRectItem, QGraphicsScene,
                              QGraphicsView, QGraphicsTextItem, QGraphicsProxyWidget, QScrollArea, 
                              QGroupBox, QHBoxLayout, QFrame, QSizePolicy, QGraphicsEllipseItem)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QFont, QFontDatabase, QPixmap, QMouseEvent, QPalette
from PyQt6 import QtCore, QtWidgets, QtGui

# Custom proxy style to adjust the slider appearance
class SliderProxyStyle(QtWidgets.QProxyStyle):
    def pixelMetric(self, metric, option, widget):
        if metric == QtWidgets.QStyle.PixelMetric.PM_SliderThickness:
            return 100
        elif metric == QtWidgets.QStyle.PixelMetric.PM_SliderLength:
            return 80
        return super().pixelMetric(metric, option, widget)

# Custom graphics rectangle with specified dimensions and color
class CustomGraphicsItem(QGraphicsRectItem):
    def __init__(self, width, height, color, border_radius=10, border_color="#C5CFF4", border_width=2):
        super().__init__()
        self.setRect(0, 0, width, height)
        self.color = QColor(color)
        self.border_radius = border_radius
        self.border_color = QColor(border_color)
        self.border_width = border_width
        
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self.color)
        
        # Set the pen for the border (instead of NoPen)
        painter.setPen(QPen(self.border_color, self.border_width))
        
        painter.drawRoundedRect(self.rect(), self.border_radius, self.border_radius)

# Overlay window to display live transcription
class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
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

        # Set size and position
        self.setGeometry(0, 50, 400, 80)
        desktop = QApplication.primaryScreen().geometry()
        self.move(desktop.width() - self.width() - 20, 20)

    def paintEvent(self, event):
        # Custom paint event to draw the overlay rectangle
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw semi-transparent background
        painter.setBrush(QColor(40, 40, 40, 180))  # Dark gray with alpha
        painter.setPen(QPen(QColor(60, 60, 220), 2))  # Blue border
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 10, 10)

        # Draw text (Change to live transcription - WIP)
        painter.setPen(QColor(255, 255, 255))  # White text
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(10, 30, "Transcription line 1")
        painter.drawText(10, 60, "Transcription line 2")

# Create a custom titlebar class
class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Style the titlebar
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Window)
        self.setMaximumHeight(40)
        self.setStyleSheet("""
            background-color: transparent;
            color: white;
        """)
        
        # Add logo on the left
        self.logo = QLabel()
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the logo
        logo_path = os.path.join(current_dir, 'logo.png')
        self.logo = QLabel()
        self.logo.setPixmap(QPixmap(logo_path).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.logo.setStyleSheet("margin-left: 10px; margin-right: 10px;")
        self.layout.addWidget(self.logo)
        
        # Add title
        self.title = QLabel("ASTREA")
        self.title.setStyleSheet("font-size: 20px; color: #A099ED;")
        font = QFont("Constantia")
        self.title.setFont(font)
        self.layout.addWidget(self.title)
        
        # Add spacer
        self.layout.addStretch()
        
        # Add minimize button
        self.min_button = QPushButton("─")
        self.min_button.setFixedSize(40, 40)
        self.min_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #3D4F84;
            }
        """)
        self.min_button.clicked.connect(self.show_minimized)
        self.layout.addWidget(self.min_button)
        
        # Add maximize/restore button
        self.max_button = QPushButton("□")
        self.max_button.setFixedSize(40, 40)
        self.max_button.setStyleSheet(self.min_button.styleSheet())
        self.max_button.clicked.connect(self.toggle_maximize_restore)
        self.layout.addWidget(self.max_button)
        
        # Add close button
        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(40, 40)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #e81123;
            }
        """)
        self.close_button.clicked.connect(self.close_window)
        self.layout.addWidget(self.close_button)
        
        # For window dragging
        self.mouse_pos = None
        
    def show_minimized(self):
        self.parent.showMinimized()
    
    def toggle_maximize_restore(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.max_button.setText("□")
        else:
            self.parent.showMaximized()
            self.max_button.setText("❐")
    
    def close_window(self):
        self.parent.close()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pos = event.globalPosition().toPoint()
    
    def mouseMoveEvent(self, event):
        if self.mouse_pos:
            delta = event.globalPosition().toPoint() - self.mouse_pos
            self.parent.move(self.parent.x() + delta.x(), self.parent.y() + delta.y())
            self.mouse_pos = event.globalPosition().toPoint()

# Main window for Astrea
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add references to store the MouseController instance and overlay window
        self.mouse_controller = None
        self.overlay_window = None
        # Dictionaries to store slider and label references
        self.sliders = {}
        self.value_labels = {}

        self.init_ui()

        # Create overlay window
        self.overlay_window = OverlayWindow()
        self.overlay_window.show()

    def register_mouse_controller(self, controller):
        #Register the mouse controller to allow adjusting its settings
        self.mouse_controller = controller
        # Initialize sliders with current value
        self.update_sliders_from_controller()

    def update_sliders_from_controller(self):
        if not self.mouse_controller:
            return
        
        # Update base speed slider
        if 'base_speed' in self.sliders:
            self.sliders['base_speed'].setValue(int(self.mouse_controller.base_speed))

        # Update max speed slider
        if 'max_speed' in self.sliders:
            self.sliders['max_speed'].setValue(int(self.mouse_controller.max_speed))

        # Update acceleration slider
        if 'exp_factor' in self.sliders:
            value = int(self.mouse_controller.exp_factor * 10)
            self.sliders['exp_factor'].setValue(value)

        # Update vertical sensitivity sliders
        for multiplier in ['up_multiplier', 'down_multiplier']:
            if multiplier in self.sliders:
                value = int(getattr(self.mouse_controller, multiplier))
                self.sliders[multiplier].setValue(value)

        # Update threshold sliders
        if 'movement_threshold' in self.sliders:
            value = int(self.mouse_controller.movement_threshold)
            self.sliders['movement_threshold'].setValue(value)

        if 'click_threshold' in self.sliders:
            value = int(self.mouse_controller.click_threshold)
            self.sliders['click_threshold'].setValue(value) 

        # Update click cooldown slider
        if 'click_cooldown' in self.sliders:
            value = int(self.mouse_controller.click_cooldown * 10)
            self.sliders['click_cooldown'].setValue(value) 

    def set_video_frame(self, frame):
        # Update the video frame in the UI
        if not hasattr(self, 'scene_video_label'):
            return
            
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        self.scene_video_label.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def register_calibrate_callback(self, callback):
        # Register callback for calibration button
        self.calibrate_button.clicked.disconnect()
        self.calibrate_button.clicked.connect(callback)
    
    def update_calibration_status(self, is_calibrated, values=None):
        # Update the calibration status display
        if is_calibrated:
            self.calibration_status_label.setText("Calibrated")
            self.calibration_status_label.setStyleSheet("""
                background-color: #4CAF50; 
                color: white;
                border-radius: 10px;
                border_radius=10
                padding: 10px;
                font-size: 16pt;
                font-weight: bold;
            """)
        else:
            self.calibration_status_label.setText("Not Calibrated")
            self.calibration_status_label.setStyleSheet("""
                background-color: #FFA500; 
                color: white;
                border-radius: 10px;
                border_radius=10
                padding: 10px;
                font-size: 16pt;
                font-weight: bold;
            """)
    
    def init_ui(self):
        # Initialize the user interface components
        self.setWindowTitle('Astrea')
        self.setMinimumWidth(200)

        # Hide the default titlebar
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Create a main container widget
        container = QWidget()
        container.setStyleSheet("background-color: #090A19;")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Create and add custom titlebar
        self.title_bar = CustomTitleBar(self)
        container_layout.addWidget(self.title_bar)
        
        # Create the separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(10)
        separator.setStyleSheet("background-color:#3D4F84;")
        container_layout.addWidget(separator)

        # Create a widget to hold the menu bar
        menu_widget = QWidget()
        menu_widget.setStyleSheet("background-color: #161833;")
        menu_layout = QVBoxLayout(menu_widget)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(0)
        
        # Configure and add menu bar to the menu widget
        self.setup_menu_bar()
        menu_bar = self.menuBar()
        menu_layout.addWidget(menu_bar)
        
        # Add the menu widget to the container
        container_layout.addWidget(menu_widget)
        
        # Create central widget for content
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #A0A7CC;")
        
        # Create stacked widget for multiple pages
        self.stacked_widget = QStackedWidget()
        
        # Create a layout for the content widget and add the stacked widget to it
        content_layout = QVBoxLayout(content_widget)
        content_layout.addWidget(self.stacked_widget)

        # Add the content widget to the container layout
        container_layout.addWidget(content_widget)
        
        # Set the container as the central widget
        self.setCentralWidget(container)

        # Create pages
        self.create_home_page()
        self.create_settings_page()
        self.create_help_page()

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.help_page)

        # Show the window
        self.showMaximized()

    def create_home_page(self):
        # Create the home page and add components
        self.home_page = QWidget()
        home_layout = QFormLayout()
        self.home_page.setLayout(home_layout)

        # Create graphics scene for visualization
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)  # Removes the frame

        # Add video feed box
        video_box = CustomGraphicsItem(750, 595, "#B6BEDF", border_radius=10)
        video_box.setPos(100, 100)
        scene.addItem(video_box)

        # Create a QLabel for the video feed in the scene
        self.scene_video_label = QLabel()
        self.scene_video_label.setFixedSize(700, 500)
        self.scene_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene_video_label.setStyleSheet("background-color: transparent;")
        
        # Create a proxy widget to add the label to the scene
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(self.scene_video_label)
        
        # Position the video label inside the video box
        proxy.setPos(video_box.pos() + QtCore.QPointF(25, 50))
        scene.addItem(proxy)

        # Add video feed label
        video_label = QGraphicsTextItem("Video Feed")
        video_label.setDefaultTextColor(Qt.GlobalColor.black)
        video_label_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        video_label.setFont(video_label_font)

        cmd_title = QGraphicsTextItem("Voice Commands")
        cmd_title.setDefaultTextColor(Qt.GlobalColor.black)
        cmd_title_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        cmd_title.setFont(cmd_title_font)

        # Position the label at the top center of the video feed box
        label_x = video_box.x() + (video_box.rect().width() - video_label.boundingRect().width()) / 2
        label_y = video_box.y() + 10
        video_label.setPos(label_x, label_y)
        scene.addItem(video_label)

        # Create calibration button
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setFont(QFont("Lucida Sans"))
        self.calibrate_button.setFixedSize(150, 50)
        self.calibrate_button.clicked.connect(self.on_button_clicked)
        self.calibrate_button.setStyleSheet(self.get_button_style())

        # Create a proxy widget for the button
        calib_button_proxy = QGraphicsProxyWidget()
        calib_button_proxy.setWidget(self.calibrate_button)
        calib_button_proxy.setPos(550, 735)

        self.calibrate_button.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        calib_button_proxy.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        scene.addItem(calib_button_proxy)

        # Create overlay toggle button
        self.overlay_button = QPushButton("Toggle Transcription Overlay")
        self.overlay_button.setFont(QFont("Lucida Sans"))
        self.overlay_button.setFixedSize(350, 50)
        self.overlay_button.clicked.connect(self.toggle_overlay)
        self.overlay_button.setStyleSheet(self.get_button_style())

        # Create a proxy widget for the button
        overlay_b_proxy = QGraphicsProxyWidget()
        overlay_b_proxy.setWidget(self.overlay_button)
        overlay_b_proxy.setPos(150, 735)

        self.overlay_button.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        overlay_b_proxy.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        scene.addItem(overlay_b_proxy)

        # Add voice commands box
        voice_cmd_box = CustomGraphicsItem(500, 600, "#B6BEDF", border_radius=10)
        voice_cmd_box.setPos(900, 0)
        scene.addItem(voice_cmd_box)
        
        # Add label to voice commands box
        cmd_title = QGraphicsTextItem("Voice Commands")
        cmd_title.setDefaultTextColor(Qt.GlobalColor.black)
        cmd_title_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        cmd_title.setFont(cmd_title_font)

        # Position the title at the top center of the box
        cmd_title_x = voice_cmd_box.x() + (voice_cmd_box.rect().width() - cmd_title.boundingRect().width()) / 2
        cmd_title_y = voice_cmd_box.y() + 15
        cmd_title.setPos(cmd_title_x, cmd_title_y)
        scene.addItem(cmd_title)

        # List of voice commands
        commands = [
            "🗣️  \"Start listening\" – Activate voice commands",
            "🔇  \"Stop listening\" – Deactivate voice commands",
            "➡️  \"Right\" – Right click",
            "⬅️  \"Left\" – Left click",
            "✋  \"Hold\" – Hold down left click",
            "🖱️  \"Release\" – Release held click",
            "🎯  \"Calibrate\" – Calibrate the cursor",
            "⏸️  \"Pause mouse\" – Freeze cursor movement",
            "▶️  \"Resume mouse\" – Unfreeze cursor movement"
        ]

        start_x = voice_cmd_box.x() + 30
        start_y = voice_cmd_box.y() + 60
        line_spacing = 40

        command_font = QFont("Segoe UI", 14)

        for i, command in enumerate(commands):
            cmd_item = QGraphicsTextItem(command)
            cmd_item.setDefaultTextColor(Qt.GlobalColor.black)
            cmd_item.setFont(command_font)
            cmd_item.setPos(start_x, start_y + i * line_spacing)
            scene.addItem(cmd_item)

        # Add calibration status box
        calib_box = CustomGraphicsItem(500, 150, "#B6BEDF", border_radius=10)
        calib_box.setPos(900, 635)
        scene.addItem(calib_box)

        # Add calibration status title
        calib_title = QGraphicsTextItem("Calibration Status")
        calib_title.setDefaultTextColor(Qt.GlobalColor.black)
        calib_title.setFont(cmd_title_font)
        
        # Position the title at the top center of the calibration box
        calib_title_x = calib_box.x() + (calib_box.rect().width() - calib_title.boundingRect().width()) / 2
        calib_title_y = calib_box.y() + 10  # Margin from top of box
        calib_title.setPos(calib_title_x, calib_title_y)
        scene.addItem(calib_title)
        
        # Create calibration status label
        self.calibration_status_label = QLabel("Not Calibrated")
        self.calibration_status_label.setFont(QFont("Lucida Sans", 16, QFont.Weight.Bold))
        self.calibration_status_label.setFixedSize(300, 60)
        self.calibration_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_status_label.setStyleSheet("""
            background-color: #FFA500;
            color: white;
            border-radius: 10px;
            padding: 10px;
        """)
        
        # Create a proxy widget for the status label
        status_label_proxy = QGraphicsProxyWidget()
        status_label_proxy.setWidget(self.calibration_status_label)
        status_label_proxy.setPos(calib_box.x() + (calib_box.rect().width() - self.calibration_status_label.width()) / 2, 
                                calib_box.y() + 70)
        scene.addItem(status_label_proxy)

        home_layout.addRow(view)
    
    def create_settings_page(self):
        # Create the settings page and add components
        self.settings_page = QWidget()
        settings_layout = QVBoxLayout()
        self.settings_page.setLayout(settings_layout)

        # Create scroll bar
        scroll_bar = QScrollArea()
        scroll_bar.setWidgetResizable(True)

        # Create content widget for settings
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)

        scroll_bar.setWidget(content_widget)

        # Add scroll bar 
        settings_layout.addWidget(scroll_bar)

        # Create group box for cursor speed settings
        speed_group = QGroupBox("Cursor Speed Settings")
        speed_group.setStyleSheet("QGroupBox { font-size: 16pt; font-weight: bold; color: #162552; border: 2px solid #B6BEDF; padding-top: 15px; margin-top: 10px; }")
        speed_layout = QVBoxLayout()
        speed_group.setLayout(speed_layout)
        content_layout.addWidget(speed_group)

        # Add base speed slider
        base_speed_label, base_speed_row, base_speed_value = self.create_slider_with_label(
            "Base Speed", 1, 20, 8, "base_speed", 
            lambda value: self.update_controller_value("base_speed", value, float)
        )
        speed_layout.addWidget(base_speed_label)
        speed_layout.addWidget(base_speed_row)
        
        # Add max speed slider
        max_speed_label, max_speed_row, max_speed_value = self.create_slider_with_label(
            "Max Speed", 10, 100, 40, "max_speed", 
            lambda value: self.update_controller_value("max_speed", value, float)
        )
        speed_layout.addWidget(max_speed_label)
        speed_layout.addWidget(max_speed_row)
        
        # Add acceleration slider
        accel_label, accel_row, accel_value = self.create_slider_with_label(
            "Acceleration", 5, 30, 15, "exp_factor", 
            lambda value: self.update_controller_value("exp_factor", value/10, float)
        )
        speed_layout.addWidget(accel_label)
        speed_layout.addWidget(accel_row)
        
        # Create group box for vertical sensitivity settings
        vert_group = QGroupBox("Vertical Sensitivity Settings")
        vert_group.setStyleSheet("QGroupBox { font-size: 16pt; font-weight: bold; color: #162552; border: 2px solid #B6BEDF; padding-top: 15px; margin-top: 10px; }")
        vert_layout = QVBoxLayout()
        vert_group.setLayout(vert_layout)
        content_layout.addWidget(vert_group)
        
        # Add up multiplier slider
        up_label, up_row, up_value = self.create_slider_with_label(
            "Upward Sensitivity", 5, 30, 20, "up_multiplier", 
            lambda value: self.update_controller_value("up_multiplier", value/10, float)
        )
        vert_layout.addWidget(up_label)
        vert_layout.addWidget(up_row)
        
        # Add down multiplier slider
        down_label, down_row, down_value = self.create_slider_with_label(
            "Downward Sensitivity", 5, 30, 20, "down_multiplier", 
            lambda value: self.update_controller_value("down_multiplier", value/10, float)
        )
        vert_layout.addWidget(down_label)
        vert_layout.addWidget(down_row)
        
        # Create group box for threshold settings
        threshold_group = QGroupBox("Threshold Settings")
        threshold_group.setStyleSheet("QGroupBox { font-size: 16pt; font-weight: bold; color: #162552; border: 2px solid #B6BEDF; padding-top: 15px; margin-top: 10px; }")
        threshold_layout = QVBoxLayout()
        threshold_group.setLayout(threshold_layout)
        content_layout.addWidget(threshold_group)
        
        # Add movement threshold slider
        movement_label, movement_row, movement_value = self.create_slider_with_label(
            "Movement Threshold (degrees)", 1, 20, 5, "movement_threshold", 
            lambda value: self.update_controller_value("movement_threshold", value, float)
        )
        threshold_layout.addWidget(movement_label)
        threshold_layout.addWidget(movement_row)
        
        # Add click threshold slider
        click_label, click_row, click_value = self.create_slider_with_label(
            "Click Threshold (degrees)", 1, 30, 10, "click_threshold", 
            lambda value: self.update_controller_value("click_threshold", value, float)
        )
        threshold_layout.addWidget(click_label)
        threshold_layout.addWidget(click_row)
        
        # Add click cooldown slider
        cooldown_label, cooldown_row, cooldown_value = self.create_slider_with_label(
            "Click Cooldown (seconds)", 0, 5, 1, "click_cooldown", 
            lambda value: self.update_controller_value("click_cooldown", value/10, float)
        )
        threshold_layout.addWidget(cooldown_label)
        threshold_layout.addWidget(cooldown_row)
        
        # Add spacer at the bottom
        content_layout.addSpacing(20)
        
        # Add reset button at the bottom
        reset_button = QPushButton("Reset to Default Settings")
        reset_button.setFixedSize(300, 50)
        reset_button.setFont(QFont("Lucida Sans"))
        reset_button.setStyleSheet(self.get_button_style())
        reset_button.clicked.connect(self.reset_to_default)
        content_layout.addWidget(reset_button)

        # Save the base speed slider for backwards compatibility
        self.sensitivity_slider = self.sliders.get('base_speed')

    def create_slider_with_label(self, label_text, min_val, max_val, default_val, key, callback_func):
        # Create label
        slider_label = QLabel(label_text)
        slider_label.setStyleSheet("font-size: 14pt; margin-top: 10px;")

        # Create horizontal layout for slider and value
        slider_row = QHBoxLayout()
        
        # Create slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setSingleStep(1)
        slider.setPageStep(2)
        slider.setTickPosition(QSlider.TickPosition.TicksAbove)
        slider.setMaximumWidth(600)
        
        # Apply custom style to slider
        style = SliderProxyStyle(slider.style())
        slider.setStyle(style)
        slider.setStyleSheet(self.get_slider_style())
        
        # Create value label
        value_label = QLabel(f"{label_text}: {default_val}")
        value_label.setStyleSheet("font-size: 12pt;")

        # Add slider and value label to the row layout
        slider_row.addWidget(slider, 85)
        slider_row.addWidget(value_label, 15)
        
        # Connect value change signal
        slider.valueChanged.connect(lambda value: self.update_slider_display(key, value, value_label, callback_func))
        
        # Store the slider and its value label
        self.sliders[key] = slider
        self.value_labels[key] = value_label

        # Create container widget for slider row
        row_widget = QWidget()
        row_widget.setLayout(slider_row)
        
        return slider_label, row_widget, value_label
    
    def update_slider_display(self, key, value, label, callback_func):
        # Updates the display value and calls the controller update function
        # Format the display value based on the key
        if key in ['exp_factor', 'up_multiplier', 'down_multiplier', 'click_cooldown']:
            display_value = value
            label.setText(f"{key.replace('_', ' ').title()}: {display_value:.1f}")
        else:
            label.setText(f"{key.replace('_', ' ').title()}: {value}")
            
        # Call the controller update function
        if callback_func:
            callback_func(value)

    def update_controller_value(self, property_name, value, conversion_func=None):
        # Updates a property value in the mouse controller
        if self.mouse_controller:
            if conversion_func:
                value = conversion_func(value)
            setattr(self.mouse_controller, property_name, value)
            print(f"Updated {property_name} to {value}")
            
    def reset_to_default(self):
        # Reset all sliders to their default values
        if self.mouse_controller:
            # Set default values
            self.mouse_controller.base_speed = 8.0
            self.mouse_controller.max_speed = 40.0
            self.mouse_controller.exp_factor = 1.5
            self.mouse_controller.up_multiplier = 2.0
            self.mouse_controller.down_multiplier = 2.0
            self.mouse_controller.movement_threshold = 5.0
            self.mouse_controller.click_threshold = 10.0
            self.mouse_controller.click_cooldown = 0.5
            
            # Update sliders to reflect default values
            self.update_sliders_from_controller()
    
    def create_help_page(self):
        # Create the help page and add components
        self.help_page = QWidget()
        help_layout = QVBoxLayout()
        self.help_page.setLayout(help_layout)
        
        # Create title label centered at the top
        title = QLabel("Getting Started")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 40px; font-weight: bold; margin: 20px 0 40px 0; color: #162552;")
        help_layout.addWidget(title)
        
        # Create container for the boxes
        boxes_container = QWidget()
        boxes_layout = QHBoxLayout()
        boxes_container.setLayout(boxes_layout)
        
        # Define the steps content
        steps = [
            ("1", "Position yourself in front of the camera", "#6661A1"),
            ("2", "Press calibration button or say \"calibrate\" to calibrate", "#3D4F84"),
            ("3", "Move your head to control the cursor", "#3A356F"),
            ("4", "Use voice commands for clicking and other functions", "#162552")
        ]
        
        # Create each step box
        for number, text, color in steps:
            # Create the box widget
            box = QWidget()
            box.setFixedWidth(300)
            box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            box.setStyleSheet(f"""
                background-color: #B6BEDF;
                border-radius: 15px;
                padding: 20px 15px;
                min-height: 150px;
                border: 2px solid #C5CFF4;
            """)
            
            box_layout = QVBoxLayout()
            box_layout.setContentsMargins(15, 15, 15, 15)
            box_layout.setSpacing(15)
            box.setLayout(box_layout)
            
            number_view = QGraphicsView()
            number_view.setFixedSize(100, 100)
            number_view.setStyleSheet("background: transparent; border: none;")
            number_view.setRenderHint(QPainter.RenderHint.Antialiasing)
            number_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            number_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            number_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Create the scene
            number_scene = QGraphicsScene()
            number_scene.setSceneRect(-40, -40, 80, 80)
            number_view.setScene(number_scene)

            # Create the circle centered in the scene
            circle = QGraphicsEllipseItem(-30, -30, 60, 60)
            circle.setBrush(QColor(color))
            circle.setPen(QPen(Qt.PenStyle.NoPen))
            number_scene.addItem(circle)

            # Add number text
            text_item = QGraphicsTextItem(number)
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            text_item.setFont(QFont("Lucida Sans", 24, QFont.Weight.Bold))

            # Position the text in the center of the circle
            text_rect = text_item.boundingRect()
            text_x = -text_rect.width() / 2
            text_y = -text_rect.height() / 2
            text_item.setPos(text_x, text_y)

            number_scene.addItem(text_item)
            box_layout.addWidget(number_view, 0, Qt.AlignmentFlag.AlignCenter)
            
            # Create step text
            text_label = QLabel(text)
            text_label.setWordWrap(True)
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet("font-size: 20px; margin-top: 15px; color: #000000; line-height: 1.4;")
            text_label.setFont(QFont("Fira Sans"))
            box_layout.addWidget(text_label)
            
            # Add spacer at the bottom to push content up
            box_layout.addStretch()
            
            # Add box to container with some margin
            boxes_layout.addWidget(box)
            if number != "4":  # Don't add spacing after the last box
                boxes_layout.addSpacing(20)
        
        # Add the boxes container to the main layout
        boxes_container.setStyleSheet("margin-top: 30px;")
        help_layout.addWidget(boxes_container)
        
        # Add stretch to push everything to the top and center
        help_layout.addStretch()

    def change_help_content(self, topic):
        # Change content based on selected topic and update button styles
        for button in self.topic_buttons:
            button.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 15px;
                    font-size: 16px;
                    border: none;
                    background-color: #CFDAE9;
                }
                QPushButton:hover {
                    background-color: #B4C3D7;
                }
                QPushButton:pressed {
                    background-color: #0288D1;
                }
            """)
        
        # Find the button for this topic and highlight it
        for button in self.topic_buttons:
            if button.text() == topic:
                button.setStyleSheet(button.styleSheet() +
                                    "QPushButton { background-color: #0288D1; color: white; }")
                
        # Set the selected page
        if topic == "Getting Started":
            self.help_content.setCurrentIndex(0)

    def create_getting_started_page(self):
        # Create the Getting Started page content
        pass
    
    def setup_menu_bar(self):
        # Configure menu bar with navigation actions
        menu_bar = self.menuBar()

        # Customize menu bar style
        large_font = QtGui.QFont()
        large_font.setPointSize(16)  # Adjust font size as needed
        menu_bar.setStyleSheet(self.get_menu_bar_style())

        # Home menu
        home_action = QAction('&Home', self)
        home_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        home_action.setFont(large_font)
        menu_bar.addAction(home_action)

        # Settings menu
        settings_action = QAction('&Settings', self)
        settings_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        settings_action.setFont(large_font)  # Apply large font to Settings too
        menu_bar.addAction(settings_action)

        # Help menu
        help_action = QAction('&Help', self)
        help_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        help_action.setFont(large_font)  # Apply large font to Help too
        menu_bar.addAction(help_action)

    def on_button_clicked(self):
        # Handle button click events
        sender = self.sender()
        button_text = sender.text()
        print(f"Button '{button_text}' clicked")
        
        # Get current sensitivity value
        current_value = self.sensitivity_slider.value()
        
        # Adjust sensitivity based on button
        if button_text == "+":
            new_value = min(current_value + 1, self.sensitivity_slider.maximum())
            self.sensitivity_slider.setValue(new_value)
        elif button_text == "-":
            new_value = max(current_value - 1, self.sensitivity_slider.minimum())
            self.sensitivity_slider.setValue(new_value)
        elif button_text == "Calibrate (C)":
            # This is handled by the registered callback
            pass
    
    def toggle_overlay(self):
        # Toggle the overlay window on button click
        if self.overlay_window.isVisible():
            self.overlay_window.close()
        else:
            self.overlay_window.show()

    def no_cam_err_msg(self):
        # Create a black pixmap
        pixmap = QPixmap(700, 500)
        pixmap.fill(QColor(0, 0, 0))

        # Use painter to draw on the pixmap
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 0, 0))
        painter.setFont(QFont('Arial', 20))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "No camera detected")

        painter.setFont(QFont('Arial', 16))
        rect = pixmap.rect()
        rect.translate(0, 40)  # Move down for second line
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Please connect a camera and restart the app")
        painter.end()

        # Update the video label
        if hasattr(self, 'scene_video_label'):
            self.scene_video_label.setPixmap(pixmap)
            
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Camera not detected or disconnected")

    def get_menu_bar_style(self):
        # Return CSS stylesheet for menu bar
        return """
            QMenuBar {
                min-height: 30px;
                font-size: 16pt;
                background-color: #162552;
            }
            QMenuBar::item {
                padding: 10px 20px;
                margin: 5px;
                background-color: transparent;
                color: #B8B8B8;
            }
            QMenuBar::item:selected {
                background-color: #3D4F84;
                color: white;
                border-radius: 5px;
            }
        """

    def get_button_style(self):
        # Return CSS stylesheet for control buttons
        return """
            QPushButton {
                background-color: #3A356F; /* Blue background */
                color: white;              /* White text */
                border-radius: 15px;       /* Rounded corners */
                font-weight: bold;         /* Bold text */
                font-size: 20px;           /* Larger text */
                border: none;              /* No border */
                text-align: center;        /* Center text horizontally */
                padding: 8px;              /* Add some padding */
            }
            QPushButton:hover {
                background-color: #6661A1; /* Lighter blue when hovering */
            }
            QPushButton:pressed {
                background-color: #2E295E; /* Darker blue when pressed */
            }
        """
    
    def get_slider_style(self):
        # Return CSS stylesheet for slider control
        return """
            QSlider::handle:horizontal {
                background: #FCD88B;
                width: 85px;
                height: 80px;
                margin: -30px 0px;
                border-radius: 40px;
            }
            QSlider::add-page:horizontal {
                background: #B6BEDF;            /* Color for the right side of the handle */
                border-radius: 10px;
                height: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #EDCB80;            /* Color for the left side of the handle */
                border-radius: 10px;
                height: 10px;
            }
            QSlider::groove:horizontal {
                height: 30px;                   /* Make the track/groove thicker */
                background: #d3d3d3;            /* Light gray background for the track */
                border-radius: 10px;            /* Rounded corners for the track */
            }
        """

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())