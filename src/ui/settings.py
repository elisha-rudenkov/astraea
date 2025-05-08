import sys, os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QToolBar, QTextEdit, QWidget, QSlider,
                              QLabel, QFormLayout, QVBoxLayout, QPushButton, QHBoxLayout, 
                              QTabWidget, QStackedWidget, QGraphicsRectItem, QGraphicsScene,
                              QGraphicsView, QGraphicsTextItem, QGraphicsProxyWidget, QScrollArea, 
                              QGroupBox, QHBoxLayout, QFrame, QSizePolicy, QGraphicsEllipseItem)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QFont, QFontDatabase, QPixmap, QMouseEvent, QPalette
from PyQt6 import QtCore, QtWidgets, QtGui

from typing import Optional

from .transcription_overlay import TextDisplayWidget
from .command_maker import CommandMaker

# Responsive layout helper class
class ResponsiveLayout:
    def __init__(self, base_width=1920, base_height=1080):
        """Initialize the responsive layout system with base resolution"""
        self.base_width = base_width
        self.base_height = base_height
        self.update_scale_factors()
        
    def update_scale_factors(self):
        """Update scale factors based on current screen resolution"""
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.width_scale = screen_size.width() / self.base_width
        self.height_scale = screen_size.height() / self.base_height
        
    def scale_pos(self, x, y):
        """Scale a position based on current screen size"""
        return int(x * self.width_scale), int(y * self.height_scale)
    
    def scale_size(self, width, height):
        """Scale a size based on current screen size"""
        return int(width * self.width_scale), int(height * self.height_scale)
    
    def scale_font_size(self, size):
        """Scale a font size based on current screen size"""
        # Using the smaller scale factor to ensure text remains readable
        scale = min(self.width_scale, self.height_scale)
        return int(size * scale)
    
    def scale_margin(self, margin):
        """Scale a margin or padding value"""
        scale = min(self.width_scale, self.height_scale)
        return int(margin * scale)

class CustomGraphicsItem(QGraphicsRectItem):
    def __init__(self, width, height, color, border_radius=10, border_color="#C5CFF4", border_width=2, responsive=None):
        super().__init__()
        
        if responsive:
            width, height = responsive.scale_size(width, height)
            border_radius = responsive.scale_margin(border_radius)
            border_width = responsive.scale_margin(border_width)
            
        self.setRect(0, 0, width, height)
        self.color = QColor(color)
        self.border_radius = border_radius
        self.border_color = QColor(border_color)
        self.border_width = border_width
        
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self.color)
        
        # Set the pen for the border
        painter.setPen(QPen(self.border_color, self.border_width))
        
        painter.drawRoundedRect(self.rect(), self.border_radius, self.border_radius)

# Custom proxy style to adjust the slider appearance
class SliderProxyStyle(QtWidgets.QProxyStyle):
    def __init__(self, style, responsive):
        super().__init__(style)
        self.responsive = responsive
        
    def pixelMetric(self, metric, option, widget):
        if metric == QtWidgets.QStyle.PixelMetric.PM_SliderThickness:
            return self.responsive.scale_size(100, 100)[0]
        elif metric == QtWidgets.QStyle.PixelMetric.PM_SliderLength:
            return self.responsive.scale_size(80, 80)[0]
        return super().pixelMetric(metric, option, widget)

# Create custom titlebar class
class CustomTitleBar(QWidget):
    def __init__(self, parent=None, responsive=None):
        super().__init__(parent)
        self.parent = parent
        self.responsive = responsive
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Style the titlebar
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Window)
        
        if responsive:
            height = responsive.scale_size(0, 40)[1]
            self.setMaximumHeight(height)
        else:
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
        
        logo_size = 32
        if responsive:
            logo_size = responsive.scale_size(32, 32)[0]
            
        self.logo.setPixmap(QPixmap(logo_path).scaled(logo_size, logo_size, 
                                                     Qt.AspectRatioMode.KeepAspectRatio, 
                                                     Qt.TransformationMode.SmoothTransformation))
        
        margin = 10
        if responsive:
            margin = responsive.scale_margin(10)
        self.logo.setStyleSheet(f"margin-left: {margin}px; margin-right: {margin}px;")
        self.layout.addWidget(self.logo)
        
        # Add title with responsive font size
        self.title = QLabel("ASTREA")
        font_size = 20
        if responsive:
            font_size = responsive.scale_font_size(20)
        self.title.setStyleSheet(f"font-size: {font_size}px; color: #A099ED;")
        font = QFont("Constantia")
        self.title.setFont(font)
        self.layout.addWidget(self.title)
        
        # Add spacer
        self.layout.addStretch()
        
        # Create responsive button size
        button_size = 60
        if responsive:
            button_size = responsive.scale_size(60, 40)[0]
            
        # Add minimize button
        self.min_button = QPushButton("−")
        self.min_button.setFixedSize(button_size, button_size)
        
        # Set responsive font size for buttons
        button_font_size = 16
        if responsive:
            button_font_size = responsive.scale_font_size(25)
            
        # Titlebar button style
        titlebar_button_style = f"""
            QPushButton {{
                background-color: transparent;
                color: white;
                border: none;
                font-size: {button_font_size}px;
                font-family: 'Segoe UI', Arial;
                font-weight: bold;
                padding-bottom: 15px;
            }}
        """
        
        # Style for minimize and maximize buttons
        self.min_button.setStyleSheet(f"""
            {titlebar_button_style}
            QPushButton:hover {{
                background-color: #3D4F84;
            }}
        """)
        self.min_button.clicked.connect(self.show_minimized)
        self.layout.addWidget(self.min_button)
        
        # Add maximize/restore button
        self.max_button = QPushButton("□")
        self.max_button.setFixedSize(button_size, button_size)
        self.max_button.setStyleSheet(self.min_button.styleSheet())
        self.max_button.clicked.connect(self.toggle_maximize_restore)
        self.layout.addWidget(self.max_button)
        
        # Add close button
        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(button_size, button_size)
        self.close_button.setStyleSheet(f"""
            {titlebar_button_style}
            QPushButton:hover {{
                background-color: #e81123;
            }}
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
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event):
        if self.mouse_pos:
            if event.buttons() == Qt.MouseButton.LeftButton:
                # Use native window move behavior
                if self.parent.windowHandle():
                    self.parent.windowHandle().startSystemMove()
                self.mouse_pos = None  # Reset after initiating move
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize_restore()

    def show_context_menu(self, global_pos):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Minimize", self.show_minimized)
        menu.addAction("Maximize/Restore", self.toggle_maximize_restore)
        menu.addAction("Close", self.close_window)
        menu.exec(global_pos)    

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
        
        # Initialize responsive layout helper
        self.responsive = ResponsiveLayout()
        
        # Connect resize event to update layout
        self.resizeEvent = self.on_resize

        self.init_ui()

    def on_resize(self, event):
        """Handle the resize event to update the responsive layout"""
        self.responsive.update_scale_factors()
        # Update layouts that need to be responsive
        super().resizeEvent(event)

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
        if hasattr(self, 'calibrate_button') and self.calibrate_button.receivers(self.calibrate_button.clicked) > 0:
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
        self.setMinimumWidth(self.responsive.scale_size(200, 0)[0])

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
        self.title_bar = CustomTitleBar(self, self.responsive)
        container_layout.addWidget(self.title_bar)
        
        # Create the separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(self.responsive.scale_margin(10))
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

        # Hidden window for making commands
        self.command_maker = CommandMaker()

        # Create overlay window
        self.transcription_box = TextDisplayWidget()
        self.transcription_box.show()

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

        # Calculate scene size based on screen size
        screen_rect = QApplication.primaryScreen().availableGeometry()
        scene_width = screen_rect.width() * 0.95
        scene_height = screen_rect.height() * 0.85
        
        # Calculate total available height
        available_height = scene_height * 0.80
        
        # Add voice commands box first
        voice_cmd_width = scene_width * 0.35
        voice_cmd_height = scene_height * 0.7
        voice_cmd_box = CustomGraphicsItem(voice_cmd_width, voice_cmd_height, "#B6BEDF", 
                                        border_radius=self.responsive.scale_margin(10))
        
        # Position voice command box
        voice_cmd_x = scene_width * 0.6
        voice_cmd_y = scene_height * 0.05
        voice_cmd_box.setPos(voice_cmd_x, voice_cmd_y)
        scene.addItem(voice_cmd_box)
        
        # Add voice commands label
        voice_label = QGraphicsTextItem("Voice Commands")
        voice_label.setDefaultTextColor(Qt.GlobalColor.black)
        font_size = self.responsive.scale_font_size(18)
        voice_label_font = QFont("Segoe UI", font_size, QFont.Weight.Bold)
        voice_label.setFont(voice_label_font)

        # Position the label at the top center of the voice commands box
        label_x = voice_cmd_box.x() + (voice_cmd_box.rect().width() - voice_label.boundingRect().width()) / 2
        label_y = voice_cmd_box.y() + self.responsive.scale_margin(10)
        voice_label.setPos(label_x, label_y)
        scene.addItem(voice_label)

        # Add calibration status box second
        calib_width = voice_cmd_width
        calib_height = scene_height * 0.2
        calib_box = CustomGraphicsItem(calib_width, calib_height, "#B6BEDF", 
                                    border_radius=self.responsive.scale_margin(10))
        
        # Position calibration box below voice commands box
        calib_x = voice_cmd_x
        calib_y = voice_cmd_box.y() + voice_cmd_height + self.responsive.scale_margin(30)
        calib_box.setPos(calib_x, calib_y)
        scene.addItem(calib_box)

        # Add calibration status title
        calib_title = QGraphicsTextItem("Calibration Status")
        calib_title.setDefaultTextColor(Qt.GlobalColor.black)
        calib_title.setFont(voice_label_font)
        
        # Position the title at the top center of the calibration box
        calib_title_x = calib_box.x() + (calib_box.rect().width() - calib_title.boundingRect().width()) / 2
        calib_title_y = calib_box.y() + self.responsive.scale_margin(10)  # Margin from top of box
        calib_title.setPos(calib_title_x, calib_title_y)
        scene.addItem(calib_title)
        
        # Add video feed box to fit between the top of voice commands and bottom of calibration
        # Calculate the vertical position and height for video feed
        # Set video box dimensions
        video_box_width = scene_width * 0.5
        video_box_height = scene_height * 0.75  # Increase height to nearly full scene

        # Center video box vertically
        video_box_x = scene_width * 0.05
        video_box_y = (scene_height - video_box_height) / 2

        video_box = CustomGraphicsItem(video_box_width, video_box_height, "#B6BEDF", 
                                    border_radius=self.responsive.scale_margin(10))
        video_box.setPos(video_box_x, video_box_y)
        scene.addItem(video_box)

        # Add video feed label
        video_label = QGraphicsTextItem("Video Feed")
        video_label.setDefaultTextColor(Qt.GlobalColor.black)
        video_label.setFont(voice_label_font)

        # Position the label at the top center of the video feed box
        label_x = video_box.x() + (video_box.rect().width() - video_label.boundingRect().width()) / 2
        label_y = video_box.y() + self.responsive.scale_margin(10)
        video_label.setPos(label_x, label_y)
        scene.addItem(video_label)

        # Adjust video feed label (QLabel) to fill the box
        video_label_width = video_box_width * 0.9
        video_label_height = video_box_height * 0.85  # Nearly fill box, allow some padding

        self.scene_video_label = QLabel()
        self.scene_video_label.setFixedSize(int(video_label_width), int(video_label_height))
        self.scene_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene_video_label.setStyleSheet("background-color: transparent;")

        # Create a proxy widget to add the QLabel to the scene
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(self.scene_video_label)

        # Center QLabel in video box
        proxy_x = video_box.x() + (video_box_width - video_label_width) / 2
        proxy_y = video_box.y() + video_box_height * 0.1
        proxy.setPos(proxy_x, proxy_y)
        scene.addItem(proxy)


        # Create calibration status label
        self.calibration_status_label = QLabel("Not Calibrated")
        status_font_size = self.responsive.scale_font_size(16)
        self.calibration_status_label.setFont(QFont("Lucida Sans", status_font_size, QFont.Weight.Bold))
        
        status_width, status_height = self.responsive.scale_size(300, 60)
        self.calibration_status_label.setFixedSize(status_width, status_height)
        self.calibration_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        border_radius = self.responsive.scale_margin(10)
        padding = self.responsive.scale_margin(10)
        self.calibration_status_label.setStyleSheet(f"""
            background-color: #FFA500;
            color: white;
            border-radius: {border_radius}px;
            padding: {padding}px;
        """)
        
        # Create a proxy widget for the status label
        status_label_proxy = QGraphicsProxyWidget()
        status_label_proxy.setWidget(self.calibration_status_label)
        
        # Position the status label
        status_x = calib_box.x() + (calib_box.rect().width() - status_width) / 2
        status_y = calib_box.y() + calib_box.rect().height() / 2
        status_label_proxy.setPos(status_x, status_y)
        scene.addItem(status_label_proxy)
        
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
            "▶️  \"Resume mouse\" – Unfreeze cursor movement",
            "⚙️  \"Create command\" – Open the command creator wizard"
        ]

        start_x = voice_cmd_box.x() + self.responsive.scale_margin(30)
        start_y = voice_cmd_box.y() + self.responsive.scale_margin(60)
        line_spacing = self.responsive.scale_margin(40)

        command_font_size = self.responsive.scale_font_size(14)
        command_font = QFont("Segoe UI", command_font_size)

        for i, command in enumerate(commands):
            cmd_item = QGraphicsTextItem(command)
            cmd_item.setDefaultTextColor(Qt.GlobalColor.black)
            cmd_item.setFont(command_font)
            cmd_item.setPos(start_x, start_y + i * line_spacing)
            scene.addItem(cmd_item)

        # Create calibration button
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setFont(QFont("Lucida Sans"))
        button_width, button_height = self.responsive.scale_size(150, 50)
        self.calibrate_button.setFixedSize(button_width, button_height)
        self.calibrate_button.clicked.connect(self.on_button_clicked)
        self.calibrate_button.setStyleSheet(self.get_button_style())

        # Create a proxy widget for the button
        calib_button_proxy = QGraphicsProxyWidget()
        calib_button_proxy.setWidget(self.calibrate_button)
        calib_button_proxy.setPos(video_box.x() + video_box_width - button_width - self.responsive.scale_margin(250), 
                                video_box.y() + video_box_height + self.responsive.scale_margin(40))

        self.calibrate_button.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        calib_button_proxy.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        scene.addItem(calib_button_proxy)

        # Create overlay toggle button
        self.overlay_button = QPushButton("Toggle Transcription Overlay")
        self.overlay_button.setFont(QFont("Lucida Sans"))
        overlay_button_width, overlay_button_height = self.responsive.scale_size(350, 50)
        self.overlay_button.setFixedSize(overlay_button_width, overlay_button_height)
        self.overlay_button.clicked.connect(self.toggle_overlay)
        self.overlay_button.setStyleSheet(self.get_button_style())

        # Create a proxy widget for the button
        overlay_b_proxy = QGraphicsProxyWidget()
        overlay_b_proxy.setWidget(self.overlay_button)
        overlay_b_proxy.setPos(video_box.x() + self.responsive.scale_margin(75), 
                            video_box.y() + video_box_height + self.responsive.scale_margin(40))

        self.overlay_button.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        overlay_b_proxy.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        scene.addItem(overlay_b_proxy)

        # Set view size to match available area
        view.setSceneRect(0, 0, scene_width, scene_height)
        
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
        
        # Scale font size for group box
        font_size = self.responsive.scale_font_size(16)
        border_width = self.responsive.scale_margin(2)
        padding_top = self.responsive.scale_margin(15)
        margin_top = self.responsive.scale_margin(10)
        
        speed_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-size: {font_size}pt; 
                font-weight: bold; 
                color: #162552; 
                border: {border_width}px solid #B6BEDF; 
                padding-top: {padding_top}px; 
                margin-top: {margin_top}px; 
            }}
        """)
        
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
        
        # Apply responsive spacing between sliders
        speed_layout.addSpacing(self.responsive.scale_margin(15))
        
        # Add max speed slider
        max_speed_label, max_speed_row, max_speed_value = self.create_slider_with_label(
            "Max Speed", 10, 100, 40, "max_speed", 
            lambda value: self.update_controller_value("max_speed", value, float)
        )
        speed_layout.addWidget(max_speed_label)
        speed_layout.addWidget(max_speed_row)
        speed_layout.addSpacing(self.responsive.scale_margin(15))
        
        # Add acceleration slider
        accel_label, accel_row, accel_value = self.create_slider_with_label(
            "Acceleration", 5, 30, 15, "exp_factor", 
            lambda value: self.update_controller_value("exp_factor", value/10, float)
        )
        speed_layout.addWidget(accel_label)
        speed_layout.addWidget(accel_row)
        
        # Create group box for vertical sensitivity settings
        vert_group = QGroupBox("Vertical Sensitivity Settings")
        # Apply responsive styling to the group box
        vert_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-size: {font_size}pt; 
                font-weight: bold; 
                color: #162552; 
                border: {border_width}px solid #B6BEDF; 
                padding-top: {padding_top}px; 
                margin-top: {margin_top}px; 
            }}
        """)
        
        vert_layout = QVBoxLayout()
        vert_group.setLayout(vert_layout)
        content_layout.addWidget(vert_group)
        
        # Add up multiplier slider
        up_label, up_row, up_value = self.create_slider_with_label(
            "Upward Sensitivity", 5, 30, 5, "up_multiplier", 
            lambda value: self.update_controller_value("up_multiplier", value, float)
        )
        vert_layout.addWidget(up_label)
        vert_layout.addWidget(up_row)
        vert_layout.addSpacing(self.responsive.scale_margin(15))
        
        # Add down multiplier slider
        down_label, down_row, down_value = self.create_slider_with_label(
            "Downward Sensitivity", 5, 30, 5, "down_multiplier", 
            lambda value: self.update_controller_value("down_multiplier", value, float)
        )
        vert_layout.addWidget(down_label)
        vert_layout.addWidget(down_row)
        
        # Create group box for threshold settings
        threshold_group = QGroupBox("Threshold Settings")
        # Apply responsive styling to the group box
        threshold_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-size: {font_size}pt; 
                font-weight: bold; 
                color: #162552; 
                border: {border_width}px solid #B6BEDF; 
                padding-top: {padding_top}px; 
                margin-top: {margin_top}px; 
            }}
        """)
        
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
        threshold_layout.addSpacing(self.responsive.scale_margin(15))
        
        # Add click threshold slider
        click_label, click_row, click_value = self.create_slider_with_label(
            "Click Threshold (degrees)", 1, 30, 10, "click_threshold", 
            lambda value: self.update_controller_value("click_threshold", value, float)
        )
        threshold_layout.addWidget(click_label)
        threshold_layout.addWidget(click_row)
        threshold_layout.addSpacing(self.responsive.scale_margin(15))
        
        # Add click cooldown slider
        cooldown_label, cooldown_row, cooldown_value = self.create_slider_with_label(
            "Click Cooldown (seconds)", 0, 5, 1, "click_cooldown", 
            lambda value: self.update_controller_value("click_cooldown", value/10, float)
        )
        threshold_layout.addWidget(cooldown_label)
        threshold_layout.addWidget(cooldown_row)
        
        # Add spacer at the bottom with responsive height
        content_layout.addSpacing(self.responsive.scale_margin(20))
        
        # Add reset button at the bottom with responsive sizing
        reset_button = QPushButton("Reset to Default Settings")
        reset_button_width, reset_button_height = self.responsive.scale_size(300, 50)
        reset_button.setFixedSize(reset_button_width, reset_button_height)
        font = QFont("Lucida Sans")
        font.setPointSize(self.responsive.scale_font_size(12))
        reset_button.setFont(font)
        reset_button.setStyleSheet(self.get_button_style())
        reset_button.clicked.connect(self.reset_to_default)
        content_layout.addWidget(reset_button, 0, Qt.AlignmentFlag.AlignCenter)

        # Save the base speed slider for backwards compatibility
        self.sensitivity_slider = self.sliders.get('base_speed')

    def create_slider_with_label(self, label_text, min_val, max_val, default_val, key, callback_func):
        # Create label with responsive font size
        slider_label = QLabel(label_text)
        font_size = self.responsive.scale_font_size(14)
        margin_top = self.responsive.scale_margin(10)
        slider_label.setStyleSheet(f"font-size: {font_size}pt; margin-top: {margin_top}px;")

        # Create horizontal layout for slider and value
        slider_row = QHBoxLayout()
        
        # Create slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setSingleStep(1)
        slider.setPageStep(2)
        slider.setTickPosition(QSlider.TickPosition.TicksAbove)
        
        # Set responsive maximum width
        slider_width = self.responsive.scale_size(600, 0)[0]
        slider.setMaximumWidth(slider_width)
        
        # Apply custom style with responsive sizing
        style = SliderProxyStyle(slider.style(), self.responsive)
        slider.setStyle(style)
        slider.setStyleSheet(self.get_slider_style())
        
        # Create value label with responsive font size
        value_label = QLabel(f"{label_text}: {default_val}")
        value_font_size = self.responsive.scale_font_size(12)
        value_label.setStyleSheet(f"font-size: {value_font_size}pt;")

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
            self.mouse_controller.base_speed = 30.0
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
        
        # Create title label centered at the top with responsive font size
        title = QLabel("Getting Started")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font_size = self.responsive.scale_font_size(40)
        margin_value = self.responsive.scale_margin(20)
        bottom_margin = self.responsive.scale_margin(40)
        title.setStyleSheet(f"font-size: {title_font_size}px; font-weight: bold; margin: {margin_value}px 0 {bottom_margin}px 0; color: #162552;")
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
        
        # Create each step box with responsive dimensions
        for number, text, color in steps:
            # Create the box widget with responsive width
            box_width = self.responsive.scale_size(300, 0)[0]
            box = QWidget()
            box.setFixedWidth(box_width)
            box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            
            # Apply responsive styling
            border_radius = self.responsive.scale_margin(15)
            padding = self.responsive.scale_margin(20)
            min_height = self.responsive.scale_size(0, 150)[1]
            border_width = self.responsive.scale_margin(2)
            
            box.setStyleSheet(f"""
                background-color: #B6BEDF;
                border-radius: {border_radius}px;
                padding: {padding}px {padding - 5}px;
                min-height: {min_height}px;
                border: {border_width}px solid #C5CFF4;
            """)
            
            box_layout = QVBoxLayout()
            margin = self.responsive.scale_margin(15)
            box_layout.setContentsMargins(margin, margin, margin, margin)
            box_layout.setSpacing(margin)
            box.setLayout(box_layout)
            
            # Create number view with responsive dimensions
            circle_size = self.responsive.scale_size(100, 100)
            number_view = QGraphicsView()
            number_view.setFixedSize(circle_size[0], circle_size[1])
            number_view.setStyleSheet("background: transparent; border: none;")
            number_view.setRenderHint(QPainter.RenderHint.Antialiasing)
            number_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            number_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            number_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Create the scene with responsive dimensions
            scene_size = self.responsive.scale_size(80, 80)
            scene_offset = scene_size[0] / 2
            number_scene = QGraphicsScene()
            number_scene.setSceneRect(-scene_offset, -scene_offset, scene_size[0], scene_size[1])
            number_view.setScene(number_scene)

            # Create the circle centered in the scene with responsive dimensions
            circle_radius = self.responsive.scale_size(60, 60)[0] / 2
            circle = QGraphicsEllipseItem(-circle_radius, -circle_radius, circle_radius * 2, circle_radius * 2)
            circle.setBrush(QColor(color))
            circle.setPen(QPen(Qt.PenStyle.NoPen))
            number_scene.addItem(circle)

            # Add number text with responsive font size
            text_item = QGraphicsTextItem(number)
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            font_size = self.responsive.scale_font_size(24)
            text_item.setFont(QFont("Lucida Sans", font_size, QFont.Weight.Bold))

            # Position the text in the center of the circle
            text_rect = text_item.boundingRect()
            text_x = -text_rect.width() / 2
            text_y = -text_rect.height() / 2
            text_item.setPos(text_x, text_y)

            number_scene.addItem(text_item)
            box_layout.addWidget(number_view, 0, Qt.AlignmentFlag.AlignCenter)
            
            # Create step text with responsive styling
            text_label = QLabel(text)
            text_label.setWordWrap(True)
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_font_size = self.responsive.scale_font_size(20)
            margin_top = self.responsive.scale_margin(15)
            text_label.setStyleSheet(f"font-size: {text_font_size}px; margin-top: {margin_top}px; color: #000000; line-height: 1.4;")
            text_label.setFont(QFont("Fira Sans"))
            box_layout.addWidget(text_label)
            
            # Add spacer at the bottom to push content up
            box_layout.addStretch()
            
            # Add box to container with some margin
            boxes_layout.addWidget(box)
            if number != "4":  # Don't add spacing after the last box
                box_spacing = self.responsive.scale_margin(20)
                boxes_layout.addSpacing(box_spacing)
        
        # Add the boxes container to the main layout with responsive margin
        margin_top = self.responsive.scale_margin(30)
        boxes_container.setStyleSheet(f"margin-top: {margin_top}px;")
        help_layout.addWidget(boxes_container)
        
        # Add stretch to push everything to the top and center
        help_layout.addStretch()

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
        if hasattr(self, 'transcription_box') and self.transcription_box and self.transcription_box.isVisible():
            self.transcription_box.close()
        else:
            if hasattr(self, 'transcription_box') and self.transcription_box:
                self.transcription_box.show()

    def no_cam_err_msg(self):
        # Create a black pixmap with responsive dimensions
        width, height = self.responsive.scale_size(700, 500)
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0))

        # Use painter to draw on the pixmap with responsive font sizes
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 0, 0))
        
        main_font_size = self.responsive.scale_font_size(20)
        sub_font_size = self.responsive.scale_font_size(16)
        
        painter.setFont(QFont('Arial', main_font_size))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "No camera detected")

        painter.setFont(QFont('Arial', sub_font_size))
        rect = pixmap.rect()
        vertical_offset = self.responsive.scale_size(0, 40)[1]
        rect.translate(0, vertical_offset)  # Move down for second line with responsive distance
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
        min_height = self.responsive.scale_size(0, 30)[1]
        font_size = self.responsive.scale_font_size(16)
        padding_v = self.responsive.scale_margin(10)
        padding_h = self.responsive.scale_margin(20)
        margin = self.responsive.scale_margin(5)
        border_radius = self.responsive.scale_margin(5)
        
        return f"""
            QMenuBar {{
                min-height: {min_height}px;
                font-size: {font_size}pt;
                background-color: #162552;
            }}
            QMenuBar::item {{
                padding: {padding_v}px {padding_h}px;
                margin: {margin}px;
                background-color: transparent;
                color: #B8B8B8;
            }}
            QMenuBar::item:selected {{
                background-color: #3D4F84;
                color: white;
                border-radius: {border_radius}px;
            }}
        """

    def get_button_style(self):
        # Return CSS stylesheet for buttons with responsive dimensions
        font_size = self.responsive.scale_font_size(20)
        border_radius = self.responsive.scale_margin(15)
        padding = self.responsive.scale_margin(8)
        
        return f"""
            QPushButton {{
                background-color: #3A356F;
                color: white;
                border-radius: {border_radius}px;
                font-weight: bold;
                font-size: {font_size}px;
                border: none;
                text-align: center;
                padding: {padding}px;
            }}
            QPushButton:hover {{
                background-color: #6661A1;
            }}
            QPushButton:pressed {{
                background-color: #2E295E;
            }}
        """
    
    def get_slider_style(self):
        # Return CSS stylesheet for slider control with responsive dimensions
        handle_width = self.responsive.scale_size(85, 0)[0]
        handle_height = self.responsive.scale_size(80, 0)[1]
        handle_margin = self.responsive.scale_margin(30)
        handle_radius = self.responsive.scale_margin(40)
        
        track_height = self.responsive.scale_size(0, 30)[1]
        track_radius = self.responsive.scale_margin(10)
        
        return f"""
            QSlider::handle:horizontal {{
                background: #FCD88B;
                width: {handle_width}px;
                height: {handle_height}px;
                margin: -{handle_margin}px 0px;
                border-radius: {handle_radius}px;
            }}
            QSlider::add-page:horizontal {{
                background: #B6BEDF;
                border-radius: {track_radius}px;
                height: {track_height / 3}px;
            }}
            QSlider::sub-page:horizontal {{
                background: #EDCB80;
                border-radius: {track_radius}px;
                height: {track_height / 3}px;
            }}
            QSlider::groove:horizontal {{
                height: {track_height}px;
                background: #d3d3d3;
                border-radius: {track_radius}px;
            }}
        """