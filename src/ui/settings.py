import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QToolBar, QTextEdit, QWidget, QSlider,
                              QLabel, QFormLayout, QVBoxLayout, QPushButton, QHBoxLayout, 
                              QTabWidget, QStackedWidget, QGraphicsRectItem, QGraphicsScene,
                              QGraphicsView, QGraphicsTextItem, QGraphicsProxyWidget, QScrollArea, 
                              QGroupBox, QHBoxLayout)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QFont, QFontDatabase
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
    def __init__(self, width, height, color):
        super().__init__()
        self.setRect(0, 0, width, height)
        self.setBrush(QColor(color))

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
    
    def update_calibration_status(self, is_calibrated, values=None): # (Needs to be implemented)
        """Update the calibration status display"""
        # Implementation to update calibration status box
        pass
    
    def init_ui(self):
        # Initialize the user interface components
        self.setWindowTitle('Astrea')
        self.setMinimumWidth(200)
        
        # Set background color
        self.setStyleSheet("background-color: #E9F2FF;")

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QFormLayout()
        central_widget.setLayout(main_layout)
        
        # Create stacked widget for multiple pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Create pages
        self.create_home_page()
        self.create_settings_page()
        self.create_help_page()

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.help_page)

        # Configure menu bar
        self.setup_menu_bar()

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

        # Add video feed box
        video_box = CustomGraphicsItem(750, 595, "#D9D9D9")
        video_box.setPos(100, 100)
        scene.addItem(video_box)

        # Create a QLabel for the video feed in the scene
        self.scene_video_label = QLabel()
        self.scene_video_label.setFixedSize(700, 500)  # Slightly smaller than the video_box
        self.scene_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene_video_label.setStyleSheet("background-color: transparent;")
        
        # Create a proxy widget to add the label to the scene
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(self.scene_video_label)
        
        # Position the video label inside the video box (with some margin)
        proxy.setPos(video_box.pos() + QtCore.QPointF(25, 50))
        scene.addItem(proxy)

        # Add video feed label
        video_label = QGraphicsTextItem("Video Feed")
        video_label.setDefaultTextColor(Qt.GlobalColor.black)

        font = QFont("Lucida Sans", 16)
        video_label.setFont(font)

        # Position the label at the top center of the video feed box
        label_x = video_box.x() + (video_box.rect().width() - video_label.boundingRect().width()) / 2
        label_y = video_box.y() + 10 # Margin from top of box
        video_label.setPos(label_x, label_y)
        scene.addItem(video_label) # Add label to the scene

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

        # Add head position monitoring box
        head_pos_box = CustomGraphicsItem(500, 415, "#D9D9D9")
        head_pos_box.setPos(900, 0)
        scene.addItem(head_pos_box)

        # Add label to head position box
        help_title = QGraphicsTextItem("Voice Commands")
        help_title.setDefaultTextColor(Qt.GlobalColor.black)
        title_font = help_title.font()
        title_font = QFont("Fira Sans", 16)
        help_title.setFont(title_font)

        # Position the title at the top center of the head position box
        title_x = head_pos_box.x() + (head_pos_box.rect().width() - help_title.boundingRect().width()) / 2
        title_y = head_pos_box.y() + 10  # Margin from top of box
        help_title.setPos(title_x, title_y)
        scene.addItem(help_title)

        # Add text to the head position box
        line_1_text = QGraphicsTextItem("- Say \"start listening\" to activate voice commands")
        line_1_text.setDefaultTextColor(Qt.GlobalColor.black)
        text_font = line_1_text.font()
        text_font = QFont("Fira Sans", 14)
        line_1_text.setFont(text_font)
        line_1_text.setPos(head_pos_box.x() + 20, head_pos_box.y() + 50)
        scene.addItem(line_1_text)

        line_2_text = QGraphicsTextItem("- Say \"stop listening\" to deactivate voice commands")
        line_2_text.setDefaultTextColor(Qt.GlobalColor.black)
        line_2_text.setFont(text_font)
        line_2_text.setPos(head_pos_box.x() + 20, head_pos_box.y() + 80)
        scene.addItem(line_2_text)

        line_3_text = QGraphicsTextItem("- Say \"right\" to right click")
        line_3_text.setDefaultTextColor(Qt.GlobalColor.black)
        line_3_text.setFont(text_font)
        line_3_text.setPos(head_pos_box.x() + 20, head_pos_box.y() + 110)
        scene.addItem(line_3_text)

        line_4_text = QGraphicsTextItem("- Say \"left\" to left click")
        line_4_text.setDefaultTextColor(Qt.GlobalColor.black)
        line_4_text.setFont(text_font)
        line_4_text.setPos(head_pos_box.x() + 20, head_pos_box.y() + 140)
        scene.addItem(line_4_text)

        line_5_text = QGraphicsTextItem("- Say \"hold\" to hold down left click")
        line_5_text.setDefaultTextColor(Qt.GlobalColor.black)
        line_5_text.setFont(text_font)
        line_5_text.setPos(head_pos_box.x() + 20, head_pos_box.y() + 170)
        scene.addItem(line_5_text)

        line_6_text = QGraphicsTextItem("- Say \"release\" to release after holding down click")
        line_6_text.setDefaultTextColor(Qt.GlobalColor.black)
        line_6_text.setFont(text_font)
        line_6_text.setPos(head_pos_box.x() + 20, head_pos_box.y() + 200)
        scene.addItem(line_6_text)

        # Add calibration status box
        calib_box = CustomGraphicsItem(500, 200, "#D9D9D9")
        calib_box.setPos(900, 425)
        scene.addItem(calib_box)

        # Add sensitivity status box
        sens_box = CustomGraphicsItem(500, 150, "#D9D9D9")
        sens_box.setPos(900, 635)
        scene.addItem(sens_box)

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
        speed_group.setStyleSheet("QGroupBox { font-size: 16pt; font-weight: bold; padding-top: 15px; margin-top: 10px; }")
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
        vert_group.setStyleSheet("QGroupBox { font-size: 16pt; font-weight: bold; padding-top: 15px; margin-top: 10px; }")
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
        threshold_group.setStyleSheet("QGroupBox { font-size: 16pt; font-weight: bold; padding-top: 15px; margin-top: 10px; }")
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
        help_layout = QHBoxLayout()
        self.help_page.setLayout(help_layout)
        
        # Create the left sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("background-color: #CFDAE9")
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)

        # Create the sidebar buttons
        help_topics = [
            "Getting Started",
            "Calibration Guide",
            "Troubleshooting",
            "FAQ",
            "Contact Us"
        ]

        self.topic_buttons = []
        for topic in help_topics:
            button = QPushButton(topic)
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

            # Connect button clicks to the change_help_content function
            button.clicked.connect(lambda checked, t=topic: self.change_help_content(t))
            sidebar_layout.addWidget(button)
            self.topic_buttons.append(button)

        # Highlight the first button (Getting Started) by default
        self.topic_buttons[0].setStyleSheet(self.topic_buttons[0].styleSheet() +
                                            "QPushButton { background-color: #0288D1; color: white; }")
        
        # Add stretch to push buttons to the top
        sidebar_layout.addStretch()

        # Create the right content area with a stacked widget
        self.help_content = QStackedWidget()
        self.help_content.setStyleSheet("background-color: white;")

        # Create different topic pages
        self.create_getting_started_page()
        self.create_calibration_guide_page()
        self.create_troubleshooting_page()
        self.create_faq_page()
        self.create_contact_page()

        # Add left sidebar and right content to main layout
        help_layout.addWidget(sidebar)
        help_layout.addWidget(self.help_content, 1) # stretch factor 1

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
        elif topic == "Calibration Guide":
            self.help_content.setCurrentIndex(1)
        elif topic == "Troubleshooting":
            self.help_content.setCurrentIndex(2)
        elif topic == "FAQ":
            self.help_content.setCurrentIndex(3)
        elif topic == "Contact Us":
            self.help_content.setCurrentIndex(4)

    def create_getting_started_page(self):
        # Create the Getting Started page content
        getting_started = QWidget()
        layout = QVBoxLayout()
        getting_started.setLayout(layout)
        
        # Add title
        title = QLabel("Getting Started with Astrea")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Add video placeholder
        video_placeholder = QWidget()
        video_placeholder.setStyleSheet("background-color: #D9D9D9; min-height: 300px;")
        video_layout = QVBoxLayout()
        video_placeholder.setLayout(video_layout)

        # Add play button in the center of the video placeholder
        play_button = QPushButton("▶")
        play_button.setFixedSize(100, 100)
        play_button.setStyleSheet("""
            QPushButton {
                background-color: #A7C8F0;
                border-radius: 50px;
                color: white;
                font-size: 40px;
            }
        """)
        
        video_layout.addWidget(play_button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(video_placeholder)
        
        # Add Quick Start Guide
        guide_title = QLabel("Quick Start Guide")
        guide_title.setStyleSheet("font-size: 20px; font-weight: bold; margin-top: 20px; margin-bottom: 10px;")
        layout.addWidget(guide_title)
        
        # Create steps
        steps = [
            ("1", "Position yourself in front of the camera"),
            ("2", "Press calibration button or say \"calibrate\" to calibrate"),
            ("3", "Move your head to control the cursor"),
            ("4", "Tilt head to click (left for left click, right for right click)")
        ]
        
        for number, text in steps:
            step_widget = QWidget()
            step_layout = QHBoxLayout()
            step_widget.setLayout(step_layout)
            
            # Create number circle
            number_label = QLabel(number)
            number_label.setFixedSize(50, 50)
            number_label.setStyleSheet("""
                background-color: #0085CA;
                color: white;
                font-size: 22px;
                font-weight: bold;
                border-radius: 25px;
                qproperty-alignment: AlignCenter;
            """)
            
            # Create step text
            text_label = QLabel(text)
            text_label.setStyleSheet("font-size: 16px; margin-left: 10px;")
            
            step_layout.addWidget(number_label)
            step_layout.addWidget(text_label, 1)
            step_layout.addStretch()
            
            layout.addWidget(step_widget)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Add to stacked widget
        self.help_content.addWidget(getting_started)

    def create_calibration_guide_page(self):
        # Create the Calibration Guide page content
        calibration_guide = QWidget()
        layout = QVBoxLayout()
        calibration_guide.setLayout(layout)
        
        title = QLabel("Calibration Guide")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        content = QLabel("Calibration instructions will go here...")
        layout.addWidget(content)
        
        # Add stretch
        layout.addStretch()
        
        # Add to stacked widget
        self.help_content.addWidget(calibration_guide)

    def create_troubleshooting_page(self):
        # Create the Troubleshooting page content
        troubleshooting = QWidget()
        layout = QVBoxLayout()
        troubleshooting.setLayout(layout)
        
        title = QLabel("Troubleshooting")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        content = QLabel("Troubleshooting tips will go here...")
        layout.addWidget(content)
        
        # Add stretch
        layout.addStretch()
        
        # Add to stacked widget
        self.help_content.addWidget(troubleshooting)

    def create_faq_page(self):
        # Create the FAQ page content
        faq = QWidget()
        layout = QVBoxLayout()
        faq.setLayout(layout)
        
        title = QLabel("Frequently Asked Questions")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        content = QLabel("FAQs will go here...")
        layout.addWidget(content)
        
        # Add stretch
        layout.addStretch()
        
        # Add to stacked widget
        self.help_content.addWidget(faq)

    def create_contact_page(self):
        # Create the Contact Us page content
        contact = QWidget()
        layout = QVBoxLayout()
        contact.setLayout(layout)
        
        title = QLabel("Contact Us")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        content = QLabel("Contact information will go here...")
        layout.addWidget(content)
        
        # Add stretch
        layout.addStretch()
        
        # Add to stacked widget
        self.help_content.addWidget(contact)

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

    def get_menu_bar_style(self):
        # Return CSS stylesheet for menu bar
        return """
            QMenuBar {
                min-height: 50px;
                font-size: 16pt;
                background-color: #2F3F70;
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
                background-color: #214CDA; /* Blue background */
                color: white;              /* White text */
                border-radius: 15px;       /* Rounded corners */
                font-weight: bold;         /* Bold text */
                font-size: 20px;           /* Larger text */
                border: none;              /* No border */
                text-align: center;        /* Center text horizontally */
                padding: 8px;              /* Add some padding */
            }
            QPushButton:hover {
                background-color: #3A62E0; /* Lighter blue when hovering */
            }
            QPushButton:pressed {
                background-color: #1A3DB0; /* Darker blue when pressed */
            }
        """
    
    def get_slider_style(self):
        # Return CSS stylesheet for slider control
        return """
            QSlider::handle:horizontal {
                background:rgb(33, 76, 218);
                width: 85px;
                height: 80px;
                margin: -30px 0px;
                border-radius: 40px;
            }
            QSlider::add-page:horizontal {
                background: #d3d3d3;            /* Color for the right side of the handle */
                border-radius: 10px;
                height: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #9dc5fa;            /* Color for the left side of the handle */
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