import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QToolBar, QTextEdit, QWidget, QSlider,
                              QLabel, QFormLayout, QVBoxLayout, QPushButton, QHBoxLayout, 
                              QTabWidget, QStackedWidget, QGraphicsRectItem, QGraphicsScene,
                              QGraphicsView, QGraphicsTextItem, QGraphicsProxyWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
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

# Main window for Astrea
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a reference to store the MouseController instance
        self.mouse_controller = None
        self.init_ui()

    def register_mouse_controller(self, controller):
        #Register the mouse controller to allow adjusting its settings
        self.mouse_controller = controller
        # Initialize slider with current value
        if hasattr(self, 'sensitivity_slider') and self.mouse_controller:
            self.sensitivity_slider.setValue(int(self.mouse_controller.base_speed))
            self.update_slider_value(self.sensitivity_slider.value())

    def set_video_frame(self, frame):
        # Update the video frame in the UI
        if not hasattr(self, 'scene_video_label'):
            return
            
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        self.scene_video_label.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def get_sensitivity(self):
        # Get the current sensitivity value from the slider in the form layout
        for i in range(self.settings_page.layout().rowCount()):
            item = self.settings_page.layout().itemAt(i, QFormLayout.ItemRole.FieldRole)
            if item and isinstance(item.widget(), QSlider):
                return item.widget().value()
        return 50  # Default value

    def register_calibrate_callback(self, callback):
        # Register callback for calibration button
        self.calibrate_button.clicked.disconnect()
        self.calibrate_button.clicked.connect(callback)
        
    def update_head_position_display(self, pitch, yaw, roll): # (Needs to be implemented)
        """Update the head position display in the UI"""
        # Implementation to update head position monitoring box
        pass

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
        self.show()

        from command_settings import InfoOverlay
        self.overlay = InfoOverlay(self)
        self.overlay.show_overlay()

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

        font = video_label.font()
        font.setPointSize(16)
        video_label.setFont(font)

        # Position the label at the top center of the video feed box
        label_x = video_box.x() + (video_box.rect().width() - video_label.boundingRect().width()) / 2
        label_y = video_box.y() + 10 # Margin from top of box
        video_label.setPos(label_x, label_y)
        scene.addItem(video_label) # Add label to the scene

        # Create calibration button
        self.calibrate_button = QPushButton("Calibrate (C)")
        self.calibrate_button.setFixedSize(200, 50)
        self.calibrate_button.clicked.connect(self.on_button_clicked)
        self.calibrate_button.setStyleSheet(self.get_button_style_2())

        # Create a proxy widget for the button
        button_proxy = QGraphicsProxyWidget()
        button_proxy.setWidget(self.calibrate_button)
        button_proxy.setPos(450, 735)

        self.calibrate_button.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        button_proxy.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        scene.addItem(button_proxy)

        # Add head position monitoring box
        head_pos_box = CustomGraphicsItem(500, 415, "#D9D9D9")
        head_pos_box.setPos(900, 0)
        scene.addItem(head_pos_box)

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
        settings_layout = QFormLayout()
        self.settings_page.setLayout(settings_layout)

        # Create content widget for settings
        content_widget = QWidget()
        content_layout = QFormLayout()
        content_widget.setLayout(content_layout)

        # Add settings adjustment buttons
        button_layout = self.create_adjustment_buttons()
        content_layout.addRow(button_layout)

        # Add slider control
        slider = self.create_sensitivity_slider()
        content_layout.addRow(slider)

        # Add cursor settings tab label
        self.result_label = QLabel('Cursor Settings', self)  # Default value
        self.result_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: black;")
        settings_layout.addWidget(self.result_label)

        # Add sensitivity label
        self.result_label = QLabel('', self)
        content_layout.addWidget(self.result_label)
        self.result_label.setStyleSheet("font-size: 12pt")
        settings_layout.addWidget(content_widget)

    def create_help_page(self):
        # Create the help page and add components
        self.help_page = QWidget()
        help_layout = QFormLayout()
        self.help_page.setLayout(help_layout)
        help_layout.addRow(QLabel("Help Page Content"))

    def create_adjustment_buttons(self):
        # Create container for label and buttons
        container = QVBoxLayout()
        
        # Add label above buttons
        button_label = QLabel("Cursor Base Speed")
        button_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_label.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 5px;")
        container.addWidget(button_label)

        # Create settings adjustment buttons
        button_layout = QHBoxLayout()

        # Create increment button
        inc_button = QPushButton("+")
        inc_button.setFixedSize(100, 100)
        inc_button.clicked.connect(self.on_button_clicked)
        inc_button.setStyleSheet(self.get_button_style())
        button_layout.addWidget(inc_button)

        # Create decrement button
        dec_button = QPushButton("-")
        dec_button.setFixedSize(100, 100)
        dec_button.clicked.connect(self.on_button_clicked)
        dec_button.setStyleSheet(self.get_button_style())
        button_layout.addWidget(dec_button)

        # Add button layout to container
        container.addLayout(button_layout)
        
        return container
    
    def create_sensitivity_slider(self):
        # Create and configure sensitivity slider control
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 20)  # Range from 1 to 20 for base_speed
        slider.setValue(8)  # Default to match the default base_speed in MouseController
        slider.setSingleStep(1)
        slider.setPageStep(2)
        slider.setTickPosition(QSlider.TickPosition.TicksAbove)
        slider.setMaximumWidth(500)

        # Apply custom style to slider
        style = SliderProxyStyle(slider.style())
        slider.setStyle(style)
        slider.setStyleSheet(self.get_slider_style())

        # Connect value change signal
        slider.valueChanged.connect(self.update_slider_value)
        
        # Store the slider as an instance variable
        self.sensitivity_slider = slider

        return slider
    
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
        
    def update_slider_value(self, value):
        # Update current slider value and mouse controller settings
        self.result_label.setText(f'Sensitivity: {value}')
        
        # Update the mouse controller if it's registered
        if self.mouse_controller:
            self.mouse_controller.base_speed = float(value)
            # Adjust max_speed proportionally
            self.mouse_controller.max_speed = float(value) * 5.0
            print(f"Updated sensitivity - base_speed: {self.mouse_controller.base_speed}, max_speed: {self.mouse_controller.max_speed}")

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
                border-radius: 30px;       /* Rounded corners */
                font-weight: normal;         /* Bold text */
                font-size: 50px;           /* Larger text */
                border: none;              /* No border */
                text-align: center;        /* Center text horizontally */
                padding: 0px;              /* Remove default padding */
            }
            QPushButton:hover {
                background-color: #3A62E0; /* Lighter blue when hovering */
            }
            QPushButton:pressed {
                background-color: #1A3DB0; /* Darker blue when pressed */
            }
        """
    def get_button_style_2(self):
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