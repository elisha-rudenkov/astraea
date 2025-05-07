# Astraea 🖱️👁️🎤

Empower hands-free computer navigation using facial gestures and voice commands.

Astraea is a Windows application that transforms head movements, eye states, and voice commands into computer controls, designed for individuals with motor disabilities. Built with optimized AI models, it leverages on-device processing for real-time, privacy-focused accessibility.

## Key Features ✨

* **Face-Driven Cursor:** Map head movements to mouse navigation using advanced face detection.
* **Eye Gesture Controls:**
  * Left Click: Hold eyes closed for 1-2 seconds
  * Right Click: Double-blink detection (two rapid closures)
* **Voice Commands:**
  * Control mouse movement and clicks
  * Execute custom commands
  * Pause/resume mouse control
* **Low-Latency Performance:** Optimized for real-time processing
* **Customizable UI:** Adjust sensitivity, calibration, and gesture thresholds

## Requirements 📋

* Windows 10 or later
* Python 3.8 or later
* Webcam
* Microphone
* (Optional) GPU for enhanced performance

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astraea.git
cd astraea
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU acceleration:
```bash
pip install onnxruntime-gpu
```

## Usage 🎯

1. Launch the application:
```bash
python main.py
```

2. **Calibration:**
   * Center your face in the virtual bounding box
   * Press 'C' or use the Calibrate button in the UI
   * Adjust sensitivity settings as needed

3. **Controls:**
   * Move your head to control the cursor
   * Use eye gestures for clicks
   * Use voice commands for additional control
   * Press SPACE to temporarily pause mouse movement

4. **Voice Commands:**
   * "Pause mouse" - Temporarily disable mouse control
   * "Resume mouse" - Re-enable mouse control
   * "Click" - Perform a left click
   * "Right click" - Perform a right click
   * Custom commands can be configured in the UI

## Tech Stack 🛠️

* **AI Models:**
  * Face Detection and Tracking
  * Facial Landmark Analysis
  * Voice Recognition

* **Core Technologies:**
  * Python with OpenCV for camera processing
  * PyQt6 for the user interface
  * ONNX Runtime for model inference
  * Speech Recognition for voice commands

## Privacy & Performance 🌟

* **Privacy-First:** All processing happens on-device
* **Hardware-Agnostic:** Works with standard webcams and microphones
* **Open Source:** Built for and by the accessibility community
* **Customizable:** Adjust settings to match your needs and hardware capabilities

## Contributing 🤝

We welcome contributions! Please feel free to submit issues and pull requests.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.
