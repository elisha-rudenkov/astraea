# Astraea 🖱️👁️🎤

Empower hands-free computer navigation using facial gestures and voice commands.

Astraea is a Windows application that transforms head movements and voice commands into computer controls, designed for individuals with motor disabilities. Built with optimized Qualcomm AI models, it leverages on-device processing for real-time, privacy-focused accessibility.

## Key Features ✨

* **Face-Driven Cursor:** Map head movements to mouse navigation using advanced face detection.
* **Voice Commands:**
  * Control mouse movement and clicks
  * Execute and create custom shortcuts/macros commands
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
   * Say 'Start Listening' for Astrea to start interpretting commands
   * Center your face in the virtual bounding box
   * Say 'Calibrate' or press 'C' on the keyboard
   * Adjust sensitivity settings as needed

3. **Controls:**
   * Move your head to control the cursor
   * Use the listed voice commands for additional control

4. **Voice Commands:**
   * "Pause mouse" - Temporarily disable mouse control
   * "Resume mouse" - Re-enable mouse control
   * "Left" - Perform a left click
   * "Right" - Perform a right click
   * "Create Command" - Opens a walkthrough on how to make a custom shortcut/macro

## Tech Stack 🛠️

* **AI Models:**
  * Face Detection and Tracking
  * Facial Landmark Analysis
  * Voice Recognition

* **Core Technologies:**
  * Python with OpenCV for camera processing
  * PyQt6 for the user interface
  * ONNX Runtime for model inference
  * Speech Recognition with Whisper AI for voice commands

## Privacy & Performance 🌟

* **Privacy-First:** All processing happens on-device
* **Hardware-Agnostic:** Works with standard webcams and microphones
* **Open Source:** Built for and by the accessibility community
* **Customizable:** Adjust settings to match your needs and hardware capabilities

## Contributing 🤝

We welcome contributions! Please feel free to submit issues and pull requests.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.
