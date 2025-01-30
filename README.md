Astraea 🖱️👁️
Empower hands-free computer navigation using facial gestures.

AccessiCursor is a Windows application that transforms head movements and eye states into mouse controls, designed for individuals with motor disabilities. Built with Qualcomm AI Hub's optimized models, it leverages on-device AI for real-time, privacy-focused accessibility.

Key Features ✨
Face-Driven Cursor: Map head movements to mouse navigation using the Lightweight-Face-Detection model.

Eye Gesture Controls:

Left Click: Hold eyes closed for 1-2 seconds.

Right Click: Double-blink detection (two rapid closures).

Low-Latency Performance: Optimized for Snapdragon X Elite laptops.

Customizable UI: Adjust sensitivity, calibration, and gesture thresholds.

Tech Stack 🛠️
Qualcomm AI Hub Models:

Facial-Attribute-Detection-Quantized (eye state, liveness detection).

Lightweight-Face-Detection (real-time face tracking).

Python: OpenCV for camera processing, PyQt for UI.

On-Device AI: Deployed via Qualcomm’s SNPE/QNN runtime.

How It Works 🎯
Calibrate: Center your face in a virtual bounding box.

Move: Deviate from the center to control the cursor directionally.

Click: Use eye gestures for left/right clicks.

Why Astraea? 🌟
Privacy-First: No cloud processing—data stays on-device.

Hardware-Agnostic: Works with standard webcams.

Open Source: Built for and by the accessibility community.


