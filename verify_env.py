"""
Environment verification script for Multimodal Attendance & Engagement Agent.
Verifies all core dependencies, basic functionality, and prints environment information.
"""
import sys
import importlib

REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("sklearn", "scikit-learn"),
    ("streamlit", "streamlit"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
]

def check_environment() -> bool:
    print("=" * 60)
    print("Multimodal Attendance & Engagement Agent - Environment Check")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print("-" * 60)
    
    all_passed = True
    for module_name, package_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed (version unspecified)")
            print(f"[OK] {package_name:<18} (import {module_name:<12}) -> v{ver}")
        except ImportError as e:
            print(f"[FAILED] {package_name:<18} (import {module_name:<12}) -> Error: {e}")
            all_passed = False

    print("-" * 60)
    if all_passed:
        # Also check OpenCV cascade availability as a baseline check for vision
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                print("[WARNING] OpenCV Haar cascade xml could not be loaded.")
            else:
                print(f"[OK] OpenCV built-in face detector model loaded from: {cascade_path}")
        except Exception as e:
            print(f"[WARNING] OpenCV face cascade test: {e}")

        print("=" * 60)
        print("All required dependencies are installed and operational.")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("Some dependencies are missing. Run: pip install -r requirements.txt")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
