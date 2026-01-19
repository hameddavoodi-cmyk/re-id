"""
Quick test script to verify installation and device detection.
Run this before using the main app to ensure everything is set up correctly.
"""

import sys


def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")

    try:
        import streamlit
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False

    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False

    return True


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")

    try:
        from utils import get_device
        device, device_name = get_device()
        print(f"✓ Device detected: {device_name}")
        print(f"  Device object: {device}")
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False


def test_matchers():
    """Test if matcher classes can be instantiated."""
    print("\nTesting matcher classes...")

    try:
        from matcher_local_features import LocalFeatureMatcher
        matcher = LocalFeatureMatcher()
        print("✓ LocalFeatureMatcher initialized successfully")
    except Exception as e:
        print(f"✗ LocalFeatureMatcher failed: {e}")
        return False

    try:
        import torch
        from matcher_deep_learning import DeepFeatureMatcher
        device = torch.device("cpu")
        matcher = DeepFeatureMatcher(device)
        print("✓ DeepFeatureMatcher initialized successfully")
    except Exception as e:
        print(f"✗ DeepFeatureMatcher failed: {e}")
        return False

    return True


def test_tracker():
    """Test if tracker can be instantiated."""
    print("\nTesting tracker...")

    try:
        from matcher_local_features import LocalFeatureMatcher
        from cow_tracker import CowTracker

        matcher = LocalFeatureMatcher()
        tracker = CowTracker(matcher)
        print("✓ CowTracker initialized successfully")
        return True
    except Exception as e:
        print(f"✗ CowTracker failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Cow Tracking System - Setup Test")
    print("="*60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_device_detection()
    all_passed &= test_matchers()
    all_passed &= test_tracker()

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Setup is complete.")
        print("\nYou can now run the app with:")
        print("  streamlit run app.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTry installing missing packages with:")
        print("  pip install -r requirements.txt")
        return 1

    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
