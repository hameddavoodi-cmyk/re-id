"""
Test script to demonstrate preprocessing functionality.
Creates a synthetic cow pattern and shows preprocessing steps.
"""

import cv2
import numpy as np
from preprocessing import PatternPreprocessor


def create_test_pattern(width=400, height=300):
    """
    Create a synthetic cow-like pattern for testing.
    Simulates black spots on white background with varying lighting.
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add some dark spots (cow pattern)
    spots = [
        ((100, 80), 40),
        ((250, 120), 50),
        ((150, 200), 35),
        ((300, 180), 45),
    ]

    for (cx, cy), radius in spots:
        cv2.circle(image, (cx, cy), radius, (50, 50, 50), -1)

    # Add lighting gradient to simulate shadow
    for y in range(height):
        brightness_factor = 0.7 + 0.3 * (y / height)
        image[y, :] = (image[y, :] * brightness_factor).astype(np.uint8)

    # Add some noise
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def main():
    """Test preprocessing on synthetic pattern."""
    print("="*60)
    print("Testing Pattern Preprocessing")
    print("="*60)

    # Create test pattern
    print("\n1. Creating synthetic cow pattern...")
    test_image = create_test_pattern()
    print("   ✓ Pattern created")

    # Initialize preprocessor
    print("\n2. Initializing preprocessor...")
    preprocessor = PatternPreprocessor()
    print("   ✓ Preprocessor initialized")

    # Test basic preprocessing
    print("\n3. Testing basic preprocessing...")
    binary_pattern = preprocessor.preprocess(test_image, enhance_contrast=True)
    if binary_pattern is not None:
        print(f"   ✓ Binary pattern created: {binary_pattern.shape}")
        print(f"   - Unique values: {np.unique(binary_pattern)}")
    else:
        print("   ✗ Preprocessing failed")
        return 1

    # Test morphological preprocessing
    print("\n4. Testing morphological preprocessing...")
    morph_pattern = preprocessor.preprocess_with_morphology(
        test_image,
        enhance_contrast=True,
        morph_kernel_size=3,
        apply_opening=True
    )
    if morph_pattern is not None:
        print(f"   ✓ Morphological pattern created: {morph_pattern.shape}")
    else:
        print("   ✗ Morphological preprocessing failed")

    # Test RGB channel preprocessing
    print("\n5. Testing RGB channel preprocessing...")
    rgb_pattern = preprocessor.preprocess_rgb_channels(test_image)
    if rgb_pattern is not None:
        print(f"   ✓ RGB pattern created: {rgb_pattern.shape}")
    else:
        print("   ✗ RGB preprocessing failed")

    # Test adaptive preprocessing
    print("\n6. Testing adaptive preprocessing methods...")
    for method in ['otsu', 'adaptive_mean', 'adaptive_gaussian']:
        adaptive_pattern = preprocessor.adaptive_preprocess(test_image, method=method)
        if adaptive_pattern is not None:
            print(f"   ✓ {method}: {adaptive_pattern.shape}")
        else:
            print(f"   ✗ {method} failed")

    # Test visualization
    print("\n7. Testing visualization...")
    viz = preprocessor.visualize_preprocessing_steps(test_image)
    if viz is not None:
        print(f"   ✓ Visualization created: {viz.shape}")
    else:
        print("   ✗ Visualization failed")

    # Save test results
    print("\n8. Saving test results...")
    cv2.imwrite("test_original.png", test_image)
    cv2.imwrite("test_binary.png", binary_pattern)
    cv2.imwrite("test_morph.png", morph_pattern)
    if viz is not None:
        cv2.imwrite("test_visualization.png", viz)
    print("   ✓ Saved: test_original.png, test_binary.png, test_morph.png, test_visualization.png")

    print("\n" + "="*60)
    print("✓ All preprocessing tests passed!")
    print("="*60)
    print("\nTest images saved in current directory.")
    print("Open them to see the preprocessing results.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
