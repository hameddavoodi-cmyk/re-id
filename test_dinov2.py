"""
Test script for DINOv2 patch-level matching

This script demonstrates how to use the DINOv2 model for cow re-identification.
It shows both the basic usage and advanced features.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from dinov2_model import DINOv2EmbeddingModel, DINOv2PatchMatcher


def test_basic_usage():
    """Test basic usage with synthetic images"""
    print("=" * 60)
    print("TEST 1: Basic Usage with Synthetic Images")
    print("=" * 60)

    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA GPU"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"Device: {device_name}\n")

    # Initialize model
    print("Initializing DINOv2-Base model...")
    model = DINOv2EmbeddingModel("facebook/dinov2-base", device=device)
    print("Model loaded!\n")

    # Create test images
    print("Creating synthetic test images...")
    # Image 1: Random pattern
    img1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # Image 2: Similar to image 1 with some noise
    img2 = img1.copy()
    noise = np.random.randint(-30, 30, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Image 3: Completely different
    img3 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Image 3 shape: {img3.shape}\n")

    # Test similarities
    print("Computing similarities...")
    score_1_2 = model.get_similarity_score(img1, img2)
    score_1_3 = model.get_similarity_score(img1, img3)
    score_1_1 = model.get_similarity_score(img1, img1)

    print(f"Similarity (img1 vs img2 - similar): {score_1_2:.3f}")
    print(f"Similarity (img1 vs img3 - different): {score_1_3:.3f}")
    print(f"Similarity (img1 vs img1 - identical): {score_1_1:.3f}\n")

    # Interpretation
    print("Interpretation:")
    print(f"  - Identical images should score ~1.0: {score_1_1:.3f} ✓" if score_1_1 > 0.95 else f"  - Identical images: {score_1_1:.3f} ✗")
    print(f"  - Similar images should score high: {score_1_2:.3f} ✓" if score_1_2 > 0.8 else f"  - Similar images: {score_1_2:.3f}")
    print(f"  - Different images should score lower: {score_1_3:.3f}\n")


def test_patch_extraction():
    """Test patch extraction and filtering"""
    print("=" * 60)
    print("TEST 2: Patch Extraction and Filtering")
    print("=" * 60)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = DINOv2EmbeddingModel("facebook/dinov2-base", device=device)

    # Create test image with some uniform regions
    img = np.zeros((280, 280, 3), dtype=np.uint8)

    # Add some patterns
    img[0:140, 0:140] = [200, 50, 50]      # Red region
    img[0:140, 140:280] = [50, 200, 50]    # Green region
    img[140:280, 0:140] = [50, 50, 200]    # Blue region
    img[140:280, 140:280] = [255, 255, 255]  # White region (uniform)

    print("Test image created with 4 quadrants (R, G, B, White)\n")

    # Extract patches
    print("Extracting patches...")
    patches, embeddings = model.extract_patch_embeddings(img)
    print(f"Total patches extracted: {len(patches)}")
    print(f"Embedding dimension: {embeddings.shape[1]}\n")

    # Filter with different thresholds
    for threshold in [0.001, 0.01, 0.02]:
        patches_f, embeddings_f = model.filter_distinctive_patches(
            img, patches, embeddings, variance_threshold=threshold
        )
        print(f"Variance threshold {threshold}: {len(embeddings_f)}/{len(embeddings)} patches kept")

    print()


def test_patch_matcher_wrapper():
    """Test the DINOv2PatchMatcher wrapper class"""
    print("=" * 60)
    print("TEST 3: DINOv2PatchMatcher Wrapper")
    print("=" * 60)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Initialize matcher
    print("Initializing DINOv2PatchMatcher...")
    matcher = DINOv2PatchMatcher("facebook/dinov2-base", device=device)
    print("Matcher initialized!\n")

    # Create reference image
    print("Setting reference image...")
    ref_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    success = matcher.set_reference(ref_img)
    print(f"Reference set: {success}")
    print(f"Has reference: {matcher.has_reference()}\n")

    # Test comparison
    print("Comparing with query images...")

    # Query 1: Same as reference
    query1 = ref_img.copy()
    score1 = matcher.compare(query1)
    print(f"Score (identical): {score1:.3f}")

    # Query 2: Similar to reference
    query2 = ref_img.copy()
    noise = np.random.randint(-20, 20, query2.shape, dtype=np.int16)
    query2 = np.clip(query2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    score2 = matcher.compare(query2)
    print(f"Score (similar): {score2:.3f}")

    # Query 3: Different
    query3 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    score3 = matcher.compare(query3)
    print(f"Score (different): {score3:.3f}\n")


def test_with_real_images():
    """Test with real images if available"""
    print("=" * 60)
    print("TEST 4: Real Images (if available)")
    print("=" * 60)

    # Check for test images
    test_image_paths = [
        "test_original.png",
        "Untitled design.png",
    ]

    available_images = [p for p in test_image_paths if Path(p).exists()]

    if not available_images:
        print("No test images found in project directory.")
        print("Skipping real image test.\n")
        return

    print(f"Found {len(available_images)} test image(s)\n")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = DINOv2EmbeddingModel("facebook/dinov2-base", device=device)

    for img_path in available_images[:2]:  # Test with first 2 images
        print(f"Testing with: {img_path}")
        img = cv2.imread(img_path)

        if img is None:
            print(f"  Failed to load image\n")
            continue

        print(f"  Image shape: {img.shape}")

        # Extract patches
        patches, embeddings = model.extract_patch_embeddings(img)
        print(f"  Patches extracted: {len(patches)}")

        # Filter
        patches_f, embeddings_f = model.filter_distinctive_patches(
            img, patches, embeddings
        )
        print(f"  Distinctive patches: {len(embeddings_f)}")

        # Self-similarity
        score = model.get_similarity_score(img, img)
        print(f"  Self-similarity: {score:.3f}\n")


def test_performance():
    """Test performance metrics"""
    print("=" * 60)
    print("TEST 5: Performance Metrics")
    print("=" * 60)

    import time

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Test different model sizes
    models = [
        ("facebook/dinov2-small", "DINOv2-Small"),
        ("facebook/dinov2-base", "DINOv2-Base"),
    ]

    # Create test images
    img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    for model_name, display_name in models:
        print(f"\n{display_name}:")
        print("-" * 40)

        model = DINOv2EmbeddingModel(model_name, device=device)

        # Warmup
        _ = model.get_similarity_score(img1, img2)

        # Time patch extraction
        start = time.time()
        patches, embeddings = model.extract_patch_embeddings(img1)
        extract_time = time.time() - start

        print(f"Patch extraction: {extract_time*1000:.1f}ms")
        print(f"Patches: {len(patches)}, Embedding dim: {embeddings.shape[1]}")

        # Time full similarity computation
        start = time.time()
        score = model.get_similarity_score(img1, img2)
        similarity_time = time.time() - start

        print(f"Full similarity: {similarity_time*1000:.1f}ms")
        print(f"Score: {score:.3f}")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DINOv2 Patch-Level Matching Tests" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    try:
        test_basic_usage()
        print("\n")

        test_patch_extraction()
        print("\n")

        test_patch_matcher_wrapper()
        print("\n")

        test_with_real_images()
        print("\n")

        test_performance()
        print("\n")

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
