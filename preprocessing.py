"""
Image preprocessing utilities for cow pattern extraction.
Uses Gaussian blur and Otsu thresholding to extract skin patterns
and reduce lighting/shadow issues.
"""

import cv2
import numpy as np


class PatternPreprocessor:
    """
    Preprocesses cow images to extract skin patterns using Gaussian blur
    and Otsu thresholding. This normalizes lighting conditions and enhances
    pattern matching accuracy.
    """

    def __init__(self, gaussian_kernel=(5, 5), gaussian_sigma=1.0):
        """
        Initialize the pattern preprocessor.

        Args:
            gaussian_kernel (tuple): Kernel size for Gaussian blur (must be odd)
            gaussian_sigma (float): Standard deviation for Gaussian blur
        """
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def preprocess(self, image, enhance_contrast=True):
        """
        Preprocess image to extract skin pattern using Gaussian blur and Otsu thresholding.

        Args:
            image (np.ndarray): Input image (BGR format)
            enhance_contrast (bool): Whether to apply CLAHE for contrast enhancement

        Returns:
            np.ndarray: Binary pattern image (0 or 255)
        """
        if image is None or image.size == 0:
            return None

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Optional: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            self.gaussian_kernel,
            self.gaussian_sigma
        )

        # Apply Otsu's thresholding to create binary pattern
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    def preprocess_with_morphology(self, image, enhance_contrast=True,
                                   morph_kernel_size=3, apply_opening=True):
        """
        Advanced preprocessing with morphological operations to clean up the pattern.

        Args:
            image (np.ndarray): Input image (BGR format)
            enhance_contrast (bool): Whether to apply CLAHE
            morph_kernel_size (int): Kernel size for morphological operations
            apply_opening (bool): Apply morphological opening to remove noise

        Returns:
            np.ndarray: Cleaned binary pattern image
        """
        # Get basic preprocessed image
        binary = self.preprocess(image, enhance_contrast=enhance_contrast)

        if binary is None:
            return None

        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )

        if apply_opening:
            # Morphological opening (erosion followed by dilation)
            # Removes small white noise
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Morphological closing (dilation followed by erosion)
        # Fills small black holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def preprocess_rgb_channels(self, image):
        """
        Preprocess each RGB channel separately and combine for robust pattern extraction.
        Useful for colored patterns (e.g., brown/white spots on cows).

        Args:
            image (np.ndarray): Input image (BGR format)

        Returns:
            np.ndarray: Combined binary pattern from all channels
        """
        if image is None or image.size == 0:
            return None

        # Split into channels
        b, g, r = cv2.split(image)

        # Process each channel
        patterns = []
        for channel in [b, g, r]:
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(channel)

            # Blur and threshold
            blurred = cv2.GaussianBlur(
                enhanced,
                self.gaussian_kernel,
                self.gaussian_sigma
            )
            _, binary = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            patterns.append(binary)

        # Combine channels using bitwise AND to get common patterns
        # This makes the pattern more robust
        combined = cv2.bitwise_and(patterns[0], patterns[1])
        combined = cv2.bitwise_and(combined, patterns[2])

        return combined

    def visualize_preprocessing_steps(self, image):
        """
        Visualize all preprocessing steps for debugging and parameter tuning.

        Args:
            image (np.ndarray): Input image (BGR format)

        Returns:
            np.ndarray: Combined visualization showing all steps
        """
        if image is None or image.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blurred
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, self.gaussian_sigma)

        # Otsu threshold
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

        # Resize all to same size for visualization
        h, w = gray.shape
        target_h, target_w = h // 2, w // 2

        # Create grid
        row1 = np.hstack([
            cv2.resize(gray, (target_w, target_h)),
            cv2.resize(enhanced, (target_w, target_h))
        ])
        row2 = np.hstack([
            cv2.resize(blurred, (target_w, target_h)),
            cv2.resize(otsu, (target_w, target_h))
        ])

        # Add labels
        row1_labeled = row1.copy()
        row2_labeled = row2.copy()

        cv2.putText(row1_labeled, "1. Original (Gray)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(row1_labeled, "2. CLAHE Enhanced", (target_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(row2_labeled, "3. Gaussian Blur", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(row2_labeled, "4. Otsu Threshold", (target_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        visualization = np.vstack([row1_labeled, row2_labeled])

        return visualization

    def adaptive_preprocess(self, image, method='otsu'):
        """
        Adaptive preprocessing with multiple thresholding methods.

        Args:
            image (np.ndarray): Input image (BGR format)
            method (str): Thresholding method ('otsu', 'adaptive_mean', 'adaptive_gaussian')

        Returns:
            np.ndarray: Binary pattern image
        """
        if image is None or image.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, self.gaussian_kernel, self.gaussian_sigma)

        # Apply selected thresholding method
        if method == 'otsu':
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive_mean':
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'adaptive_gaussian':
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return binary
