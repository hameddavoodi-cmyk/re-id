"""
Option 1: Local Invariant Features (RECOMMENDED - Simple & Robust)
Uses ORB (Oriented FAST and Rotated BRIEF) for keypoint detection and matching.
Includes preprocessing with Gaussian blur and Otsu thresholding for robust pattern extraction.
"""

import cv2
import numpy as np
from preprocessing import PatternPreprocessor


class LocalFeatureMatcher:
    """
    Cow pattern matcher using local invariant features (ORB).
    Fast, robust, and works well for pattern matching without GPU.
    """

    def __init__(self, n_features=2000, match_threshold=0.7, use_preprocessing=True):
        """
        Initialize the local feature matcher.

        Args:
            n_features (int): Maximum number of features to detect
            match_threshold (float): Ratio test threshold for good matches (0-1)
            use_preprocessing (bool): Whether to use Gaussian + Otsu preprocessing
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_threshold = match_threshold
        self.use_preprocessing = use_preprocessing
        self.preprocessor = PatternPreprocessor() if use_preprocessing else None
        self.reference_descriptors = None
        self.reference_keypoints = None
        self.reference_image = None
        self.reference_pattern = None  # Store preprocessed pattern

    def set_reference_pattern(self, reference_image):
        """
        Set the reference cow pattern image.
        Applies preprocessing (Gaussian + Otsu) to extract robust pattern.

        Args:
            reference_image (np.ndarray): Reference pattern image (BGR)

        Returns:
            bool: True if successful, False otherwise
        """
        if reference_image is None or reference_image.size == 0:
            return False

        # Preprocess the image to extract pattern
        if self.use_preprocessing:
            pattern = self.preprocessor.preprocess(reference_image, enhance_contrast=True)
            if pattern is None:
                return False
        else:
            # Use grayscale directly without preprocessing
            pattern = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors on preprocessed pattern
        keypoints, descriptors = self.orb.detectAndCompute(pattern, None)

        if descriptors is None or len(keypoints) < 10:
            return False

        self.reference_image = reference_image
        self.reference_pattern = pattern
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors

        return True

    def match_pattern(self, query_image, min_good_matches=15):
        """
        Match the query image against the reference pattern.
        Applies same preprocessing to query image for consistent comparison.

        Args:
            query_image (np.ndarray): Query image to match (BGR)
            min_good_matches (int): Minimum number of good matches to consider a match

        Returns:
            tuple: (is_match, confidence, num_matches)
                - is_match (bool): True if pattern matches
                - confidence (float): Match confidence score (0-1)
                - num_matches (int): Number of good matches found
        """
        if self.reference_descriptors is None:
            return False, 0.0, 0

        if query_image is None or query_image.size == 0:
            return False, 0.0, 0

        # Preprocess query image using same method as reference
        if self.use_preprocessing:
            pattern = self.preprocessor.preprocess(query_image, enhance_contrast=True)
            if pattern is None:
                return False, 0.0, 0
        else:
            pattern = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors on preprocessed pattern
        keypoints, descriptors = self.orb.detectAndCompute(pattern, None)

        if descriptors is None or len(keypoints) < 10:
            return False, 0.0, 0

        # Match descriptors using KNN
        matches = self.bf_matcher.knnMatch(
            self.reference_descriptors, descriptors, k=2
        )

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)

        # Calculate confidence based on number of matches
        max_possible_matches = min(
            len(self.reference_keypoints), len(keypoints)
        )
        confidence = num_matches / max_possible_matches if max_possible_matches > 0 else 0.0

        is_match = num_matches >= min_good_matches

        return is_match, confidence, num_matches

    def visualize_matches(self, query_image):
        """
        Create visualization of matches between reference and query image.
        Shows matches on preprocessed patterns.

        Args:
            query_image (np.ndarray): Query image (BGR)

        Returns:
            np.ndarray: Visualization image showing matches
        """
        if self.reference_pattern is None or query_image is None:
            return None

        # Preprocess query image
        if self.use_preprocessing:
            pattern = self.preprocessor.preprocess(query_image, enhance_contrast=True)
            if pattern is None:
                return None
        else:
            pattern = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(pattern, None)

        if descriptors is None:
            return None

        matches = self.bf_matcher.knnMatch(
            self.reference_descriptors, descriptors, k=2
        )

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)

        # Convert binary patterns to BGR for visualization
        ref_vis = cv2.cvtColor(self.reference_pattern, cv2.COLOR_GRAY2BGR)
        query_vis = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        # Draw matches
        match_img = cv2.drawMatches(
            ref_vis,
            self.reference_keypoints,
            query_vis,
            keypoints,
            good_matches[:50],  # Limit to top 50 for clarity
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        return match_img

    def visualize_preprocessing(self, image):
        """
        Visualize the preprocessing steps applied to an image.

        Args:
            image (np.ndarray): Input image (BGR)

        Returns:
            np.ndarray: Visualization showing preprocessing steps
        """
        if not self.use_preprocessing or self.preprocessor is None:
            return None

        return self.preprocessor.visualize_preprocessing_steps(image)

    def get_preprocessed_pattern(self, image):
        """
        Get the preprocessed pattern for an image.

        Args:
            image (np.ndarray): Input image (BGR)

        Returns:
            np.ndarray: Preprocessed binary pattern
        """
        if self.use_preprocessing:
            return self.preprocessor.preprocess(image, enhance_contrast=True)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
