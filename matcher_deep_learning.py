"""
Option 2: Deep Learning Feature Extraction (Better Performance)
Uses proven pretrained models for robust pattern-based matching:
- ResNet50 (fallback, no extra dependencies)
- MegaDescriptor-S (medium, 28M params)
- MegaDescriptor-L (SOTA, 228M params)

Includes preprocessing with Gaussian blur and Otsu thresholding for robust pattern extraction.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from preprocessing import PatternPreprocessor


class DeepFeatureMatcher:
    """
    Cow pattern matcher using deep learning features.
    Supports multiple pretrained models for feature extraction.
    Better performance but requires GPU/MPS for optimal speed.
    """

    def __init__(self, device, model_name='resnet50', similarity_threshold=0.7, use_preprocessing=True):
        """
        Initialize the deep feature matcher.

        Args:
            device (torch.device): Device to run inference on
            model_name (str): Model to use ('resnet50', 'megadescriptor-s', 'megadescriptor-l')
            similarity_threshold (float): Cosine similarity threshold (-1 to 1, typically 0.3-0.9)
            use_preprocessing (bool): Whether to use Gaussian + Otsu preprocessing
        """
        self.device = device
        self.model_name = model_name.lower()
        self.similarity_threshold = similarity_threshold
        self.use_preprocessing = use_preprocessing
        self.preprocessor = PatternPreprocessor() if use_preprocessing else None

        # Load the specified model
        self.model, self.transform, self.feature_dim = self._load_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        self.reference_features = None
        self.reference_image = None
        self.reference_pattern = None

    def _load_model(self, model_name):
        """
        Load the specified pretrained model.

        Args:
            model_name (str): Model name

        Returns:
            tuple: (model, transform, feature_dim)
        """
        model_name = model_name.lower()

        if model_name == 'resnet50':
            return self._load_resnet50()
        elif model_name == 'megadescriptor-s':
            return self._load_megadescriptor_s()
        elif model_name == 'megadescriptor-l':
            return self._load_megadescriptor_l()
        else:
            print(f"Unknown model '{model_name}', falling back to ResNet50")
            return self._load_resnet50()

    def _load_resnet50(self):
        """Load ResNet50 (no extra dependencies)."""
        from torchvision.models import resnet50, ResNet50_Weights

        print("Loading ResNet50...")
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)

        # Remove classifier to get features
        model = nn.Sequential(*list(model.children())[:-1])

        # Transform for ResNet50
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        feature_dim = 2048
        print(f"✓ ResNet50 loaded (feature dim: {feature_dim})")
        return model, transform, feature_dim

    def _load_megadescriptor_s(self):
        """Load MegaDescriptor-S (28M params, requires timm)."""
        try:
            import timm
        except ImportError:
            print("ERROR: timm not installed. Install with: pip install timm")
            print("Falling back to ResNet50...")
            return self._load_resnet50()

        print("Loading MegaDescriptor-S...")
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-S-224", pretrained=True)
        model.eval()

        # Transform for MegaDescriptor-S
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

        feature_dim = model.num_features if hasattr(model, 'num_features') else 768
        print(f"✓ MegaDescriptor-S loaded (feature dim: {feature_dim})")
        return model, transform, feature_dim

    def _load_megadescriptor_l(self):
        """Load MegaDescriptor-L (228M params, SOTA, requires timm)."""
        try:
            import timm
        except ImportError:
            print("ERROR: timm not installed. Install with: pip install timm")
            print("Falling back to ResNet50...")
            return self._load_resnet50()

        print("Loading MegaDescriptor-L (large model, may take time)...")
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        model.eval()

        # Transform for MegaDescriptor-L (384x384)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

        feature_dim = model.num_features if hasattr(model, 'num_features') else 1024
        print(f"✓ MegaDescriptor-L loaded (feature dim: {feature_dim})")
        return model, transform, feature_dim

    def _extract_features(self, image):
        """
        Extract normalized feature vector from image.

        Args:
            image (np.ndarray): Input image (BGR or grayscale)

        Returns:
            np.ndarray: Normalized feature vector
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            # Grayscale to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Transform and add batch dimension
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)

            # Handle different output formats
            if len(features.shape) == 4:
                # (batch, channels, 1, 1) -> flatten
                features = features.view(features.size(0), -1)
            elif len(features.shape) == 3:
                # (batch, seq_len, dim) -> take mean
                features = features.mean(dim=1)

        # Convert to numpy and normalize
        features = features.cpu().numpy().flatten()
        features = features / (np.linalg.norm(features) + 1e-8)  # L2 normalize

        return features

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

        try:
            # Preprocess the image to extract pattern
            if self.use_preprocessing:
                pattern = self.preprocessor.preprocess(reference_image, enhance_contrast=True)
                if pattern is None:
                    return False
            else:
                # Use original image
                pattern = reference_image

            # Extract features
            features = self._extract_features(pattern)

            self.reference_features = features
            self.reference_image = reference_image
            self.reference_pattern = pattern if self.use_preprocessing else None

            print(f"✓ Reference pattern set (feature dim: {len(features)})")
            return True

        except Exception as e:
            print(f"Error setting reference pattern: {e}")
            return False

    def match_pattern(self, query_image):
        """
        Match the query image against the reference pattern.
        Applies same preprocessing to query image for consistent comparison.

        Args:
            query_image (np.ndarray): Query image to match (BGR)

        Returns:
            tuple: (is_match, similarity)
                - is_match (bool): True if pattern matches
                - similarity (float): Cosine similarity score (-1 to 1)
        """
        if self.reference_features is None:
            return False, 0.0

        if query_image is None or query_image.size == 0:
            return False, 0.0

        try:
            # Preprocess query image using same method as reference
            if self.use_preprocessing:
                pattern = self.preprocessor.preprocess(query_image, enhance_contrast=True)
                if pattern is None:
                    return False, 0.0
            else:
                pattern = query_image

            # Extract features
            query_features = self._extract_features(pattern)

            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(self.reference_features, query_features)

            # Clamp to [-1, 1] range (should already be, but just in case)
            similarity = np.clip(similarity, -1.0, 1.0)

            is_match = similarity >= self.similarity_threshold

            return is_match, float(similarity)

        except Exception as e:
            print(f"Error matching pattern: {e}")
            return False, 0.0

    def visualize_features(self, query_image):
        """
        Create visualization comparing reference and query features.

        Args:
            query_image (np.ndarray): Query image (BGR)

        Returns:
            np.ndarray: Side-by-side comparison of reference and query
        """
        if self.reference_image is None or query_image is None:
            return None

        # Resize images to same height
        h = min(self.reference_image.shape[0], query_image.shape[0])
        ref_resized = cv2.resize(self.reference_image,
                                (int(self.reference_image.shape[1] * h / self.reference_image.shape[0]), h))
        query_resized = cv2.resize(query_image,
                                   (int(query_image.shape[1] * h / query_image.shape[0]), h))

        # Concatenate horizontally
        comparison = np.hstack([ref_resized, query_resized])

        # Add similarity score if available
        is_match, similarity = self.match_pattern(query_image)
        text = f"Similarity: {similarity:.3f}"
        color = (0, 255, 0) if is_match else (0, 0, 255)

        cv2.putText(
            comparison,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        return comparison

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
            return image

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            dict: Model information
        """
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'preprocessing': self.use_preprocessing,
            'similarity_threshold': self.similarity_threshold
        }
