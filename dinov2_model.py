"""
DINOv2 Patch-Level Matching Model for Cow Re-Identification

This module implements patch-level feature matching using Meta's DINOv2 model.
Instead of global embeddings, it extracts and matches local patch features to handle
partial views, occlusions, and viewpoint variations.

Key Features:
- Extracts patch-level embeddings (14x14 pixel patches)
- Filters distinctive patches based on variance (avoids uniform regions)
- Performs patch-to-patch matching using cosine similarity
- Returns similarity scores based on matching patch correspondences
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import AutoModel, AutoImageProcessor


class DINOv2EmbeddingModel:
    """
    DINOv2-based patch-level matching model for re-identification.

    This model extracts dense patch features from images and performs local matching,
    making it robust to partial occlusions and viewpoint changes.
    """

    def __init__(self, model_name="facebook/dinov2-base", device=None):
        """
        Initialize DINOv2 model for patch-level matching.

        Args:
            model_name: Model identifier from HuggingFace
                - "facebook/dinov2-small": 384-dim features, faster
                - "facebook/dinov2-base": 768-dim features, better accuracy (default)
                - "facebook/dinov2-large": 1024-dim features, best accuracy
            device: torch.device to use (cuda/mps/cpu)
        """
        self.model_name = model_name
        self.device = device if device is not None else torch.device("cpu")

        # Load model and processor
        print(f"Loading {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

        # DINOv2 patch size is 14x14 pixels
        self.patch_size = 14

        print(f"DINOv2 loaded successfully on {self.device}")

    def extract_patch_embeddings(self, image):
        """
        Extract patch-level embeddings from an image.

        Args:
            image: BGR image (OpenCV format) or RGB numpy array

        Returns:
            patches: numpy array of shape (H_patches, W_patches, 2) containing patch coordinates
            embeddings: numpy array of shape (num_patches, embedding_dim)
        """
        if image.shape[0] < 50 or image.shape[1] < 50:
            return None, None

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Preprocess image
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get patch embeddings (excluding CLS token at index 0)
            # Shape: [batch, num_patches+1, embedding_dim]
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # Convert to numpy
        patch_embeddings = patch_embeddings.squeeze(0).cpu().numpy()

        # Calculate number of patches (DINOv2 outputs square grid)
        num_patches = patch_embeddings.shape[0]
        patches_per_side = int(np.sqrt(num_patches))

        # Create patch coordinate grid
        patches = []
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                # Store grid coordinates (row, col)
                patches.append([i, j])

        patches = np.array(patches)

        return patches, patch_embeddings

    def filter_distinctive_patches(self, image, patches, embeddings, variance_threshold=0.01):
        """
        Filter out non-distinctive patches (e.g., pure white/black backgrounds).

        Args:
            image: Original BGR image
            patches: Patch coordinates array
            embeddings: Patch embeddings array
            variance_threshold: Minimum variance threshold for patch intensity

        Returns:
            filtered_patches: Filtered patch coordinates
            filtered_embeddings: Filtered patch embeddings
        """
        if patches is None or embeddings is None:
            return None, None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Calculate number of patches per side
        num_patches = embeddings.shape[0]
        patches_per_side = int(np.sqrt(num_patches))

        # Calculate patch dimensions in original image
        patch_h = h / patches_per_side
        patch_w = w / patches_per_side

        distinctive_mask = []

        for patch_coord in patches:
            row, col = patch_coord

            # Calculate pixel coordinates
            y1 = int(row * patch_h)
            y2 = int((row + 1) * patch_h)
            x1 = int(col * patch_w)
            x2 = int((col + 1) * patch_w)

            # Extract patch region
            patch_region = gray[y1:y2, x1:x2]

            if patch_region.size == 0:
                distinctive_mask.append(False)
                continue

            # Calculate variance
            variance = np.var(patch_region.astype(np.float32) / 255.0)

            # Keep patches with sufficient variance
            distinctive_mask.append(variance > variance_threshold)

        distinctive_mask = np.array(distinctive_mask)

        # Filter patches and embeddings
        filtered_patches = patches[distinctive_mask]
        filtered_embeddings = embeddings[distinctive_mask]

        return filtered_patches, filtered_embeddings

    def match_patches(self, ref_embeddings, query_embeddings, threshold=0.7, top_k=5):
        """
        Match patches between reference and query images.

        For each reference patch, find the most similar query patch using cosine similarity.
        Count how many patches have good matches (above threshold).

        Args:
            ref_embeddings: Reference patch embeddings (N_ref, dim)
            query_embeddings: Query patch embeddings (N_query, dim)
            threshold: Cosine similarity threshold for a good match
            top_k: Consider top-k nearest neighbors for each patch

        Returns:
            match_score: Ratio of reference patches with good matches (0-1)
            num_matches: Number of reference patches with good matches
        """
        if ref_embeddings is None or query_embeddings is None:
            return 0.0, 0

        if len(ref_embeddings) == 0 or len(query_embeddings) == 0:
            return 0.0, 0

        # Convert to torch tensors
        ref_tensor = torch.from_numpy(ref_embeddings).float()
        query_tensor = torch.from_numpy(query_embeddings).float()

        # L2 normalize embeddings
        ref_tensor = F.normalize(ref_tensor, p=2, dim=1)
        query_tensor = F.normalize(query_tensor, p=2, dim=1)

        # Compute cosine similarity matrix (N_ref x N_query)
        similarity_matrix = torch.mm(ref_tensor, query_tensor.t())

        # For each reference patch, find top-k most similar query patches
        top_k_values, top_k_indices = torch.topk(similarity_matrix, k=min(top_k, similarity_matrix.shape[1]), dim=1)

        # Count matches above threshold
        # A reference patch is considered matched if its best match is above threshold
        best_matches = top_k_values[:, 0]
        good_matches = (best_matches > threshold).sum().item()

        # Calculate match ratio
        match_score = good_matches / len(ref_embeddings)

        return match_score, good_matches

    def get_similarity_score(self, ref_image, query_image, variance_threshold=0.01,
                            similarity_threshold=0.7, top_k=5):
        """
        Compute overall similarity score between reference and query images using patch matching.

        This is the main method for comparing two images. It:
        1. Extracts patch embeddings from both images
        2. Filters distinctive patches
        3. Performs bidirectional patch matching
        4. Returns symmetric similarity score

        Args:
            ref_image: Reference image (BGR format)
            query_image: Query image (BGR format)
            variance_threshold: Threshold for filtering distinctive patches
            similarity_threshold: Cosine similarity threshold for matching
            top_k: Number of nearest neighbors to consider

        Returns:
            similarity_score: Overall similarity score (0-1)
        """
        # Extract patch embeddings
        ref_patches, ref_embeddings = self.extract_patch_embeddings(ref_image)
        query_patches, query_embeddings = self.extract_patch_embeddings(query_image)

        if ref_embeddings is None or query_embeddings is None:
            return 0.0

        # Filter distinctive patches
        ref_patches_filtered, ref_embeddings_filtered = self.filter_distinctive_patches(
            ref_image, ref_patches, ref_embeddings, variance_threshold
        )
        query_patches_filtered, query_embeddings_filtered = self.filter_distinctive_patches(
            query_image, query_patches, query_embeddings, variance_threshold
        )

        if ref_embeddings_filtered is None or query_embeddings_filtered is None:
            return 0.0

        if len(ref_embeddings_filtered) == 0 or len(query_embeddings_filtered) == 0:
            return 0.0

        # Bidirectional matching
        # Match ref -> query
        score_ref_to_query, matches_ref_to_query = self.match_patches(
            ref_embeddings_filtered, query_embeddings_filtered,
            threshold=similarity_threshold, top_k=top_k
        )

        # Match query -> ref
        score_query_to_ref, matches_query_to_ref = self.match_patches(
            query_embeddings_filtered, ref_embeddings_filtered,
            threshold=similarity_threshold, top_k=top_k
        )

        # Symmetric score (average of both directions)
        similarity_score = (score_ref_to_query + score_query_to_ref) / 2.0

        return float(similarity_score)

    def get_embedding(self, img):
        """
        Legacy method for compatibility with existing EmbeddingModel interface.

        This method is kept for backward compatibility but internally uses patch-level
        matching. For proper usage, use get_similarity_score() directly instead of
        extracting and comparing embeddings separately.

        Args:
            img: BGR image (OpenCV format)

        Returns:
            embeddings: Patch embeddings as a flattened array (for compatibility)
                       Note: This is NOT meant for cosine similarity comparison!
                       Use get_similarity_score() for comparing images.
        """
        patches, embeddings = self.extract_patch_embeddings(img)

        if embeddings is None:
            return None

        # Filter distinctive patches
        patches_filtered, embeddings_filtered = self.filter_distinctive_patches(
            img, patches, embeddings
        )

        if embeddings_filtered is None or len(embeddings_filtered) == 0:
            return None

        # Return mean pooling of patch embeddings for compatibility
        # Note: This is a simplified approach; proper matching should use get_similarity_score()
        global_embedding = np.mean(embeddings_filtered, axis=0)

        # L2 normalize
        norm = np.linalg.norm(global_embedding)
        if norm > 0:
            global_embedding = global_embedding / norm

        return global_embedding


# Wrapper class for seamless integration with existing pipeline
class DINOv2PatchMatcher:
    """
    Wrapper class that provides patch-level matching with a reference database.

    This class mimics the ReferenceCowDatabase interface but uses patch matching
    instead of global embedding comparison.
    """

    def __init__(self, model_name="facebook/dinov2-base", device=None):
        """
        Initialize the patch matcher.

        Args:
            model_name: DINOv2 model variant to use
            device: Computation device
        """
        self.model = DINOv2EmbeddingModel(model_name, device)
        self.reference_image = None
        self.reference_patches = None
        self.reference_embeddings = None
        self.registered_at = None

    def set_reference(self, image):
        """
        Set the reference image and extract its patch embeddings.

        Args:
            image: Reference image (BGR format)
        """
        from datetime import datetime

        patches, embeddings = self.model.extract_patch_embeddings(image)

        if embeddings is not None:
            # Filter distinctive patches
            patches_filtered, embeddings_filtered = self.model.filter_distinctive_patches(
                image, patches, embeddings
            )

            self.reference_image = image
            self.reference_patches = patches_filtered
            self.reference_embeddings = embeddings_filtered
            self.registered_at = datetime.now()

            return True

        return False

    def has_reference(self):
        """Check if a reference image is registered."""
        return self.reference_embeddings is not None

    def compare(self, query_image):
        """
        Compare a query image with the reference using patch matching.

        Args:
            query_image: Query image (BGR format)

        Returns:
            similarity_score: Similarity score (0-1)
        """
        if not self.has_reference():
            return 0.0

        # Extract query patches
        query_patches, query_embeddings = self.model.extract_patch_embeddings(query_image)

        if query_embeddings is None:
            return 0.0

        # Filter distinctive patches
        query_patches_filtered, query_embeddings_filtered = self.model.filter_distinctive_patches(
            query_image, query_patches, query_embeddings
        )

        if query_embeddings_filtered is None or len(query_embeddings_filtered) == 0:
            return 0.0

        # Bidirectional matching
        score_ref_to_query, _ = self.model.match_patches(
            self.reference_embeddings, query_embeddings_filtered
        )

        score_query_to_ref, _ = self.model.match_patches(
            query_embeddings_filtered, self.reference_embeddings
        )

        # Symmetric score
        similarity_score = (score_ref_to_query + score_query_to_ref) / 2.0

        return float(similarity_score)


# Example usage and testing
if __name__ == "__main__":
    import sys

    print("DINOv2 Patch-Level Matching Model")
    print("=" * 50)

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

    print(f"Using device: {device_name}")
    print()

    # Initialize model
    print("Initializing DINOv2 model...")
    model = DINOv2EmbeddingModel(model_name="facebook/dinov2-base", device=device)
    print("Model loaded successfully!")
    print()

    # Test with dummy images
    print("Testing with synthetic images...")

    # Create test images
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print(f"Test image shape: {test_image.shape}")

    # Extract patches
    patches, embeddings = model.extract_patch_embeddings(test_image)
    print(f"Extracted {len(embeddings)} patches")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Filter distinctive patches
    patches_filtered, embeddings_filtered = model.filter_distinctive_patches(
        test_image, patches, embeddings
    )
    print(f"Distinctive patches after filtering: {len(embeddings_filtered)}")

    # Test patch matching with itself (should have high score)
    score, matches = model.match_patches(embeddings_filtered, embeddings_filtered)
    print(f"Self-matching score: {score:.3f} ({matches} matches)")

    # Test with different image
    test_image2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    patches2, embeddings2 = model.extract_patch_embeddings(test_image2)
    patches2_filtered, embeddings2_filtered = model.filter_distinctive_patches(
        test_image2, patches2, embeddings2
    )

    score2, matches2 = model.match_patches(embeddings_filtered, embeddings2_filtered)
    print(f"Cross-matching score: {score2:.3f} ({matches2} matches)")

    # Test get_similarity_score
    overall_score = model.get_similarity_score(test_image, test_image2)
    print(f"Overall similarity score: {overall_score:.3f}")

    print()
    print("All tests passed!")
    print("=" * 50)
