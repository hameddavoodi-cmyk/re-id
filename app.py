"""
Streamlit App for Cow Re-Identification with YOLO Segmentation + Deep Embeddings
Pipeline: 1 FPS sampling -> YOLO segment cow bodies -> embedding extraction -> TOP-1 matching
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import torchvision.transforms as T
from datetime import datetime
import pickle
import json
from pathlib import Path
from ultralytics import YOLO

# Check device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS"
    else:
        return torch.device("cpu"), "CPU"


# Preprocessing for reference images
def preprocess_reference(img):
    """Preprocess reference pattern image for better matching"""
    # Resize to consistent size
    img = cv2.resize(img, (256, 256))

    # Convert to LAB and enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# Embedding model loader
class EmbeddingModel:
    def __init__(self, model_name, device, use_patch_matching=False):
        self.model_name = model_name
        self.device = device
        self.use_patch_matching = use_patch_matching
        self.is_dinov2 = "dinov2" in model_name
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        if self.model_name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        elif self.model_name == "megadescriptor-s":
            import timm
            self.model = timm.create_model("hf-hub:BVRA/MegaDescriptor-S-224", pretrained=True)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        elif self.model_name == "megadescriptor-l":
            import timm
            self.model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        elif self.model_name.startswith("dinov2"):
            # Use the DINOv2 implementation from dinov2_model.py
            from dinov2_model import DINOv2EmbeddingModel

            # Map model names
            model_map = {
                "dinov2_vits14": "facebook/dinov2-small",
                "dinov2_vitb14": "facebook/dinov2-base",
                "dinov2_vitl14": "facebook/dinov2-large"
            }

            hf_model_name = model_map.get(self.model_name, "facebook/dinov2-base")
            self.dinov2_impl = DINOv2EmbeddingModel(hf_model_name, str(self.device))

            # Set placeholder for compatibility
            self.model = self.dinov2_impl.model
            self.transform = None  # DINOv2 handles its own transform
            return  # Skip eval and to(device) - already handled

        self.model.eval()
        self.model.to(self.device)

    def get_embedding(self, img):
        """Extract embedding from image (BGR format)"""
        if img.shape[0] < 50 or img.shape[1] < 50:
            return None

        # Handle DINOv2 models separately
        if self.is_dinov2:
            # Extract patches and embeddings
            patches, patch_embeddings = self.dinov2_impl.extract_patch_embeddings(img)

            if patches is None or patch_embeddings is None:
                return None

            # L2 normalize each patch embedding
            patch_embeddings = patch_embeddings / np.linalg.norm(patch_embeddings, axis=1, keepdims=True)

            # Calculate grid shape
            num_patches = patch_embeddings.shape[0]
            grid_size = int(np.sqrt(num_patches))
            grid_shape = (grid_size, grid_size)

            # Compute CLS token as mean of all patches (for compatibility)
            cls_token = np.mean(patch_embeddings, axis=0)
            cls_token = cls_token / np.linalg.norm(cls_token)

            if self.use_patch_matching:
                # Return full patch embedding info
                return {
                    'type': 'patch',
                    'cls': cls_token,
                    'patches': patch_embeddings,
                    'grid_shape': grid_shape,
                    'embed_dim': patch_embeddings.shape[1]
                }
            else:
                # Return only CLS token for backward compatibility
                return cls_token

        # Handle other models (existing behavior)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor).squeeze().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb


# Extract cow body using segmentation mask
def extract_cow_body(frame, mask):
    """
    Extract cow body from frame using segmentation mask
    Returns: masked cow image (background removed)
    """
    if mask is None or len(mask) == 0:
        return None

    # Create binary mask
    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Convert mask coordinates to polygon
    if hasattr(mask, 'xy'):
        pts = mask.xy[0].astype(np.int32)
    else:
        pts = mask.astype(np.int32)

    cv2.fillPoly(mask_img, [pts], 255)

    # Apply mask to frame
    masked = cv2.bitwise_and(frame, frame, mask=mask_img)

    # Find bounding box of mask
    x, y, w, h = cv2.boundingRect(mask_img)

    # Crop to bounding box
    cropped = masked[y:y+h, x:x+w]

    return cropped, (x, y, w, h)


# Reference cow database
class ReferenceCowDatabase:
    def __init__(self):
        self.reference_embedding = None
        self.reference_thumbnail = None
        self.registered_at = None
        self.is_patch_based = False
        # Multi-view support
        self.reference_views = []  # List of embeddings from different views
        self.view_thumbnails = []  # Thumbnails for each view
        self.use_multi_view = False

    def set_reference(self, embedding, thumbnail, is_patch_based=False):
        """Set the reference (ALPHA) cow - single view"""
        self.reference_embedding = embedding
        self.reference_thumbnail = cv2.resize(thumbnail, (100, 100))
        self.registered_at = datetime.now()
        self.is_patch_based = is_patch_based
        self.use_multi_view = False
        # Clear multi-view data when setting single reference
        self.reference_views = []
        self.view_thumbnails = []

    def set_multi_view_reference(self, embeddings, thumbnails, is_patch_based=False):
        """Set multiple reference views for the same cow (e.g., left, top, right)"""
        self.reference_views = embeddings
        self.view_thumbnails = [cv2.resize(t, (100, 100)) for t in thumbnails]
        self.registered_at = datetime.now()
        self.is_patch_based = is_patch_based
        self.use_multi_view = True
        # Set first view as default reference
        self.reference_embedding = embeddings[0]
        self.reference_thumbnail = self.view_thumbnails[0]

    def has_reference(self):
        if self.use_multi_view:
            return len(self.reference_views) > 0
        return self.reference_embedding is not None

    def compare(self, embedding):
        """
        Compare embedding with reference(s)
        Returns: similarity score (0-1)

        If multi-view is enabled, compares against all views and returns MAXIMUM score
        (best matching view wins)
        """
        if not self.has_reference():
            return 0.0

        # Multi-view matching: compare against all views, return best score
        if self.use_multi_view:
            scores = []
            for ref_view in self.reference_views:
                score = self._compare_single(ref_view, embedding)
                scores.append(score)
            # Return maximum score (best matching view)
            return float(max(scores)) if scores else 0.0
        else:
            # Single view matching
            return self._compare_single(self.reference_embedding, embedding)

    def _compare_single(self, ref_embedding, query_embedding):
        """Compare a single reference embedding with query embedding"""
        # Check if both are patch embeddings
        ref_is_patch = isinstance(ref_embedding, dict)
        query_is_patch = isinstance(query_embedding, dict)

        if ref_is_patch and query_is_patch:
            # Patch-level matching using bidirectional patch similarity
            ref_patches = ref_embedding['patches']
            query_patches = query_embedding['patches']

            # Compute bidirectional similarity (cosine similarity between patch sets)
            # Forward: ref -> query
            similarities_fwd = ref_patches @ query_patches.T  # (N_ref, N_query)
            max_fwd = np.max(similarities_fwd, axis=1)  # Best match for each ref patch
            score_fwd = np.mean(max_fwd)

            # Backward: query -> ref
            similarities_bwd = query_patches @ ref_patches.T  # (N_query, N_ref)
            max_bwd = np.max(similarities_bwd, axis=1)  # Best match for each query patch
            score_bwd = np.mean(max_bwd)

            # Average both directions for symmetry
            score = (score_fwd + score_bwd) / 2.0
            return float(score)

        elif ref_is_patch != query_is_patch:
            # Type mismatch - cannot compare
            return 0.0

        else:
            # Global embedding comparison (existing behavior)
            return float(np.dot(query_embedding, ref_embedding))


# Page configuration
st.set_page_config(
    page_title="Cow Re-ID System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def process_frame(frame, yolo, embedding_model, reference_db, detection_conf, use_segmentation=True, use_class_filter=True):
    """
    Process a single frame and return annotated frame + detected cows data
    Args:
        use_segmentation: If True, use segmentation masks; if False, use bounding boxes only
        use_class_filter: If True, filter for class 19 (cow); if False, detect all classes (for custom models)
    """
    # YOLO detection/segmentation
    if use_class_filter:
        results = yolo(frame, classes=[19], conf=detection_conf, verbose=False)  # class 19 = cow
    else:
        results = yolo(frame, conf=detection_conf, verbose=False)  # Custom model (already trained on cows)

    # Extract all detected cows with embeddings
    cow_data = []

    for result in results:
        # Check if we should use segmentation masks
        has_masks = result.masks is not None and use_segmentation

        if has_masks:
            # Segmentation mode: use masks
            for idx, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = box.conf[0].item()

                # Extract cow body using segmentation mask
                cow_body_result = extract_cow_body(frame, mask)

                if cow_body_result is None:
                    # Fallback to bounding box if segmentation fails
                    crop = frame[y1:y2, x1:x2]
                    bbox = (x1, y1, x2-x1, y2-y1)
                else:
                    crop, bbox_relative = cow_body_result
                    # Convert relative bbox to absolute
                    bbox = (x1, y1, x2-x1, y2-y1)

                if crop.shape[0] < 50 or crop.shape[1] < 50:
                    continue

                # Get embedding
                emb = embedding_model.get_embedding(crop)
                if emb is None:
                    continue

                # Compare with reference (if exists)
                if reference_db.has_reference():
                    score = reference_db.compare(emb)
                else:
                    score = 0.0

                cow_data.append({
                    'bbox': bbox,
                    'embedding': emb,
                    'score': score,
                    'crop': crop,
                    'confidence': conf
                })
        else:
            # Detection mode: use bounding boxes only
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = box.conf[0].item()

                # Extract crop from bounding box
                crop = frame[y1:y2, x1:x2]
                bbox = (x1, y1, x2-x1, y2-y1)

                if crop.shape[0] < 50 or crop.shape[1] < 50:
                    continue

                # Get embedding
                emb = embedding_model.get_embedding(crop)
                if emb is None:
                    continue

                # Compare with reference (if exists)
                if reference_db.has_reference():
                    score = reference_db.compare(emb)
                else:
                    score = 0.0

                cow_data.append({
                    'bbox': bbox,
                    'embedding': emb,
                    'score': score,
                    'crop': crop,
                    'confidence': conf
                })

    # Sort by score (highest first) to determine TOP-1
    cow_data.sort(key=lambda x: x['score'], reverse=True)

    # Assign IDs: TOP-1 = ALPHA, others = None
    for idx, cow in enumerate(cow_data):
        if idx == 0:
            cow['id'] = 'ALPHA'
            cow['color'] = (0, 255, 0)  # Green
            cow['thickness'] = 3
        else:
            cow['id'] = 'None'
            cow['color'] = (128, 128, 128)  # Gray
            cow['thickness'] = 2

    # Draw on frame
    annotated_frame = frame.copy()
    for cow in cow_data:
        x, y, w, h = cow['bbox']
        color = cow['color']
        thickness = cow['thickness']

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, thickness)

        # Draw label with background
        label = f"{cow['id']} ({cow['score']:.3f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(annotated_frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(annotated_frame, label, (x, y - 5), font, font_scale, (0, 0, 0), font_thickness)

    return annotated_frame, cow_data


def main():
    st.title("Cow Re-Identification System")
    st.markdown("YOLO Segmentation + Deep Embedding + TOP-1 Matching")
    st.markdown("---")

    # Initialize session state
    if 'reference_db' not in st.session_state:
        st.session_state.reference_db = ReferenceCowDatabase()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Device info
        device, device_name = get_device()
        st.success(f"**Device:** {device_name}")

        st.markdown("---")

        # Model selection
        st.subheader("Embedding Model")
        model_name = st.selectbox(
            "Choose Model",
            ["resnet50", "megadescriptor-s", "megadescriptor-l", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
            index=1,  # Default to MegaDescriptor-S
            format_func=lambda x: {
                "resnet50": "ResNet50 (Fast, 2048-dim)",
                "megadescriptor-s": "MegaDescriptor-S (Recommended)",
                "megadescriptor-l": "MegaDescriptor-L (Best Accuracy)",
                "dinov2_vits14": "DINOv2 ViT-S/14 (Patch Matching)",
                "dinov2_vitb14": "DINOv2 ViT-B/14 (Patch Matching)",
                "dinov2_vitl14": "DINOv2 ViT-L/14 (Best Patches)"
            }[x],
            help="DINOv2 models support patch-level matching for fine-grained pattern recognition"
        )

        # DINOv2-specific settings
        use_patch_matching = False
        if "dinov2" in model_name:
            st.markdown("**DINOv2 Settings:**")
            use_patch_matching = st.checkbox(
                "Enable Patch-Level Matching",
                value=True,
                help="Match local patches instead of global embedding for better viewpoint invariance"
            )

            if use_patch_matching:
                st.info("Patch matching: Compares distinctive local patterns (spots/markings) instead of whole cow appearance")
        else:
            use_patch_matching = False

        st.markdown("---")

        # YOLO settings
        st.subheader("YOLO Detection/Segmentation")

        # Model type selection
        yolo_type = st.radio(
            "YOLO Type:",
            ["Detection", "Segmentation"],
            horizontal=True,
            help="Detection uses bounding boxes, Segmentation extracts precise cow body"
        )

        # Model selection
        model_source = st.radio(
            "Model Source:",
            ["Pretrained", "Custom Path"],
            horizontal=True
        )

        if model_source == "Pretrained":
            if yolo_type == "Detection":
                yolo_model = st.selectbox(
                    "Select Model",
                    ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                     "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
                    index=0,
                    help="Models download automatically from Ultralytics on first use"
                )
            else:
                yolo_model = st.selectbox(
                    "Select Model",
                    ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
                     "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"],
                    index=0,
                    help="YOLO26 Segment models pretrained on COCO dataset | Models auto-download on first use"
                )

                # Show YOLO26 performance table
                if "yolo26" in yolo_model:
                    with st.expander("YOLO26 Segmentation Performance (COCO)"):
                        st.markdown("""
                        | Model | mAP box 50-95 | mAP mask 50-95 | Speed CPU (ms) | Speed T4 TensorRT (ms) | Params (M) | FLOPs (B) |
                        |-------|---------------|----------------|----------------|------------------------|------------|-----------|
                        | YOLO26n-seg | 39.6 | 33.9 | 53.3 | 2.1 | 2.7 | 9.1 |
                        | YOLO26s-seg | 47.3 | 40.0 | 118.4 | 3.3 | 10.4 | 34.2 |
                        | YOLO26m-seg | 52.5 | 44.1 | 328.2 | 6.7 | 23.6 | 121.5 |
                        | YOLO26l-seg | 54.4 | 45.5 | 387.0 | 8.0 | 28.0 | 139.8 |
                        | YOLO26x-seg | 56.5 | 47.0 | 787.0 | 16.4 | 62.8 | 313.5 |

                        *Speed averaged over COCO val images using Amazon EC2 P4d instance*
                        """)
        else:
            # Custom path
            custom_paths = [
                "/Users/hamed/work/Computer-Vision-Engine/models/Final weights.pt",
                "Custom path..."
            ]
            selected_path = st.selectbox("Saved Models", custom_paths)

            if selected_path == "Custom path...":
                yolo_model = st.text_input("Enter model path", value="")
            else:
                yolo_model = selected_path
        detection_conf = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)

        # FPS control for video processing
        processing_fps = st.slider(
            "Video Processing FPS",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            help="Frames per second to process (lower = faster processing, higher = more detections)"
        )

        if model_source == "Pretrained":
            if yolo_type == "Detection":
                st.info("Mode: Detection (bounding boxes) | Class 19 (cow)")
            else:
                st.info("Mode: Segmentation (precise masks) | Class 19 (cow)")
        else:
            if yolo_type == "Detection":
                st.info("Mode: Detection (bounding boxes) | Custom trained model")
            else:
                st.info("Mode: Segmentation (precise masks) | Custom trained model")

        st.markdown("---")

        # Preprocessing
        st.subheader("Preprocessing")
        use_preprocessing = st.checkbox(
            "Apply CLAHE + Blur to reference",
            value=True,
            help="Enhances pattern contrast and reduces noise"
        )

        st.markdown("---")

        # Matching info
        st.subheader("Matching Strategy")
        st.info("TOP-1: Highest scored cow = **ALPHA**")
        st.info("All other cows = None")
        st.info("No threshold needed")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Reference Pattern (ALPHA Cow)")
        ref_file = st.file_uploader(
            "Upload ALPHA cow pattern image",
            type=["jpg", "jpeg", "png"],
            help="Upload to set the reference ALPHA cow"
        )

        if ref_file is not None:
            file_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
            ref_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.image(
                cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB),
                caption="Reference Pattern",
                width="stretch"
            )

            if use_preprocessing:
                processed = preprocess_reference(ref_image)
                st.image(
                    cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                    caption="Preprocessed Pattern",
                    width="stretch"
                )
                st.session_state.reference_image = processed
            else:
                st.session_state.reference_image = ref_image

            # Registration mode selection
            st.markdown("---")
            registration_mode = st.radio(
                "Registration Mode:",
                ["Single View", "Multi-View (3 Images)"],
                help="Multi-view: Upload 3 different views (e.g., left, top, right) of the same cow for better accuracy"
            )

            if registration_mode == "Single View":
                # Single view registration
                if st.button("Set as ALPHA (Reference Cow)"):
                    if 'embedding_model' not in st.session_state:
                        st.session_state.embedding_model = EmbeddingModel(model_name, device, use_patch_matching)

                    emb = st.session_state.embedding_model.get_embedding(st.session_state.reference_image)
                    if emb is not None:
                        is_patch = isinstance(emb, dict)
                        st.session_state.reference_db.set_reference(emb, st.session_state.reference_image, is_patch)
                        st.success("**ALPHA cow registered successfully! (Single View)**")
                    else:
                        st.error("Failed to extract embedding")

            else:
                # Multi-view registration
                st.markdown("**Multi-View Registration**")
                st.info("Upload 3 different views of the same cow. The system will match against ALL views and use the best score.")

                view_mode = st.radio(
                    "How to provide 3 views:",
                    ["Auto-split (divide image into 3 parts)", "Manual upload (3 separate images)"],
                    help="Auto-split: divides current image into left/center/right. Manual: upload 3 separate images"
                )

                if view_mode == "Auto-split (divide image into 3 parts)":
                    # Show preview of split
                    if 'reference_image' in st.session_state:
                        img = st.session_state.reference_image
                        h, w = img.shape[:2]

                        # Split into 3 equal vertical sections
                        split_width = w // 3
                        view1 = img[:, :split_width]  # Left
                        view2 = img[:, split_width:2*split_width]  # Center
                        view3 = img[:, 2*split_width:]  # Right

                        st.markdown("**Preview of 3 views:**")
                        col_v1, col_v2, col_v3 = st.columns(3)
                        with col_v1:
                            st.image(cv2.cvtColor(view1, cv2.COLOR_BGR2RGB), caption="Left", use_container_width=True)
                        with col_v2:
                            st.image(cv2.cvtColor(view2, cv2.COLOR_BGR2RGB), caption="Center", use_container_width=True)
                        with col_v3:
                            st.image(cv2.cvtColor(view3, cv2.COLOR_BGR2RGB), caption="Right", use_container_width=True)

                        if st.button("Register Multi-View (Auto-split)"):
                            if 'embedding_model' not in st.session_state:
                                st.session_state.embedding_model = EmbeddingModel(model_name, device, use_patch_matching)

                            embeddings = []
                            thumbnails = []
                            views = [view1, view2, view3]

                            with st.spinner("Extracting embeddings from 3 views..."):
                                for idx, view_img in enumerate(views):
                                    emb = st.session_state.embedding_model.get_embedding(view_img)
                                    if emb is not None:
                                        embeddings.append(emb)
                                        thumbnails.append(view_img)
                                    else:
                                        st.error(f"Failed to extract embedding from view {idx+1}")

                            if len(embeddings) == 3:
                                is_patch = isinstance(embeddings[0], dict)
                                st.session_state.reference_db.set_multi_view_reference(embeddings, thumbnails, is_patch)
                                st.success("**ALPHA cow registered successfully! (Multi-View: 3 views)**")
                            else:
                                st.error("Failed to process all 3 views")

                else:
                    # Manual upload of 3 separate images
                    st.markdown("**Upload 3 separate images:**")

                    view_file1 = st.file_uploader("View 1 (e.g., Left side)", type=["jpg", "jpeg", "png"], key="view1")
                    view_file2 = st.file_uploader("View 2 (e.g., Top/Front)", type=["jpg", "jpeg", "png"], key="view2")
                    view_file3 = st.file_uploader("View 3 (e.g., Right side)", type=["jpg", "jpeg", "png"], key="view3")

                    if view_file1 and view_file2 and view_file3:
                        # Load images
                        view_images = []
                        for vf in [view_file1, view_file2, view_file3]:
                            file_bytes = np.asarray(bytearray(vf.read()), dtype=np.uint8)
                            view_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            if use_preprocessing:
                                view_img = preprocess_reference(view_img)
                            view_images.append(view_img)

                        # Show previews
                        col_v1, col_v2, col_v3 = st.columns(3)
                        with col_v1:
                            st.image(cv2.cvtColor(view_images[0], cv2.COLOR_BGR2RGB), caption="View 1", use_container_width=True)
                        with col_v2:
                            st.image(cv2.cvtColor(view_images[1], cv2.COLOR_BGR2RGB), caption="View 2", use_container_width=True)
                        with col_v3:
                            st.image(cv2.cvtColor(view_images[2], cv2.COLOR_BGR2RGB), caption="View 3", use_container_width=True)

                        if st.button("Register Multi-View (Manual)"):
                            if 'embedding_model' not in st.session_state:
                                st.session_state.embedding_model = EmbeddingModel(model_name, device, use_patch_matching)

                            embeddings = []
                            with st.spinner("Extracting embeddings from 3 views..."):
                                for idx, view_img in enumerate(view_images):
                                    emb = st.session_state.embedding_model.get_embedding(view_img)
                                    if emb is not None:
                                        embeddings.append(emb)
                                    else:
                                        st.error(f"Failed to extract embedding from view {idx+1}")

                            if len(embeddings) == 3:
                                is_patch = isinstance(embeddings[0], dict)
                                st.session_state.reference_db.set_multi_view_reference(embeddings, view_images, is_patch)
                                st.success("**ALPHA cow registered successfully! (Multi-View: 3 manual images)**")
                            else:
                                st.error("Failed to process all 3 views")

    with col2:
        st.subheader("Upload Image or Video")

        # Tab selection for image vs video
        input_type = st.radio("Choose Input Type:", ["Image", "Video"], horizontal=True)

        if input_type == "Image":
            image_file = st.file_uploader(
                "Upload image for detection",
                type=["jpg", "jpeg", "png"],
                key="image_upload"
            )

            if image_file is not None:
                file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.input_image = input_image
                st.session_state.input_type = "image"
                st.success("Image uploaded successfully")

        else:  # Video
            video_file = st.file_uploader(
                "Upload video for tracking",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_upload"
            )

            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                st.session_state.video_path = tfile.name
                st.session_state.input_type = "video"
                st.success("Video uploaded successfully")
                st.info("Processing at 1 FPS for efficiency")

    st.markdown("---")

    # Show reference status
    reference_db = st.session_state.reference_db
    if reference_db.has_reference():
        if reference_db.use_multi_view:
            st.success(f"**ALPHA cow registered with {len(reference_db.reference_views)} views for matching!**")
            st.info("Multi-view mode: System will compare against ALL views and use the BEST match score")

            # Show all view thumbnails
            cols = st.columns(len(reference_db.view_thumbnails))
            for idx, (col, thumb) in enumerate(zip(cols, reference_db.view_thumbnails)):
                with col:
                    st.image(
                        cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                        caption=f"View {idx+1}",
                        use_container_width=True
                    )

            st.write(f"Registered at: {reference_db.registered_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.success("**ALPHA cow is registered and ready for matching!**")
            if reference_db.reference_thumbnail is not None:
                col_thumb, col_info = st.columns([1, 3])
                with col_thumb:
                    st.image(
                        cv2.cvtColor(reference_db.reference_thumbnail, cv2.COLOR_BGR2RGB),
                        caption="ALPHA",
                        width=100
                    )
                with col_info:
                    st.write(f"Registered at: {reference_db.registered_at.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No ALPHA cow registered. Upload a reference image to start.")

    st.markdown("---")

    # Processing
    if st.button("Start Processing", type="primary", width="stretch"):
        if 'input_type' not in st.session_state:
            st.error("Please upload an image or video first")
            return

        if not reference_db.has_reference():
            st.warning("No ALPHA cow registered. Processing will continue, but TOP-1 will be determined among detected cows only.")

        # Initialize models
        with st.spinner("Loading models..."):
            try:
                yolo = YOLO(yolo_model)
                st.success(f"YOLO loaded: {yolo_model}")
            except Exception as e:
                st.error(f"YOLO loading failed: {e}")
                return

            if 'embedding_model' not in st.session_state:
                st.session_state.embedding_model = EmbeddingModel(model_name, device, use_patch_matching)

            embedding_model = st.session_state.embedding_model
            st.success(f"Embedding model loaded: {model_name}")

            # Show patch matching info for DINOv2
            if "dinov2" in model_name and use_patch_matching:
                st.info("Using DINOv2 patch-level matching (256 patches per image)")

        # Determine if using segmentation
        use_segmentation = (yolo_type == "Segmentation")

        # Determine if using class filter (only for pretrained models)
        use_class_filter = (model_source == "Pretrained")

        # Process based on input type
        if st.session_state.input_type == "image":
            # Process single image
            st.markdown("### Processing Image")

            frame = st.session_state.input_image
            annotated_frame, cow_data = process_frame(frame, yolo, embedding_model, reference_db, detection_conf, use_segmentation, use_class_filter)

            # Display annotated image
            col_main, col_detections = st.columns([2, 1])

            with col_main:
                st.subheader("Annotated Image")
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(display_frame, use_container_width=True)

                # Stats
                st.markdown("#### Statistics")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Total Detections", len(cow_data))
                with stats_col2:
                    alpha_count = sum(1 for cow in cow_data if cow['id'] == 'ALPHA')
                    st.metric("ALPHA Detections", alpha_count)

            with col_detections:
                st.subheader("Detected Cows")

                if len(cow_data) == 0:
                    st.info("No cows detected")
                else:
                    # Create scrollable container with fixed height
                    with st.container(height=600):
                        for idx, cow in enumerate(cow_data):
                            st.markdown(f"**Cow #{idx + 1}**")

                            # Display crop
                            crop_rgb = cv2.cvtColor(cow['crop'], cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, use_container_width=True)

                            # Display info
                            if cow['id'] == 'ALPHA':
                                st.markdown(f"**ID:** :green[**{cow['id']}**]")
                            else:
                                st.markdown(f"**ID:** {cow['id']}")

                            st.markdown(f"**Detection Conf:** {cow['confidence']:.3f}")
                            st.markdown(f"**Similarity Score:** {cow['score']:.3f}")

                            # Show patch info for DINOv2
                            if use_patch_matching and isinstance(cow.get('embedding'), dict):
                                emb_dict = cow['embedding']
                                num_patches = emb_dict['patches'].shape[0]
                                st.caption(f"Patches: {num_patches} (Grid: {emb_dict['grid_shape'][0]}x{emb_dict['grid_shape'][1]})")

                            st.markdown(f"**Bbox:** {cow['bbox']}")
                            st.markdown("---")

        else:  # Video
            # Process video
            st.markdown(f"### Processing Video ({processing_fps} FPS)")

            video_cap = cv2.VideoCapture(st.session_state.video_path)
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))

            # Calculate frame interval based on desired processing FPS
            frame_interval = max(1, int(fps / processing_fps))

            progress_bar = st.progress(0)

            # Create layout: main video + detections sidebar
            col_main, col_detections = st.columns([2, 1])

            with col_main:
                frame_placeholder = st.empty()
                stats_placeholder = st.empty()

            with col_detections:
                st.subheader("Detected Cows (Current Frame)")
                # Create a container with fixed height for scrolling
                detections_placeholder = st.container(height=600)

            frame_count = 0
            processed_count = 0
            total_detections = 0
            alpha_detections = 0

            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process only at 1 FPS interval
                if frame_count % frame_interval != 0:
                    continue

                processed_count += 1

                # Process frame
                annotated_frame, cow_data = process_frame(frame, yolo, embedding_model, reference_db, detection_conf, use_segmentation, use_class_filter)

                total_detections += len(cow_data)
                alpha_detections += sum(1 for cow in cow_data if cow['id'] == 'ALPHA')

                # Update main display
                with col_main:
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        display_frame,
                        caption=f"Frame {frame_count}/{total_frames} (Processed: {processed_count})",
                        use_container_width=True
                    )

                    # Stats
                    with stats_placeholder.container():
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Processed Frames", processed_count)
                        with stat_col2:
                            st.metric("Total Detections", total_detections)
                        with stat_col3:
                            st.metric("ALPHA Detections", alpha_detections)

                # Update detections display
                with detections_placeholder:
                    # Clear and rebuild the display for each frame
                    if len(cow_data) == 0:
                        st.info("No cows in this frame")
                    else:
                        for idx, cow in enumerate(cow_data):
                            st.markdown(f"**Cow #{idx + 1}**")

                            # Display crop (smaller for video)
                            crop_rgb = cv2.cvtColor(cow['crop'], cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, width=150)

                            # Display compact info
                            if cow['id'] == 'ALPHA':
                                st.markdown(f"**ID:** :green[**{cow['id']}**]")
                            else:
                                st.markdown(f"**ID:** {cow['id']}")

                            st.caption(f"Det: {cow['confidence']:.2f} | Sim: {cow['score']:.2f}")

                            # Show patch info for DINOv2 in video mode
                            if use_patch_matching and isinstance(cow.get('embedding'), dict):
                                emb_dict = cow['embedding']
                                st.caption(f"Patches: {emb_dict['patches'].shape[0]}")

                            st.markdown("---")

                progress_bar.progress(frame_count / total_frames)

            video_cap.release()

            st.success("Processing complete!")

            # Display final stats
            st.markdown("### Final Statistics")
            final_col1, final_col2, final_col3, final_col4 = st.columns(4)

            with final_col1:
                st.info(f"**Total Frames:** {frame_count}")
            with final_col2:
                st.info(f"**Processed Frames:** {processed_count}")
            with final_col3:
                st.info(f"**Total Detections:** {total_detections}")
            with final_col4:
                st.info(f"**ALPHA Detections:** {alpha_detections}")


if __name__ == "__main__":
    main()
