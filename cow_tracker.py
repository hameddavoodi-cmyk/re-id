"""
Cow tracker with ID persistence.
Combines YOLO detection with pattern matching for reliable tracking.
"""

import cv2
import numpy as np
from collections import defaultdict


class CowTracker:
    """
    Tracks cows across video frames using YOLO detection and pattern matching.
    Maintains persistent IDs even when cows leave and re-enter the frame.
    """

    def __init__(self, matcher, max_disappeared=30):
        """
        Initialize the cow tracker.

        Args:
            matcher: Pattern matcher (LocalFeatureMatcher or DeepFeatureMatcher)
            max_disappeared (int): Max frames a cow can be missing before removing
        """
        self.matcher = matcher
        self.max_disappeared = max_disappeared

        # Track registered cows
        self.next_id = 1
        self.tracked_cows = {}  # {cow_id: {'bbox': (x,y,w,h), 'disappeared': int, 'confidence': float}}
        self.match_history = defaultdict(list)  # Track match confidence over time

    def reset(self):
        """Reset all tracking information."""
        self.next_id = 1
        self.tracked_cows = {}
        self.match_history = defaultdict(list)

    def update(self, detections, frame):
        """
        Update tracker with new detections from current frame.
        Uses TOP-1 matching: only tracks the detection with highest similarity.

        Args:
            detections (list): List of detection dicts with 'bbox' and 'confidence'
            frame (np.ndarray): Current video frame (BGR)

        Returns:
            dict: Updated tracked cows with IDs
        """
        # Mark all existing cows as potentially disappeared
        for cow_id in list(self.tracked_cows.keys()):
            self.tracked_cows[cow_id]['disappeared'] += 1

        # Store detection results for debugging
        self.last_detections = []
        detection_scores = []  # Store all detections with their similarity scores

        # Process each detection and compute similarity scores
        for detection in detections:
            bbox = detection['bbox']  # (x, y, w, h)
            x, y, w, h = bbox

            # Crop detected cow region
            cow_crop = frame[y:y+h, x:x+w]

            if cow_crop.size == 0:
                continue

            # Try to match with reference pattern
            is_match, match_confidence, *extra = self._match_detection(cow_crop)

            # Store detection with its similarity score
            detection_scores.append({
                'bbox': bbox,
                'confidence': match_confidence,
                'crop': cow_crop
            })

        # TOP-1 MATCHING: Find the detection with highest similarity
        if detection_scores:
            # Sort by confidence/similarity (descending)
            detection_scores.sort(key=lambda x: x['confidence'], reverse=True)

            # Get the top-1 (highest similarity)
            top1_detection = detection_scores[0]

            # Mark top-1 as matched, rest as rejected
            for i, det in enumerate(detection_scores):
                is_top1 = (i == 0)
                self.last_detections.append({
                    'bbox': det['bbox'],
                    'is_match': is_top1,
                    'confidence': det['confidence']
                })

                if is_top1:
                    # Track only the top-1 detection
                    bbox = det['bbox']
                    match_confidence = det['confidence']

                    # Check if this matches an existing tracked cow
                    cow_id = self._find_matching_cow(bbox, match_confidence)

                    if cow_id is None:
                        # New cow detected
                        cow_id = self.next_id
                        self.next_id += 1

                    # Update tracking info
                    self.tracked_cows[cow_id] = {
                        'bbox': bbox,
                        'disappeared': 0,
                        'confidence': match_confidence,
                        'last_seen_frame': frame.copy()
                    }

                    self.match_history[cow_id].append(match_confidence)

        # Remove cows that have disappeared for too long
        ids_to_remove = [
            cow_id for cow_id, info in self.tracked_cows.items()
            if info['disappeared'] > self.max_disappeared
        ]

        for cow_id in ids_to_remove:
            del self.tracked_cows[cow_id]

        return self.tracked_cows

    def _match_detection(self, cow_crop):
        """
        Match detected cow crop with reference pattern.

        Args:
            cow_crop (np.ndarray): Cropped image of detected cow

        Returns:
            tuple: Matching result from matcher
        """
        return self.matcher.match_pattern(cow_crop)

    def _find_matching_cow(self, bbox, confidence, iou_threshold=0.3):
        """
        Find if detection matches an existing tracked cow.

        Args:
            bbox (tuple): Detection bounding box (x, y, w, h)
            confidence (float): Match confidence
            iou_threshold (float): IOU threshold for spatial matching

        Returns:
            int or None: Cow ID if match found, None otherwise
        """
        best_match_id = None
        best_iou = 0

        for cow_id, info in self.tracked_cows.items():
            if info['disappeared'] > 0:  # Only consider recently seen cows
                continue

            # Calculate IOU (Intersection over Union)
            iou = self._calculate_iou(bbox, info['bbox'])

            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match_id = cow_id

        return best_match_id

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union of two bounding boxes.

        Args:
            bbox1 (tuple): First bbox (x, y, w, h)
            bbox2 (tuple): Second bbox (x, y, w, h)

        Returns:
            float: IOU score (0-1)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def draw_tracks(self, frame, show_rejected=True):
        """
        Draw tracking information on frame.

        Args:
            frame (np.ndarray): Frame to draw on (will be modified)
            show_rejected (bool): Whether to show rejected detections in red

        Returns:
            np.ndarray: Frame with tracking visualization
        """
        output = frame.copy()

        # Draw rejected detections (YOLO detected but pattern didn't match)
        if show_rejected and hasattr(self, 'last_detections'):
            for det in self.last_detections:
                if not det['is_match']:
                    bbox = det['bbox']
                    x, y, w, h = bbox

                    # Draw red box for rejected detections
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Label
                    label = f"NO MATCH ({det['confidence']:.2%})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Background
                    cv2.rectangle(
                        output,
                        (x, y - label_size[1] - 10),
                        (x + label_size[0], y),
                        (0, 0, 255),
                        -1
                    )

                    # Text
                    cv2.putText(
                        output,
                        label,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

        # Draw tracked cows (pattern matched)
        for cow_id, info in self.tracked_cows.items():
            if info['disappeared'] > 0:
                continue

            bbox = info['bbox']
            confidence = info['confidence']

            x, y, w, h = bbox

            # Draw bounding box (green for tracked cow with matching pattern)
            color = (0, 255, 0)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)

            # Draw ID and confidence
            label = f"TARGET COW ID: {cow_id} ({confidence:.2%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Background for text
            cv2.rectangle(
                output,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )

            # Text
            cv2.putText(
                output,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return output

    def get_tracked_cow_count(self):
        """
        Get number of currently tracked cows.

        Returns:
            int: Number of tracked cows
        """
        return len([info for info in self.tracked_cows.values() if info['disappeared'] == 0])

    def get_detection_stats(self):
        """
        Get statistics about the last frame's detections.

        Returns:
            dict: Statistics including total detections, matches, rejections
        """
        if not hasattr(self, 'last_detections'):
            return {
                'total_detections': 0,
                'matched': 0,
                'rejected': 0,
                'match_rate': 0.0
            }

        total = len(self.last_detections)
        matched = sum(1 for d in self.last_detections if d['is_match'])
        rejected = total - matched

        return {
            'total_detections': total,
            'matched': matched,
            'rejected': rejected,
            'match_rate': matched / total if total > 0 else 0.0
        }
