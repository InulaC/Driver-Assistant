"""Post-processing for YOLO detections: NMS and class mapping."""

from typing import List, Tuple
import numpy as np

from .result import Detection, DetectionLabel


# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# COCO class index to DetectionLabel mapping
# Only map classes we care about for driver safety
COCO_CLASS_MAPPING: dict[int, DetectionLabel] = {
    0: DetectionLabel.PEDESTRIAN,     # person
    1: DetectionLabel.VEHICLE,        # bicycle
    2: DetectionLabel.VEHICLE,        # car
    3: DetectionLabel.VEHICLE,        # motorcycle
    5: DetectionLabel.VEHICLE,        # bus
    6: DetectionLabel.VEHICLE,        # train
    7: DetectionLabel.VEHICLE,        # truck
    9: DetectionLabel.TRAFFIC_LIGHT,  # traffic light
    11: DetectionLabel.STOP_SIGN,     # stop sign
    15: DetectionLabel.ANIMAL,        # cat
    16: DetectionLabel.ANIMAL,        # dog
    17: DetectionLabel.ANIMAL,        # horse
    19: DetectionLabel.ANIMAL,        # cow
}


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        boxes: Array of bounding boxes [N, 4] in xyxy format
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by confidence (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # Keep the highest confidence box
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Compare with remaining boxes
        remaining = sorted_indices[1:]
        ious = np.array([compute_iou(boxes[current], boxes[i]) for i in remaining])
        
        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        sorted_indices = remaining[mask]
    
    return keep


def process_yolo_output(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    filter_relevant_only: bool = True
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Process raw YOLO output and apply NMS.
    
    Args:
        output: YOLO output tensor of shape (1, 84, 8400)
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        filter_relevant_only: If True, only return classes we care about
        
    Returns:
        List of (class_id, confidence, bbox) tuples where bbox is [x1, y1, x2, y2]
    """
    # Transpose from (1, 84, 8400) to (8400, 84)
    output = output[0].T  # Shape: (8400, 84)
    
    # Parse output: first 4 values are box coords (center_x, center_y, width, height)
    # Remaining 80 values are class probabilities
    boxes_xywh = output[:, :4]  # (8400, 4)
    class_probs = output[:, 4:]  # (8400, 80)
    
    # Get best class for each detection
    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)
    
    # Filter by confidence
    conf_mask = confidences >= conf_threshold
    
    # Optionally filter to only relevant classes
    if filter_relevant_only:
        relevant_mask = np.array([cid in COCO_CLASS_MAPPING for cid in class_ids])
        mask = conf_mask & relevant_mask
    else:
        mask = conf_mask
    
    filtered_boxes_xywh = boxes_xywh[mask]
    filtered_class_ids = class_ids[mask]
    filtered_confidences = confidences[mask]
    
    if len(filtered_boxes_xywh) == 0:
        return []
    
    # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(filtered_boxes_xywh)
    boxes_xyxy[:, 0] = filtered_boxes_xywh[:, 0] - filtered_boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = filtered_boxes_xywh[:, 1] - filtered_boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = filtered_boxes_xywh[:, 0] + filtered_boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = filtered_boxes_xywh[:, 1] + filtered_boxes_xywh[:, 3] / 2  # y2
    
    # Apply NMS per class
    results = []
    unique_classes = np.unique(filtered_class_ids)
    
    for cls_id in unique_classes:
        cls_mask = filtered_class_ids == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = filtered_confidences[cls_mask]
        cls_indices_local = np.where(cls_mask)[0]
        
        keep_local = nms(cls_boxes, cls_scores, iou_threshold)
        
        for idx in keep_local:
            global_idx = cls_indices_local[idx]
            results.append((
                int(filtered_class_ids[global_idx]),
                float(filtered_confidences[global_idx]),
                boxes_xyxy[global_idx].copy()
            ))
    
    return results


def create_detections(
    raw_detections: List[Tuple[int, float, np.ndarray]],
    timestamp: float
) -> List[Detection]:
    """
    Convert raw detection tuples to Detection objects.
    
    Args:
        raw_detections: List of (class_id, confidence, bbox) tuples
        timestamp: Current frame timestamp
        
    Returns:
        List of Detection objects
    """
    detections = []
    
    for class_id, confidence, bbox in raw_detections:
        # Get label from mapping (skip if not relevant)
        label = COCO_CLASS_MAPPING.get(class_id)
        if label is None:
            continue
        
        # Get original COCO class name for debugging
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"
        
        detection = Detection(
            label=label,
            confidence=confidence,
            bbox=tuple(bbox.tolist()),  # Convert to tuple (x1, y1, x2, y2)
            class_name=class_name,
            timestamp=timestamp
        )
        detections.append(detection)
    
    return detections


def get_class_name(class_id: int) -> str:
    """Get COCO class name from class ID."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return "unknown"
