"""Alert decision engine - evaluates hazards and generates alerts."""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from .types import AlertType, AlertEvent
from src.detection.result import Detection, DetectionLabel


@dataclass
class DangerZone:
    """
    Trapezoidal danger zone for collision detection.
    
    The trapezoid represents the forward driving corridor:
    - When lanes are detected: aligns with lane boundaries
    - When lanes not detected: uses default fixed trapezoid
    - Narrower at top (farther objects), wider at bottom (closer objects)
    """
    # Default coordinates as ratios of frame dimensions (used when no lanes detected)
    # Top edge (narrow, representing farther distance)
    top_left_x: float = 0.42
    top_left_y: float = 0.65      # Start 65% down
    top_right_x: float = 0.58
    top_right_y: float = 0.65
    # Bottom edge (wider, representing closer distance)
    bottom_left_x: float = 0.2
    bottom_left_y: float = 1.0    # Stick to bottom
    bottom_right_x: float = 0.8
    bottom_right_y: float = 1.0
    
    # Dynamic zone (pixel coordinates, updated from lane detection)
    _dynamic_polygon: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    _use_dynamic: bool = field(default=False, repr=False)
    
    def update_from_lanes(
        self,
        left_lane,  # LanePolynomial or None
        right_lane,  # LanePolynomial or None
        frame_width: int,
        frame_height: int,
        margin: float = 0.05,  # Inward margin as ratio of lane width
    ) -> None:
        """
        Update danger zone to align with detected lanes.
        
        Args:
            left_lane: Left lane polynomial or None
            right_lane: Right lane polynomial or None
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            margin: Inward margin from lane lines (ratio of lane width)
        """
        if left_lane is None or right_lane is None:
            # Can't create dynamic zone without both lanes
            self._use_dynamic = False
            self._dynamic_polygon = None
            return
        
        try:
            # Define y positions for the trapezoid
            y_top = int(frame_height * self.top_left_y)  # Use same y ratio as default
            y_bottom = frame_height - 1  # Stick to bottom
            
            # Get lane x positions at top and bottom
            left_x_top = left_lane.evaluate(y_top)
            right_x_top = right_lane.evaluate(y_top)
            left_x_bottom = left_lane.evaluate(y_bottom)
            right_x_bottom = right_lane.evaluate(y_bottom)
            
            # Calculate lane widths and apply inward margin
            top_width = right_x_top - left_x_top
            bottom_width = right_x_bottom - left_x_bottom
            
            top_margin = top_width * margin
            bottom_margin = bottom_width * margin
            
            # Create polygon with margin
            self._dynamic_polygon = [
                (int(left_x_top + top_margin), y_top),        # Top left
                (int(right_x_top - top_margin), y_top),       # Top right
                (int(right_x_bottom - bottom_margin), y_bottom),  # Bottom right
                (int(left_x_bottom + bottom_margin), y_bottom),   # Bottom left
            ]
            
            # Validate polygon (ensure it's reasonable)
            if (top_width > 20 and bottom_width > 50 and 
                left_x_top < right_x_top and left_x_bottom < right_x_bottom):
                self._use_dynamic = True
            else:
                self._use_dynamic = False
                self._dynamic_polygon = None
                
        except Exception:
            self._use_dynamic = False
            self._dynamic_polygon = None
    
    def get_polygon(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Get polygon coordinates for given frame dimensions."""
        if self._use_dynamic and self._dynamic_polygon is not None:
            return self._dynamic_polygon
        
        # Return default fixed trapezoid
        return [
            (int(self.top_left_x * width), int(self.top_left_y * height)),
            (int(self.top_right_x * width), int(self.top_right_y * height)),
            (int(self.bottom_right_x * width), int(self.bottom_right_y * height)),
            (int(self.bottom_left_x * width), int(self.bottom_left_y * height)),
        ]
    
    def contains_point(self, x: float, y: float, width: int, height: int) -> bool:
        """Check if a point is inside the danger zone."""
        if self._use_dynamic and self._dynamic_polygon is not None:
            return self._point_in_polygon(x, y, self._dynamic_polygon)
        
        # Use default trapezoid logic
        nx = x / width
        ny = y / height
        
        if ny < self.top_left_y:
            return False
        
        # Interpolate the horizontal bounds based on y position
        t = (ny - self.top_left_y) / (self.bottom_left_y - self.top_left_y)
        left_x = self.top_left_x + t * (self.bottom_left_x - self.top_left_x)
        right_x = self.top_right_x + t * (self.bottom_right_x - self.top_right_x)
        
        return left_x <= nx <= right_x
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[int, int]]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def intersects_bbox(self, bbox: tuple, width: int, height: int) -> bool:
        """Check if a bounding box intersects the danger zone."""
        x1, y1, x2, y2 = bbox
        
        # Check center point
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if self.contains_point(cx, cy, width, height):
            return True
        
        # Check bottom center (most relevant for collision)
        if self.contains_point(cx, y2, width, height):
            return True
        
        # Check corners
        for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            if self.contains_point(x, y, width, height):
                return True
        
        return False
    
    @property
    def is_dynamic(self) -> bool:
        """Check if currently using dynamic lane-based zone."""
        return self._use_dynamic


class AlertDecisionEngine:
    """
    Evaluates hazards and determines which alert to dispatch.
    
    Features:
    - Priority-based alert selection (only highest priority fires)
    - Cooldown to prevent alert spam
    - Confidence thresholds
    """
    
    def __init__(
        self,
        cooldown_ms: float = 500.0,
        traffic_light_cooldown_ms: float = 5000.0,  # Separate cooldown for traffic lights
        confidence_threshold: float = 0.4,
        frame_width: int = 640,
        frame_height: int = 480,
        danger_zone_config = None,  # DangerZoneConfig from config.yaml
    ):
        self.cooldown_ms = cooldown_ms
        self.traffic_light_cooldown_ms = traffic_light_cooldown_ms
        self.confidence_threshold = confidence_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Create danger zone with config values if provided
        if danger_zone_config is not None:
            self.danger_zone = DangerZone(
                top_left_x=danger_zone_config.top_left_x,
                top_left_y=danger_zone_config.top_left_y,
                top_right_x=danger_zone_config.top_right_x,
                top_right_y=danger_zone_config.top_right_y,
                bottom_left_x=danger_zone_config.bottom_left_x,
                bottom_left_y=danger_zone_config.bottom_left_y,
                bottom_right_x=danger_zone_config.bottom_right_x,
                bottom_right_y=danger_zone_config.bottom_right_y,
            )
        else:
            self.danger_zone = DangerZone()
        
        # Track last alert time per type for cooldown
        self._last_alert_time: dict[AlertType, float] = {}
        self._current_alert: Optional[AlertEvent] = None
    
    def evaluate(
        self,
        detections: List[Detection],
        lane_departure: Optional[str] = None,  # "left", "right", or None
        lane_result = None,  # LaneResult for dynamic danger zone
    ) -> Optional[AlertEvent]:
        """
        Evaluate all inputs and return highest priority alert if any.
        
        Args:
            detections: List of Detection objects from YOLO
            lane_departure: Lane departure direction or None
            lane_result: LaneResult object for dynamic danger zone (optional)
            
        Returns:
            AlertEvent if alert should fire, None otherwise
        """
        timestamp = time.monotonic()
        candidates: List[AlertEvent] = []
        
        # Update danger zone from lane detection (if both lanes detected)
        if lane_result is not None:
            self.danger_zone.update_from_lanes(
                left_lane=lane_result.left_lane,
                right_lane=lane_result.right_lane,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )
        
        # Check for collision risks (Priority 1)
        collision_detections = self._check_collision_risks(detections)
        if collision_detections:
            candidates.append(AlertEvent(
                alert_type=AlertType.COLLISION_IMMINENT,
                timestamp=timestamp,
                confidence=max(d.confidence for d in collision_detections),
                trigger_source="danger_zone",
                metadata={"count": len(collision_detections)},
            ))
        
        # Check for animal warnings (Priority 2)
        animals = [d for d in detections if d.label == DetectionLabel.ANIMAL 
                   and d.confidence >= self.confidence_threshold]
        if animals:
            candidates.append(AlertEvent(
                alert_type=AlertType.ANIMAL_WARNING,
                timestamp=timestamp,
                confidence=max(a.confidence for a in animals),
                trigger_source="detection",
            ))
        
        # Check lane departure (Priority 2)
        if lane_departure == "left":
            candidates.append(AlertEvent(
                alert_type=AlertType.LANE_DEPARTURE_LEFT,
                timestamp=timestamp,
                trigger_source="lane_detection",
            ))
        elif lane_departure == "right":
            candidates.append(AlertEvent(
                alert_type=AlertType.LANE_DEPARTURE_RIGHT,
                timestamp=timestamp,
                trigger_source="lane_detection",
            ))
        
        # Check traffic signals (Priority 3) - single generic alert
        traffic_lights = [d for d in detections if d.label == DetectionLabel.TRAFFIC_LIGHT
                         and d.confidence >= self.confidence_threshold]
        if traffic_lights:
            # Check traffic light specific cooldown before adding candidate
            if self._check_traffic_light_cooldown(timestamp):
                candidates.append(AlertEvent(
                    alert_type=AlertType.TRAFFIC_LIGHT_DETECTED,
                    timestamp=timestamp,
                    confidence=max(t.confidence for t in traffic_lights),
                    trigger_source="detection",
                ))
        
        # Check stop signs (Priority 3)
        stop_signs = [d for d in detections if d.label == DetectionLabel.STOP_SIGN
                      and d.confidence >= self.confidence_threshold]
        if stop_signs:
            candidates.append(AlertEvent(
                alert_type=AlertType.STOP_SIGN,
                timestamp=timestamp,
                confidence=max(s.confidence for s in stop_signs),
                trigger_source="detection",
            ))
        
        if not candidates:
            self._current_alert = None
            return None
        
        # Sort by priority (lowest number = highest priority)
        candidates.sort()
        best_candidate = candidates[0]
        
        # Check cooldown
        if not self._check_cooldown(best_candidate.alert_type, timestamp):
            return None
        
        # Update state
        self._last_alert_time[best_candidate.alert_type] = timestamp
        self._current_alert = best_candidate
        
        return best_candidate
    
    def _check_collision_risks(self, detections: List[Detection]) -> List[Detection]:
        """Check which detections are in the danger zone."""
        collision_risks = []
        
        for det in detections:
            # Only check pedestrians, vehicles, and animals
            if det.label not in (DetectionLabel.PEDESTRIAN, DetectionLabel.VEHICLE, DetectionLabel.ANIMAL):
                continue
            
            if det.confidence < self.confidence_threshold:
                continue
            
            if self.danger_zone.intersects_bbox(
                det.bbox, 
                self.frame_width, 
                self.frame_height
            ):
                collision_risks.append(det)
        
        return collision_risks
    
    def _check_cooldown(self, alert_type: AlertType, current_time: float) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_time = self._last_alert_time.get(alert_type)
        if last_time is None:
            return True
        
        elapsed_ms = (current_time - last_time) * 1000
        
        # Traffic light has its own cooldown
        if alert_type == AlertType.TRAFFIC_LIGHT_DETECTED:
            return elapsed_ms >= self.traffic_light_cooldown_ms
        
        return elapsed_ms >= self.cooldown_ms
    
    def _check_traffic_light_cooldown(self, current_time: float) -> bool:
        """Check if traffic light cooldown has elapsed (separate from other alerts)."""
        last_time = self._last_alert_time.get(AlertType.TRAFFIC_LIGHT_DETECTED)
        if last_time is None:
            return True
        
        elapsed_ms = (current_time - last_time) * 1000
        return elapsed_ms >= self.traffic_light_cooldown_ms
    
    def get_danger_zone_polygon(self) -> List[tuple]:
        """Get danger zone polygon for visualization."""
        return self.danger_zone.get_polygon(self.frame_width, self.frame_height)
    
    def reset(self) -> None:
        """Reset alert state."""
        self._last_alert_time.clear()
        self._current_alert = None
