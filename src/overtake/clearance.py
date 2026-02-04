"""Clearance zone geometry calculation for Overtake Assistant."""

from typing import List, Tuple, Optional
from src.lane.result import LanePolynomial


def calculate_clearance_zone(
    left_lane: LanePolynomial,
    right_lane: LanePolynomial,
    frame_width: int,
    frame_height: int,
    zone_y_top_ratio: float = 0.65,
    zone_width_ratio: float = 1.0,
    overtake_side: str = "right",
) -> List[Tuple[int, int]]:
    """
    Calculate the clearance zone polygon for overtake evaluation.
    
    For left-hand traffic (drive on left, overtake on right):
        The clearance zone is to the RIGHT of the right lane.
    For right-hand traffic (drive on right, overtake on left):
        The clearance zone is to the LEFT of the left lane.
    
    The zone width matches our lane width multiplied by zone_width_ratio.
    
    Args:
        left_lane: Left lane polynomial
        right_lane: Right lane polynomial
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        zone_y_top_ratio: Top of zone as ratio of frame height
        zone_width_ratio: Zone width as ratio of detected lane width
        overtake_side: "right" for left-hand traffic, "left" for right-hand traffic
        
    Returns:
        List of 4 (x, y) tuples forming a quadrilateral polygon:
        [top_left, top_right, bottom_right, bottom_left]
    """
    # Define y positions for the zone
    y_top = int(frame_height * zone_y_top_ratio)
    y_bottom = frame_height - 1
    
    # Get lane x positions at top and bottom
    left_x_top = left_lane.evaluate(y_top)
    left_x_bottom = left_lane.evaluate(y_bottom)
    right_x_top = right_lane.evaluate(y_top)
    right_x_bottom = right_lane.evaluate(y_bottom)
    
    # Calculate lane width at each y position
    lane_width_top = right_x_top - left_x_top
    lane_width_bottom = right_x_bottom - left_x_bottom
    
    zone_width_top = lane_width_top * zone_width_ratio
    zone_width_bottom = lane_width_bottom * zone_width_ratio
    
    if overtake_side == "right":
        # Left-hand traffic: clearance zone is to the RIGHT of right lane
        zone_left_top = int(right_x_top)
        zone_left_bottom = int(right_x_bottom)
        zone_right_top = min(frame_width - 1, int(right_x_top + zone_width_top))
        zone_right_bottom = min(frame_width - 1, int(right_x_bottom + zone_width_bottom))
    else:
        # Right-hand traffic: clearance zone is to the LEFT of left lane
        zone_left_top = max(0, int(left_x_top - zone_width_top))
        zone_left_bottom = max(0, int(left_x_bottom - zone_width_bottom))
        zone_right_top = int(left_x_top)
        zone_right_bottom = int(left_x_bottom)
    
    return [
        (zone_left_top, y_top),           # Top left
        (zone_right_top, y_top),          # Top right
        (zone_right_bottom, y_bottom),    # Bottom right
        (zone_left_bottom, y_bottom),     # Bottom left
    ]


def is_zone_valid(
    zone: List[Tuple[int, int]],
    frame_width: int,
    min_width_px: int = 30,
) -> bool:
    """
    Validate that the clearance zone is geometrically reasonable.
    
    Args:
        zone: Polygon coordinates
        frame_width: Frame width in pixels
        min_width_px: Minimum zone width to be considered valid
        
    Returns:
        True if zone is valid for evaluation
    """
    if len(zone) != 4:
        return False
    
    top_left, top_right, bottom_right, bottom_left = zone
    
    # Check minimum widths
    top_width = top_right[0] - top_left[0]
    bottom_width = bottom_right[0] - bottom_left[0]
    
    if top_width < min_width_px or bottom_width < min_width_px:
        return False
    
    # Check that zone is within frame (at least partially)
    if top_right[0] < 0 or bottom_right[0] < 0:
        return False
    
    # Check proper ordering (left should be to the left of right)
    if top_left[0] >= top_right[0] or bottom_left[0] >= bottom_right[0]:
        return False
    
    return True


def point_in_clearance_zone(
    x: float,
    y: float,
    zone: List[Tuple[int, int]],
) -> bool:
    """
    Check if a point is inside the clearance zone polygon.
    
    Uses ray casting algorithm.
    
    Args:
        x: X coordinate of point
        y: Y coordinate of point
        zone: Polygon coordinates
        
    Returns:
        True if point is inside the polygon
    """
    n = len(zone)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = zone[i]
        xj, yj = zone[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def bbox_intersects_zone(
    bbox: Tuple[float, float, float, float],
    zone: List[Tuple[int, int]],
) -> bool:
    """
    Check if a bounding box intersects the clearance zone.
    
    Uses center point for simplicity (conservative approach).
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        zone: Polygon coordinates
        
    Returns:
        True if the bbox center is inside the zone
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return point_in_clearance_zone(center_x, center_y, zone)
