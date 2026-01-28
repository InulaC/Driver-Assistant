"""Alert types and event definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class AlertType(Enum):
    """Alert types ordered by priority (lower number = higher priority)."""
    COLLISION_IMMINENT = "collision_imminent"      # Priority 1 - Highest
    LANE_DEPARTURE_LEFT = "lane_departure_left"    # Priority 2
    LANE_DEPARTURE_RIGHT = "lane_departure_right"  # Priority 2
    TRAFFIC_LIGHT_DETECTED = "traffic_light_detected"  # Priority 3 - Generic
    STOP_SIGN = "stop_sign"                        # Priority 3
    ANIMAL_WARNING = "animal_warning"              # Priority 2
    SYSTEM_WARNING = "system_warning"              # Priority 4 - Lowest
    
    @property
    def priority(self) -> int:
        """Get priority level (1=highest, 4=lowest)."""
        priority_map = {
            AlertType.COLLISION_IMMINENT: 1,
            AlertType.LANE_DEPARTURE_LEFT: 2,
            AlertType.LANE_DEPARTURE_RIGHT: 2,
            AlertType.ANIMAL_WARNING: 2,
            AlertType.TRAFFIC_LIGHT_DETECTED: 3,
            AlertType.STOP_SIGN: 3,
            AlertType.SYSTEM_WARNING: 4,
        }
        return priority_map.get(self, 4)
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            AlertType.COLLISION_IMMINENT: "COLLISION WARNING",
            AlertType.LANE_DEPARTURE_LEFT: "LANE DEPARTURE LEFT",
            AlertType.LANE_DEPARTURE_RIGHT: "LANE DEPARTURE RIGHT",
            AlertType.TRAFFIC_LIGHT_DETECTED: "TRAFFIC LIGHT DETECTED",
            AlertType.STOP_SIGN: "STOP SIGN",
            AlertType.ANIMAL_WARNING: "ANIMAL WARNING",
            AlertType.SYSTEM_WARNING: "SYSTEM WARNING",
        }
        return names.get(self, self.value)


@dataclass
class AlertEvent:
    """Represents an alert event."""
    alert_type: AlertType
    timestamp: float
    confidence: float = 1.0
    trigger_source: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def priority(self) -> int:
        """Get alert priority."""
        return self.alert_type.priority
    
    def __lt__(self, other: "AlertEvent") -> bool:
        """Compare by priority for sorting (lower priority number = higher urgency)."""
        return self.priority < other.priority
