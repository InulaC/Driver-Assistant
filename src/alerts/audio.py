"""Audio alert manager using Python's winsound (Windows) or beep fallback."""

import platform
import threading
import time
from typing import Optional
from .types import AlertType, AlertEvent


class AudioAlertManager:
    """
    Manages audio alerts using Python built-in sound generation.
    
    On Windows: Uses winsound.Beep() for tone generation
    On Linux/Pi: Uses system beep or prints to console
    
    Features:
    - Non-blocking playback in separate thread
    - Priority preemption (higher priority interrupts lower)
    - Cooldown between same alert type
    """
    
    # Frequency and duration patterns for each alert type
    # Format: [(frequency_hz, duration_ms), ...]
    ALERT_PATTERNS = {
        AlertType.COLLISION_IMMINENT: [
            (2000, 150), (0, 50), (2000, 150), (0, 50), (2000, 150), (0, 50), (2000, 300)
        ],  # Urgent rapid beeps
        AlertType.LANE_DEPARTURE_LEFT: [
            (1000, 200), (0, 100), (1000, 200), (0, 100), (1000, 200)
        ],  # Three medium beeps
        AlertType.LANE_DEPARTURE_RIGHT: [
            (1000, 200), (0, 100), (1000, 200), (0, 100), (1000, 200)
        ],  # Three medium beeps
        AlertType.TRAFFIC_LIGHT_DETECTED: [
            (800, 500), (0, 200), (800, 500)
        ],  # Two long low beeps
        AlertType.STOP_SIGN: [
            (600, 300), (0, 100), (600, 300)
        ],  # Two low beeps
        AlertType.ANIMAL_WARNING: [
            (1500, 200), (1200, 200), (1500, 200)
        ],  # Alternating beeps
        AlertType.SYSTEM_WARNING: [
            (500, 1000)
        ],  # Single long low beep
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._is_windows = platform.system() == "Windows"
        self._playing = False
        self._current_priority = 999
        self._stop_flag = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        
        # Try to import winsound on Windows
        self._winsound = None
        if self._is_windows:
            try:
                import winsound
                self._winsound = winsound
            except ImportError:
                pass
    
    def play_alert(self, alert: AlertEvent) -> bool:
        """
        Play audio alert for the given event.
        
        Returns True if alert was started, False if skipped.
        """
        if not self.enabled:
            return False
        
        # Check priority - higher priority (lower number) preempts
        if self._playing and alert.priority >= self._current_priority:
            return False
        
        # Stop current playback if any
        self.stop()
        
        # Start new playback
        self._current_priority = alert.priority
        self._stop_flag.clear()
        self._play_thread = threading.Thread(
            target=self._play_pattern,
            args=(alert.alert_type,),
            daemon=True
        )
        self._playing = True
        self._play_thread.start()
        
        return True
    
    def _play_pattern(self, alert_type: AlertType) -> None:
        """Play the sound pattern for an alert type."""
        pattern = self.ALERT_PATTERNS.get(alert_type, [(800, 200)])
        
        try:
            for freq, duration in pattern:
                if self._stop_flag.is_set():
                    break
                
                if freq == 0:
                    # Silent pause
                    time.sleep(duration / 1000.0)
                elif self._winsound:
                    # Windows beep
                    self._winsound.Beep(freq, duration)
                else:
                    # Fallback: just wait (no sound on non-Windows without additional setup)
                    time.sleep(duration / 1000.0)
        finally:
            self._playing = False
            self._current_priority = 999
    
    def stop(self) -> None:
        """Stop current playback."""
        self._stop_flag.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
        self._playing = False
        self._current_priority = 999
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing an alert."""
        return self._playing


def beep_alert(alert_type: AlertType) -> None:
    """
    Simple synchronous beep for an alert type.
    Use this for quick testing without threading.
    """
    if platform.system() != "Windows":
        print(f"[ALERT] {alert_type.display_name}")
        return
    
    try:
        import winsound
        patterns = AudioAlertManager.ALERT_PATTERNS.get(alert_type, [(800, 200)])
        for freq, duration in patterns:
            if freq > 0:
                winsound.Beep(freq, duration)
            else:
                time.sleep(duration / 1000.0)
    except Exception:
        print(f"[ALERT] {alert_type.display_name}")
