"""
GPIO Status LED Controller Module.

Controls status indicator LEDs on Raspberry Pi GPIO pins.

Hardware Connection:
- GPIO 17 → System Running LED (green recommended)
- GPIO 27 → Alert Active LED (red recommended)

Each LED should have appropriate current-limiting resistor (330Ω-1kΩ).
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LEDState(Enum):
    """LED state."""
    OFF = 0
    ON = 1
    BLINK_SLOW = 2
    BLINK_FAST = 3


@dataclass
class GPIOConfig:
    """GPIO pin configuration."""
    system_led_pin: int = 17  # System running indicator
    alert_led_pin: int = 27   # Alert active indicator
    enabled: bool = True


class GPIOStatusController:
    """
    Controls GPIO status LEDs on Raspberry Pi.
    
    Features:
    - System running LED (GPIO 17) - on when system initialized
    - Alert LED (GPIO 27) - on during active alerts
    - Blinking patterns for different states
    - Safe setup and cleanup
    - Graceful fallback when GPIO unavailable
    
    Usage:
        gpio = GPIOStatusController(system_pin=17, alert_pin=27)
        if gpio.initialize():
            gpio.set_system_led(True)  # System started
            gpio.set_alert_led(True)   # Alert active
            # ... 
            gpio.cleanup()
    """
    
    def __init__(
        self,
        system_pin: int = 17,
        alert_pin: int = 27,
        enabled: bool = True,
        blink_slow_hz: float = 1.0,
        blink_fast_hz: float = 4.0,
    ):
        """
        Initialize GPIO controller.
        
        Args:
            system_pin: BCM pin number for system LED
            alert_pin: BCM pin number for alert LED
            enabled: If False, all operations are no-ops
            blink_slow_hz: Slow blink frequency
            blink_fast_hz: Fast blink frequency
        """
        self.system_pin = system_pin
        self.alert_pin = alert_pin
        self.enabled = enabled
        self.blink_slow_period = 1.0 / blink_slow_hz
        self.blink_fast_period = 1.0 / blink_fast_hz
        
        # State
        self._gpio = None
        self._initialized = False
        self._system_state = LEDState.OFF
        self._alert_state = LEDState.OFF
        
        # Blink thread
        self._blink_running = False
        self._blink_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    @property
    def is_available(self) -> bool:
        """Check if GPIO is available."""
        return self._initialized and self._gpio is not None
    
    def initialize(self) -> bool:
        """
        Initialize GPIO pins.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.enabled:
            logger.info("GPIO controller disabled")
            return True
        
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            
            # Set BCM mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup output pins
            GPIO.setup(self.system_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.alert_pin, GPIO.OUT, initial=GPIO.LOW)
            
            self._initialized = True
            
            # Start blink thread
            self._start_blink_thread()
            
            logger.info(f"GPIO initialized: system={self.system_pin}, alert={self.alert_pin}")
            return True
            
        except ImportError:
            logger.warning("RPi.GPIO not available - GPIO functions disabled")
            return False
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        # Stop blink thread
        self._blink_running = False
        if self._blink_thread is not None:
            self._blink_thread.join(timeout=1.0)
            self._blink_thread = None
        
        # Turn off LEDs and cleanup GPIO
        if self._gpio is not None and self._initialized:
            try:
                self._gpio.output(self.system_pin, self._gpio.LOW)
                self._gpio.output(self.alert_pin, self._gpio.LOW)
                self._gpio.cleanup([self.system_pin, self.alert_pin])
            except Exception as e:
                logger.warning(f"GPIO cleanup warning: {e}")
        
        self._initialized = False
        logger.info("GPIO cleaned up")
    
    def set_system_led(self, on: bool = True, blink: bool = False) -> None:
        """
        Set system LED state.
        
        Args:
            on: True to turn on, False to turn off
            blink: If True and on, LED will blink slowly
        """
        with self._lock:
            if on:
                self._system_state = LEDState.BLINK_SLOW if blink else LEDState.ON
            else:
                self._system_state = LEDState.OFF
        
        if not blink and self._gpio is not None and self._initialized:
            try:
                self._gpio.output(self.system_pin, self._gpio.HIGH if on else self._gpio.LOW)
            except Exception as e:
                logger.warning(f"Failed to set system LED: {e}")
    
    def set_alert_led(self, on: bool = True, blink: bool = False) -> None:
        """
        Set alert LED state.
        
        Args:
            on: True to turn on, False to turn off
            blink: If True and on, LED will blink fast
        """
        with self._lock:
            if on:
                self._alert_state = LEDState.BLINK_FAST if blink else LEDState.ON
            else:
                self._alert_state = LEDState.OFF
        
        if not blink and self._gpio is not None and self._initialized:
            try:
                self._gpio.output(self.alert_pin, self._gpio.HIGH if on else self._gpio.LOW)
            except Exception as e:
                logger.warning(f"Failed to set alert LED: {e}")
    
    def pulse_alert(self, duration_ms: int = 200) -> None:
        """
        Briefly pulse the alert LED.
        
        Args:
            duration_ms: Pulse duration in milliseconds
        """
        if not self.is_available:
            return
        
        def _pulse():
            try:
                self._gpio.output(self.alert_pin, self._gpio.HIGH)
                time.sleep(duration_ms / 1000.0)
                with self._lock:
                    if self._alert_state == LEDState.OFF:
                        self._gpio.output(self.alert_pin, self._gpio.LOW)
            except Exception as e:
                logger.warning(f"Alert pulse failed: {e}")
        
        threading.Thread(target=_pulse, daemon=True).start()
    
    def _start_blink_thread(self) -> None:
        """Start background thread for blinking patterns."""
        self._blink_running = True
        self._blink_thread = threading.Thread(target=self._blink_loop, daemon=True)
        self._blink_thread.start()
    
    def _blink_loop(self) -> None:
        """Background thread for LED blinking."""
        last_toggle_system = 0
        last_toggle_alert = 0
        system_on = False
        alert_on = False
        
        while self._blink_running:
            try:
                now = time.monotonic()
                
                with self._lock:
                    system_state = self._system_state
                    alert_state = self._alert_state
                
                # Handle system LED blinking
                if system_state == LEDState.BLINK_SLOW:
                    if now - last_toggle_system >= self.blink_slow_period / 2:
                        system_on = not system_on
                        last_toggle_system = now
                        if self._gpio and self._initialized:
                            self._gpio.output(
                                self.system_pin, 
                                self._gpio.HIGH if system_on else self._gpio.LOW
                            )
                
                # Handle alert LED blinking
                if alert_state == LEDState.BLINK_FAST:
                    if now - last_toggle_alert >= self.blink_fast_period / 2:
                        alert_on = not alert_on
                        last_toggle_alert = now
                        if self._gpio and self._initialized:
                            self._gpio.output(
                                self.alert_pin,
                                self._gpio.HIGH if alert_on else self._gpio.LOW
                            )
                
                time.sleep(0.05)  # 50ms tick
                
            except Exception as e:
                logger.debug(f"Blink loop error: {e}")
                time.sleep(0.1)


class StubGPIOController:
    """
    Stub GPIO controller for testing without hardware.
    """
    
    def __init__(self, system_pin: int = 17, alert_pin: int = 27, **kwargs):
        self.system_pin = system_pin
        self.alert_pin = alert_pin
        self._system_led = False
        self._alert_led = False
        self._initialized = False
    
    @property
    def is_available(self) -> bool:
        return self._initialized
    
    def initialize(self) -> bool:
        self._initialized = True
        logger.info("Stub GPIO controller initialized")
        return True
    
    def cleanup(self) -> None:
        self._system_led = False
        self._alert_led = False
        self._initialized = False
    
    def set_system_led(self, on: bool = True, blink: bool = False) -> None:
        self._system_led = on
        logger.debug(f"Stub: System LED {'ON' if on else 'OFF'}")
    
    def set_alert_led(self, on: bool = True, blink: bool = False) -> None:
        self._alert_led = on
        logger.debug(f"Stub: Alert LED {'ON' if on else 'OFF'}")
    
    def pulse_alert(self, duration_ms: int = 200) -> None:
        logger.debug(f"Stub: Alert pulse {duration_ms}ms")
    
    @property
    def system_led_state(self) -> bool:
        """Get system LED state (for testing)."""
        return self._system_led
    
    @property
    def alert_led_state(self) -> bool:
        """Get alert LED state (for testing)."""
        return self._alert_led


def create_gpio_controller(
    enabled: bool = True,
    system_pin: int = 17,
    alert_pin: int = 27,
    **kwargs,
) -> GPIOStatusController | StubGPIOController:
    """
    Factory function to create GPIO controller.
    
    Args:
        enabled: If False, returns StubGPIOController
        system_pin: BCM pin for system LED
        alert_pin: BCM pin for alert LED
        
    Returns:
        GPIOStatusController if available, StubGPIOController otherwise
    """
    if not enabled:
        logger.info("GPIO controller disabled, using stub")
        return StubGPIOController(system_pin=system_pin, alert_pin=alert_pin)
    
    # Check if RPi.GPIO is available
    try:
        import RPi.GPIO
        return GPIOStatusController(
            system_pin=system_pin,
            alert_pin=alert_pin,
            enabled=enabled,
            **kwargs,
        )
    except ImportError:
        logger.warning("RPi.GPIO not available, using stub controller")
        return StubGPIOController(system_pin=system_pin, alert_pin=alert_pin)
