"""
Sensors module for the Vehicle Safety Alert System.

Provides optional IR distance sensor integration.
"""

from .ir_distance import IRDistanceSensor, StubIRSensor, create_ir_sensor

__all__ = [
    "IRDistanceSensor",
    "StubIRSensor",
    "create_ir_sensor",
]
