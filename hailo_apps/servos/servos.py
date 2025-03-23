from adafruit_servokit import ServoKit
from pydantic import BaseModel, NonNegativeInt, field_validator


class ServoAngles(BaseModel):
    x: NonNegativeInt = 103
    y: NonNegativeInt = 80

    @field_validator("x", "y", mode="before")
    def angle_validator(cls, value: int) -> int:
        return max(0, min(value, 180))


class Servos:
    def __init__(self, init_servo_angles: ServoAngles | None = None):
        self.servo_kit = ServoKit(channels=16)
        if init_servo_angles is not None:
            self.set_angles(servo_angles=init_servo_angles)

    def set_angles(self, servo_angles: ServoAngles) -> None:
        self.servo_kit.servo[0].angle = servo_angles.x
        self.servo_kit.servo[1].angle = servo_angles.y
