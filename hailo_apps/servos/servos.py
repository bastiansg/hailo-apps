from adafruit_servokit import ServoKit
from pydantic import BaseModel, NonNegativeInt, Field


class ServoAngles(BaseModel):
    x: NonNegativeInt = Field(le=180, default=103)
    y: NonNegativeInt = Field(le=180, default=80)


class Servos:
    def __init__(self, init_servo_angles: ServoAngles):
        self.servo_kit = ServoKit(channels=16)
        self.set_angles(servo_angles=init_servo_angles)

    def set_angles(self, servo_angles: ServoAngles) -> None:
        self.servo_kit.servo[0].angle = servo_angles.x
        self.servo_kit.servo[1].angle = servo_angles.y
