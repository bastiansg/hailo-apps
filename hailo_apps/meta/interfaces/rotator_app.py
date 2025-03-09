import numpy as np

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from hailo_apps.servos import ServoAngles, Servos
from pydantic import BaseModel, PositiveInt, NonNegativeInt, Field

from .picam_app import PicamApp, ImageSize


T = TypeVar("T", bound="RotatorApp")


class RotatorParams(BaseModel):
    update_angle: PositiveInt = Field(le=5, default=5)
    min_delta_x_angle: PositiveInt = Field(le=20, default=20)
    min_delta_y_angle: PositiveInt = Field(le=40, default=40)


class Centroid(BaseModel):
    x: NonNegativeInt
    y: NonNegativeInt


class RotatorApp(PicamApp["RotatorApp"], ABC, Generic[T]):
    def __init__(
        self,
        init_servo_angles: ServoAngles,
        rotator_params: RotatorParams,
    ):
        super().__init__(image_size=ImageSize())

        self.servos = Servos(init_servo_angles=init_servo_angles)
        self.rotator_params = rotator_params

        self.x_angle = init_servo_angles.x
        self.y_angle = init_servo_angles.y

    @abstractmethod
    def get_centroid(self, np_image: np.ndarray) -> Centroid | None:
        pass

    def get_new_angle(
        self,
        axis_angle: int,
        axis_length: int,
        centroid_coord: int,
        min_delta_angle: int,
    ) -> int:
        axis_delta = (axis_length // 2) - centroid_coord
        if abs(axis_delta) <= min_delta_angle:
            return axis_angle

        return (
            (axis_angle + self.rotator_params.update_angle)
            if axis_delta > 0
            else (axis_angle - self.rotator_params.update_angle)
        )

    def on_frame(self, np_image: np.ndarray) -> None:
        centroid = self.get_centroid(np_image=np_image)
        if centroid is None:
            return

        new_x_angle = self.get_new_angle(
            axis_angle=self.x_angle,
            axis_length=self.image_size.width,
            centroid_coord=centroid.x,
            min_delta_angle=self.rotator_params.min_delta_x_angle,
        )

        new_y_angle = self.get_new_angle(
            axis_angle=self.y_angle,
            axis_length=self.image_size.height,
            centroid_coord=centroid.y,
            min_delta_angle=self.rotator_params.min_delta_y_angle,
        )

        self.servos.set_angles(
            servo_angles=ServoAngles(
                x=new_x_angle,
                y=new_y_angle,
            ),
        )

        self.x_angle = new_x_angle
        self.y_angle = new_y_angle
