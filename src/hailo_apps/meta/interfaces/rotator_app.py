import numpy as np

from collections import deque
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from hailo_apps.servos import ServoAngles, Servos
from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    Field,
    ConfigDict,
    StrictInt,
    model_validator,
)

from .picam_app import PicamApp, ImageSize


T = TypeVar("T", bound="RotatorApp")


class RotatorParams(BaseModel):
    update_angle: PositiveInt = Field(le=5, default=5)
    min_delta_x_angle: PositiveInt = 100
    min_delta_y_angle: PositiveInt = 120
    # min_delta_x_angle: PositiveInt = 80
    # min_delta_y_angle: PositiveInt = 100
    min_x_angle: NonNegativeInt = Field(lt=180, default=0)
    max_x_angle: NonNegativeInt = Field(le=180, default=180)
    min_y_angle: NonNegativeInt = Field(lt=180, default=20)
    max_y_angle: NonNegativeInt = Field(le=180, default=180)
    y_quadrant_count: PositiveInt = 10
    y_target_quadrant: NonNegativeInt = 6

    @model_validator(mode="after")
    def validate_y_target_quadrant(self) -> "RotatorParams":
        if self.y_target_quadrant > self.y_quadrant_count:
            raise ValueError(
                "y_target_quadrant must not exceed y_quadrant_count"
            )

        return self


class Centroid(BaseModel):
    x: NonNegativeInt
    y: NonNegativeInt


class HistoryItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    np_image: np.ndarray
    centroid: Centroid | None = None
    x_delta: StrictInt = 0
    y_delta: StrictInt = 0


class RotatorApp(PicamApp["RotatorApp"], ABC, Generic[T]):  # type: ignore
    def __init__(
        self,
        model_url: str,
        image_size: ImageSize,
        init_servo_angles: ServoAngles,
        rotator_params: RotatorParams,
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
        history_length: int = 0,
    ):
        super().__init__(
            model_url=model_url,
            image_size=image_size,
            debug_mode=debug_mode,
            debug_path=debug_path,
        )

        self.servos = Servos(init_servo_angles=init_servo_angles)
        self.rotator_params = rotator_params

        self.x_angle = init_servo_angles.x
        self.y_angle = init_servo_angles.y

        self.history = deque(maxlen=history_length)

    @abstractmethod
    def get_centroid(self, np_image: np.ndarray) -> Centroid | None:
        pass

    def get_new_angle(
        self,
        axis_angle: int,
        axis_length: int,
        centroid_coord: int,
        min_delta_angle: int,
        min_angle: int,
        max_angle: int,
        target_coord: int | None = None,
    ) -> int:
        if target_coord is None:
            target_coord = axis_length // 2

        axis_delta = target_coord - centroid_coord
        if abs(axis_delta) <= min_delta_angle:
            return axis_angle

        new_angle = (
            (axis_angle + self.rotator_params.update_angle)
            if axis_delta > 0
            else (axis_angle - self.rotator_params.update_angle)
        )

        return max(min_angle, min(new_angle, max_angle))

    def on_frame(self, np_image: np.ndarray) -> None:
        centroid = self.get_centroid(np_image=np_image)

        if centroid is None:
            self.history.append(
                HistoryItem(
                    np_image=np_image,
                    centroid=centroid,
                    x_delta=self.rotator_params.update_angle,
                    y_delta=self.rotator_params.update_angle,
                )
            )

            return

        new_x_angle = self.get_new_angle(
            axis_angle=self.x_angle,
            axis_length=self.image_size.width,
            centroid_coord=centroid.x,
            min_delta_angle=self.rotator_params.min_delta_x_angle,
            min_angle=self.rotator_params.min_x_angle,
            max_angle=self.rotator_params.max_x_angle,
        )

        target_coord = (
            self.image_size.height
            * (
                self.rotator_params.y_quadrant_count
                - self.rotator_params.y_target_quadrant
            )
            // self.rotator_params.y_quadrant_count
        )

        new_y_angle = self.get_new_angle(
            axis_angle=self.y_angle,
            axis_length=self.image_size.height,
            centroid_coord=centroid.y,
            min_delta_angle=self.rotator_params.min_delta_y_angle,
            min_angle=self.rotator_params.min_y_angle,
            max_angle=self.rotator_params.max_y_angle,
            target_coord=target_coord,
        )

        self.servos.set_angles(
            servo_angles=ServoAngles(
                x=new_x_angle,
                y=new_y_angle,
            ),
        )

        self.history.append(
            HistoryItem(
                np_image=np_image,
                centroid=centroid,
                x_delta=abs(self.x_angle - new_x_angle),
                y_delta=abs(self.y_angle - new_y_angle),
            )
        )

        self.x_angle = new_x_angle
        self.y_angle = new_y_angle
