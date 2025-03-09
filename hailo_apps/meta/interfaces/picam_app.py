import numpy as np

from abc import ABC, abstractmethod
from typing import Self, TypeVar, Generic

from picamera2 import Picamera2
from pydantic import BaseModel, PositiveInt, model_validator


T = TypeVar("T", bound="PicamApp")


class ImageSize(BaseModel):
    width: PositiveInt = 640
    height: PositiveInt = 480

    @model_validator(mode="after")
    def check_passwords_match(self) -> Self:
        if self.width / self.height == 4 / 3:
            return self

        raise ValueError("Aspect ratio must be 4:3")


class PicamApp(ABC, Generic[T]):
    def __init__(
        self,
        image_size: ImageSize,
    ):
        self.image_size = image_size
        self.picam = self.get_picam(image_size=image_size)
        self.is_active = True

    @staticmethod
    def get_picam(image_size: ImageSize) -> Picamera2:
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={
                "size": (
                    image_size.width,
                    image_size.height,
                ),
            }
        )

        picam.configure(config)
        return picam

    @abstractmethod
    def on_frame(self, np_image: np.ndarray) -> None:
        pass

    def run(self) -> None:
        self.picam.start()
        while self.is_active:
            np_image = self.picam.capture_array()
            np_image = np_image[:, :, :3]  # Remove alpha channel.
            self.on_frame(np_image=np_image)
