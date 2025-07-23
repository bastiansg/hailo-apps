import numpy as np

from PIL import Image
from time import sleep
from threading import Lock, Event
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from picamera2 import Picamera2
from pydantic import BaseModel, PositiveInt

from common.utils.path import create_path
from common.utils.threading import threaded

from .hailo_app import HailoApp


T = TypeVar("T", bound="PicamApp")


class ImageSize(BaseModel):
    width: PositiveInt = 640
    height: PositiveInt = 640


class PicamApp(HailoApp["PicamApp"], ABC, Generic[T]):
    def __init__(
        self,
        model_url: str,
        image_size: ImageSize,
        rotation_180: bool = True,
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
    ):
        super().__init__(
            model_url=model_url,
        )

        self.image_size = image_size
        self.rotation_180 = rotation_180
        self.debug_mode = debug_mode

        create_path(path=debug_path)
        self.debug_image_path = f"{debug_path}/debug-image.jpg"

        self.picam = self.get_picam(image_size=image_size)

        self.mutex = Lock()
        self.stop_event = Event()

    def __del__(self):
        self.picam.stop()
        self.picam.close()

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

    @threaded
    def run(self) -> None:
        with self.mutex:
            self.stop_event.clear()
            self.picam.start()

            while not self.stop_event.is_set():
                np_image = self.picam.capture_array()
                np_image = np_image[:, :, :3]  # Remove alpha channel.

                if self.rotation_180:
                    np_image = np_image[::-1, ::-1]  # 180-degree rotation.

                if self.debug_mode:
                    pil_image = Image.fromarray(np_image)
                    pil_image.save(self.debug_image_path)
                    break

                self.on_frame(np_image=np_image)

            self.picam.stop()

    def stop(self) -> None:
        self.stop_event.set()
        sleep(0.1)
